#include "backend_cuda.h"

#include <cuda_runtime.h>
#include <mma.h>
#include <cuda_fp8.h>

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <mutex>

struct ColiCudaTensor {
    void *weights;
    float *scales;
    size_t weight_bytes;
    int fmt, I, O, device;
    int tracked;
    // Zero-copy on unified memory (GB10): `weights`/`scales` point directly at the
    // host (RAM) buffers — no cudaMalloc, no memcpy, no device-side offset→signed
    // conversion. int4 stays offset-binary, so kernels must read it with off=1.
    int wrapped;
};

typedef struct {
    int device;
    int compute_major,compute_minor;
    float *x, *y, *gate, *up;
    size_t x_cap, y_cap, gate_cap, up_cap;
    uint8_t *qx; float *qscale;
    size_t qx_cap, qscale_cap;
    float *host_x,*host_y; size_t host_x_cap,host_y_cap;
    /* Device scratch for expert weights, so the kernel reads clean device memory
     * instead of zero-copy from freshly-pread (dirty, coherence-heavy) host pages. */
    uint8_t *ewg,*ewu,*ewd; size_t ewg_cap,ewu_cap,ewd_cap;
    float *esg,*esu,*esd; size_t esg_cap,esu_cap,esd_cap;
    float *aq,*al,*ar,*ac; size_t aq_cap,al_cap,ar_cap,ac_cap;
    void *asel,*acnt; size_t asel_cap,acnt_cap;  /* DSA sparse-attention selection */
    void *aqa,*akb,*amsk; size_t aqa_cap,akb_cap,amsk_cap;  /* tensor-core sparse attn: QA/KB fp16 + per-query key bitmask */
    float *pipe_buf[24]; size_t pipe_cap[24];   /* scratch persistenti del resident pipeline */
    cudaStream_t stream;
    void *group_desc; size_t group_desc_cap;
    size_t tensor_count, tensor_bytes;
} DeviceContext;

typedef struct {
    const void *g,*u,*d; const float *gs,*us,*ds;
    int gf,uf,df,rows,offset,wrapped;
} GroupDesc;

static DeviceContext g_ctx[COLI_CUDA_MAX_DEVICES];
static int g_nctx;
static uint64_t g_group_calls,g_group_experts,g_group_rows;
static double g_group_h2d_ms,g_group_kernel_ms,g_group_d2h_ms;
static std::mutex g_group_stats_mu;

static int cuda_ok(cudaError_t err, const char *what) {
    if (err == cudaSuccess) return 1;
    std::fprintf(stderr, "[CUDA] %s: %s\n", what, cudaGetErrorString(err));
    return 0;
}

static DeviceContext *find_ctx(int device) {
    for (int i = 0; i < g_nctx; i++) if (g_ctx[i].device == device) return &g_ctx[i];
    return nullptr;
}

/* cudaSetDevice on every call doubles expert-matmul time on 2 GPUs when the
 * serial expert loop alternates devices (measured on RTX 5090 + 4090: 14.3s
 * -> 25.4s per 32 tokens). The current device is per-thread in the CUDA
 * runtime, so a thread-local cache skips the redundant switches. */
static thread_local int g_current_device = -1;

static int select_ctx(DeviceContext *ctx) {
    if (!ctx) return 0;
    if (g_current_device == ctx->device) return 1;
    if (!cuda_ok(cudaSetDevice(ctx->device), "select device")) return 0;
    g_current_device = ctx->device;
    return 1;
}

__host__ __device__ static size_t row_bytes(int fmt, int I) {
    if (fmt == 0) return (size_t)I * sizeof(float);
    if (fmt == 1) return (size_t)I;
    if (fmt == 2) return (size_t)(I + 1) / 2;
    if (fmt == 3) return (size_t)(I + 3) / 4;
    if (fmt == 4) return (size_t)I;          // e4m3 fp8: 1 byte/weight
    return 0;
}

// Decode one e4m3 (fp8) byte to float via the hardware conversion.
__device__ __forceinline__ static float e4m3f(uint8_t b) {
    __half_raw hr = __nv_cvt_fp8_to_halfraw((__nv_fp8_storage_t)b, __NV_E4M3);
    return __half2float(*reinterpret_cast<__half *>(&hr));
}

// `off`=1 reads int4 as offset-binary (value = nibble − 8, the on-disk / host
// format used by zero-copy wrapped tensors); off=0 reads the signed two's-complement
// form produced by `offset_to_signed_s4` after a device copy. Both yield the same
// value (signed_interp(raw^8) == raw−8); the flag just picks which representation
// the bytes are currently in. Default off=0 so existing call sites are unchanged.
__device__ static float weight_at(const void *weights, int fmt, size_t row, int i, int off=0) {
    const uint8_t *base = static_cast<const uint8_t *>(weights) + row;
    if (fmt == 0) return reinterpret_cast<const float *>(base)[i];
    if (fmt == 1) return static_cast<float>(reinterpret_cast<const int8_t *>(base)[i]);
    if (fmt == 4) return e4m3f(base[i]);      // e4m3 fp8; per-row scale applied by caller
    const uint8_t *q = base;
    if (fmt == 2) {
        uint8_t v = q[i >> 1];
        int n=(i&1)?(v>>4):(v&15);
        return off ? static_cast<float>(n - 8) : static_cast<float>(n&8?n-16:n);
    }
    uint8_t v = q[i >> 2];
    return static_cast<float>(((v >> ((i & 3) * 2)) & 3) - 2);
}

__global__ static void offset_to_signed_s4(uint8_t *q,size_t n){
    size_t i=(size_t)blockIdx.x*blockDim.x+threadIdx.x;if(i<n)q[i]^=0x88;
}

__global__ static void quant_matmul(float *y, const float *x, const void *weights,
                                    const float *scales, int fmt, int S, int I, int O,
                                    size_t rb, int off) {
    int o = blockIdx.x;
    int s = blockIdx.y;
    float sum = 0.0f;
    size_t row = (size_t)o * rb;
    const float *xs = x + (size_t)s * I;
    for (int i = threadIdx.x; i < I; i += blockDim.x)
        sum += xs[i] * weight_at(weights, fmt, row, i, off);

    __shared__ float partial[256];
    partial[threadIdx.x] = sum;
    __syncthreads();
    for (int n = blockDim.x >> 1; n; n >>= 1) {
        if (threadIdx.x < n) partial[threadIdx.x] += partial[threadIdx.x + n];
        __syncthreads();
    }
    if (!threadIdx.x)
        y[(size_t)s * O + o] = partial[0] * (fmt ? scales[o] : 1.0f);
}

__global__ static void silu_mul(float *gate, const float *up, size_t n) {
    size_t i = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) {
        float v = gate[i];
        gate[i] = (v / (1.0f + expf(-v))) * up[i];
    }
}

/* Four warps share one A tile and compute 16x64 outputs.  This matters for
 * prefill: the first prototype reloaded/converter A once per 16 output cols. */
__global__ static void w4a16_matmul(float *y,const float *x,const uint8_t *w,
                                    const float *scale,int M,int K,int N){
#if __CUDA_ARCH__ >= 700
    using namespace nvcuda;int warp=threadIdx.x>>5,lane=threadIdx.x&31;
    int m0=blockIdx.y*16,n0=blockIdx.x*64+warp*16;
    __shared__ __half ah[256],bh[4][256];
    wmma::fragment<wmma::accumulator,16,16,16,float> acc;wmma::fill_fragment(acc,0.f);
    size_t rb=(size_t)(K+1)/2;
    for(int k0=0;k0<K;k0+=16){
        for(int z=threadIdx.x;z<256;z+=blockDim.x){
            int m=z/16,k=z%16,gm=m0+m,gk=k0+k;
            ah[z]=(gm<M&&gk<K)?__float2half(x[(size_t)gm*K+gk]):__float2half(0.f);
        }
        for(int z=lane;z<256;z+=32){
            int n=z/16,gk=k0+(z%16),gn=n0+n;float v=0.f;
            if(gn<N&&gk<K){uint8_t q=w[(size_t)gn*rb+(gk>>1)];int a=(gk&1)?q>>4:q&15;
                v=(float)(a&8?a-16:a)*scale[gn];}
            bh[warp][z]=__float2half(v);           /* [Ntile,Ktile] == B col-major */
        }
        __syncthreads();
        wmma::fragment<wmma::matrix_a,16,16,16,__half,wmma::row_major> af;
        wmma::fragment<wmma::matrix_b,16,16,16,__half,wmma::col_major> bf;
        wmma::load_matrix_sync(af,ah,16);wmma::load_matrix_sync(bf,bh[warp],16);
        wmma::mma_sync(acc,af,bf,acc);__syncthreads();
    }
    __shared__ float out[4][256];wmma::store_matrix_sync(out[warp],acc,16,wmma::mem_row_major);__syncwarp();
    for(int z=lane;z<256;z+=32){int m=z/16,n=z%16;
        if(m0+m<M&&n0+n<N)y[(size_t)(m0+m)*N+n0+n]=out[warp][z];}
#endif
}

/* Gate and up use the same input.  Eight warps compute both 16x64 projections
 * while sharing the FP32->FP16 conversion of A. */
__global__ static void w4a16_gate_up(float *gate,float *up,const float *x,
        const uint8_t *gw,const uint8_t *uw,const float *gs,const float *us,
        int M,int K,int N){
#if __CUDA_ARCH__ >= 700
    using namespace nvcuda;int warp=threadIdx.x>>5,lane=threadIdx.x&31,which=warp&1,tile=warp>>1;
    int m0=blockIdx.y*16,n0=blockIdx.x*64+tile*16;const uint8_t *w=which?uw:gw;
    const float *scale=which?us:gs;float *y=which?up:gate;size_t rb=(size_t)(K+1)/2;
    __shared__ __half ah[256],bh[8][256];
    wmma::fragment<wmma::accumulator,16,16,16,float> acc;wmma::fill_fragment(acc,0.f);
    for(int k0=0;k0<K;k0+=16){
        for(int z=threadIdx.x;z<256;z+=blockDim.x){int m=z/16,k=z%16,gm=m0+m,gk=k0+k;
            ah[z]=(gm<M&&gk<K)?__float2half(x[(size_t)gm*K+gk]):__float2half(0.f);}
        for(int z=lane;z<256;z+=32){int n=z/16,gk=k0+(z%16),gn=n0+n;float v=0.f;
            if(gn<N&&gk<K){uint8_t q=w[(size_t)gn*rb+(gk>>1)];int a=(gk&1)?q>>4:q&15;
                v=(float)(a&8?a-16:a)*scale[gn];}bh[warp][z]=__float2half(v);}
        __syncthreads();
        wmma::fragment<wmma::matrix_a,16,16,16,__half,wmma::row_major> af;
        wmma::fragment<wmma::matrix_b,16,16,16,__half,wmma::col_major> bf;
        wmma::load_matrix_sync(af,ah,16);wmma::load_matrix_sync(bf,bh[warp],16);
        wmma::mma_sync(acc,af,bf,acc);__syncthreads();
    }
    __shared__ float out[8][256];wmma::store_matrix_sync(out[warp],acc,16,wmma::mem_row_major);__syncwarp();
    for(int z=lane;z<256;z+=32){int m=z/16,n=z%16;
        if(m0+m<M&&n0+n<N)y[(size_t)(m0+m)*N+n0+n]=out[warp][z];}
#endif
}

/* FP8 (e4m3) tiled tensor-core expert matmuls — clones of w4a16_* with the weight
 * decode swapped int4 -> e4m3 (1 byte/weight, direct K stride). Weights are FP8,
 * activations FP16, MMA runs in f16 (W8A16). This is the tiled path that replaces the
 * naive quant_matmul's M-fold weight re-reads. */
__global__ static void fp8a16_matmul(float *y,const float *x,const uint8_t *w,
                                    const float *scale,int M,int K,int N){
#if __CUDA_ARCH__ >= 700
    using namespace nvcuda;int warp=threadIdx.x>>5,lane=threadIdx.x&31;
    int m0=blockIdx.y*16,n0=blockIdx.x*64+warp*16;
    __shared__ __half ah[256],bh[4][256];
    wmma::fragment<wmma::accumulator,16,16,16,float> acc;wmma::fill_fragment(acc,0.f);
    for(int k0=0;k0<K;k0+=16){
        for(int z=threadIdx.x;z<256;z+=blockDim.x){
            int m=z/16,k=z%16,gm=m0+m,gk=k0+k;
            ah[z]=(gm<M&&gk<K)?__float2half(x[(size_t)gm*K+gk]):__float2half(0.f);
        }
        for(int z=lane;z<256;z+=32){
            int n=z/16,gk=k0+(z%16),gn=n0+n;float v=0.f;
            if(gn<N&&gk<K) v=e4m3f(w[(size_t)gn*K+gk])*scale[gn];
            bh[warp][z]=__float2half(v);
        }
        __syncthreads();
        wmma::fragment<wmma::matrix_a,16,16,16,__half,wmma::row_major> af;
        wmma::fragment<wmma::matrix_b,16,16,16,__half,wmma::col_major> bf;
        wmma::load_matrix_sync(af,ah,16);wmma::load_matrix_sync(bf,bh[warp],16);
        wmma::mma_sync(acc,af,bf,acc);__syncthreads();
    }
    __shared__ float out[4][256];wmma::store_matrix_sync(out[warp],acc,16,wmma::mem_row_major);__syncwarp();
    for(int z=lane;z<256;z+=32){int m=z/16,n=z%16;
        if(m0+m<M&&n0+n<N)y[(size_t)(m0+m)*N+n0+n]=out[warp][z];}
#endif
}

__global__ static void fp8a16_gate_up(float *gate,float *up,const float *x,
        const uint8_t *gw,const uint8_t *uw,const float *gs,const float *us,
        int M,int K,int N){
#if __CUDA_ARCH__ >= 700
    using namespace nvcuda;int warp=threadIdx.x>>5,lane=threadIdx.x&31,which=warp&1,tile=warp>>1;
    int m0=blockIdx.y*16,n0=blockIdx.x*64+tile*16;const uint8_t *w=which?uw:gw;
    const float *scale=which?us:gs;float *y=which?up:gate;
    __shared__ __half ah[256],bh[8][256];
    wmma::fragment<wmma::accumulator,16,16,16,float> acc;wmma::fill_fragment(acc,0.f);
    for(int k0=0;k0<K;k0+=16){
        for(int z=threadIdx.x;z<256;z+=blockDim.x){int m=z/16,k=z%16,gm=m0+m,gk=k0+k;
            ah[z]=(gm<M&&gk<K)?__float2half(x[(size_t)gm*K+gk]):__float2half(0.f);}
        for(int z=lane;z<256;z+=32){int n=z/16,gk=k0+(z%16),gn=n0+n;float v=0.f;
            if(gn<N&&gk<K) v=e4m3f(w[(size_t)gn*K+gk])*scale[gn];
            bh[warp][z]=__float2half(v);}
        __syncthreads();
        wmma::fragment<wmma::matrix_a,16,16,16,__half,wmma::row_major> af;
        wmma::fragment<wmma::matrix_b,16,16,16,__half,wmma::col_major> bf;
        wmma::load_matrix_sync(af,ah,16);wmma::load_matrix_sync(bf,bh[warp],16);
        wmma::mma_sync(acc,af,bf,acc);__syncthreads();
    }
    __shared__ float out[8][256];wmma::store_matrix_sync(out[warp],acc,16,wmma::mem_row_major);__syncwarp();
    for(int z=lane;z<256;z+=32){int m=z/16,n=z%16;
        if(m0+m<M&&n0+n<N)y[(size_t)(m0+m)*N+n0+n]=out[warp][z];}
#endif
}

/* int8 (W8A16) tiled tensor-core matmuls — clones of fp8a16_* with the weight decode
 * swapped e4m3 -> signed int8 (1 byte/weight, direct K stride). For the shared expert /
 * resident int8 weights that ran on the naive quant_matmul (nsys: 60% of GPU kernel
 * time from its S-fold weight re-reads). */
__global__ static void i8a16_matmul(float *y,const float *x,const uint8_t *w,
                                    const float *scale,int M,int K,int N){
#if __CUDA_ARCH__ >= 700
    using namespace nvcuda;int warp=threadIdx.x>>5,lane=threadIdx.x&31;
    int m0=blockIdx.y*16,n0=blockIdx.x*64+warp*16;
    __shared__ __half ah[256],bh[4][256];
    wmma::fragment<wmma::accumulator,16,16,16,float> acc;wmma::fill_fragment(acc,0.f);
    for(int k0=0;k0<K;k0+=16){
        for(int z=threadIdx.x;z<256;z+=blockDim.x){
            int m=z/16,k=z%16,gm=m0+m,gk=k0+k;
            ah[z]=(gm<M&&gk<K)?__float2half(x[(size_t)gm*K+gk]):__float2half(0.f);
        }
        for(int z=lane;z<256;z+=32){
            int n=z/16,gk=k0+(z%16),gn=n0+n;float v=0.f;
            if(gn<N&&gk<K) v=(float)((const signed char*)w)[(size_t)gn*K+gk]*scale[gn];
            bh[warp][z]=__float2half(v);
        }
        __syncthreads();
        wmma::fragment<wmma::matrix_a,16,16,16,__half,wmma::row_major> af;
        wmma::fragment<wmma::matrix_b,16,16,16,__half,wmma::col_major> bf;
        wmma::load_matrix_sync(af,ah,16);wmma::load_matrix_sync(bf,bh[warp],16);
        wmma::mma_sync(acc,af,bf,acc);__syncthreads();
    }
    __shared__ float out[4][256];wmma::store_matrix_sync(out[warp],acc,16,wmma::mem_row_major);__syncwarp();
    for(int z=lane;z<256;z+=32){int m=z/16,n=z%16;
        if(m0+m<M&&n0+n<N)y[(size_t)(m0+m)*N+n0+n]=out[warp][z];}
#endif
}

__global__ static void i8a16_gate_up(float *gate,float *up,const float *x,
        const uint8_t *gw,const uint8_t *uw,const float *gs,const float *us,
        int M,int K,int N){
#if __CUDA_ARCH__ >= 700
    using namespace nvcuda;int warp=threadIdx.x>>5,lane=threadIdx.x&31,which=warp&1,tile=warp>>1;
    int m0=blockIdx.y*16,n0=blockIdx.x*64+tile*16;const uint8_t *w=which?uw:gw;
    const float *scale=which?us:gs;float *y=which?up:gate;
    __shared__ __half ah[256],bh[8][256];
    wmma::fragment<wmma::accumulator,16,16,16,float> acc;wmma::fill_fragment(acc,0.f);
    for(int k0=0;k0<K;k0+=16){
        for(int z=threadIdx.x;z<256;z+=blockDim.x){int m=z/16,k=z%16,gm=m0+m,gk=k0+k;
            ah[z]=(gm<M&&gk<K)?__float2half(x[(size_t)gm*K+gk]):__float2half(0.f);}
        for(int z=lane;z<256;z+=32){int n=z/16,gk=k0+(z%16),gn=n0+n;float v=0.f;
            if(gn<N&&gk<K) v=(float)((const signed char*)w)[(size_t)gn*K+gk]*scale[gn];
            bh[warp][z]=__float2half(v);}
        __syncthreads();
        wmma::fragment<wmma::matrix_a,16,16,16,__half,wmma::row_major> af;
        wmma::fragment<wmma::matrix_b,16,16,16,__half,wmma::col_major> bf;
        wmma::load_matrix_sync(af,ah,16);wmma::load_matrix_sync(bf,bh[warp],16);
        wmma::mma_sync(acc,af,bf,acc);__syncthreads();
    }
    __shared__ float out[8][256];wmma::store_matrix_sync(out[warp],acc,16,wmma::mem_row_major);__syncwarp();
    for(int z=lane;z<256;z+=32){int m=z/16,n=z%16;
        if(m0+m<M&&n0+n<N)y[(size_t)(m0+m)*N+n0+n]=out[warp][z];}
#endif
}

__global__ static void quantize_s4_rows(uint8_t *q,float *scale,const float *x,int S,int K){
    int s=blockIdx.x; if(s>=S)return; const float *xs=x+(size_t)s*K;
    float v=0; for(int i=threadIdx.x;i<K;i+=blockDim.x)v=fmaxf(v,fabsf(xs[i]));
    __shared__ float m[256]; m[threadIdx.x]=v; __syncthreads();
    for(int n=128;n;n>>=1){if(threadIdx.x<n)m[threadIdx.x]=fmaxf(m[threadIdx.x],m[threadIdx.x+n]);__syncthreads();}
    float sc=m[0]>0?m[0]/7.f:1.f; if(!threadIdx.x)scale[s]=sc;
    uint8_t *dst=q+(size_t)s*((K+1)/2);
    for(int b=threadIdx.x;b<(K+1)/2;b+=blockDim.x){
        int i=b*2,a=__float2int_rn(xs[i]/sc),c=i+1<K?__float2int_rn(xs[i+1]/sc):0;
        a=max(-8,min(7,a)); c=max(-8,min(7,c)); dst[b]=(uint8_t)((a&15)|((c&15)<<4));
    }
}

__global__ static void grouped_s4_wmma(float *y,const uint8_t *x,const float *xscale,
                                        const GroupDesc *desc,int K,int O,int which){
#if __CUDA_ARCH__ >= 750
    using namespace nvcuda;
    int warp=threadIdx.x/32,lane=threadIdx.x%32,tile=blockIdx.x*8+warp,c=blockIdx.y;
    if(tile*8>=O)return; GroupDesc d=desc[c];
    const void *w=which==0?d.g:(which==1?d.u:d.d);
    const float *ws=which==0?d.gs:(which==1?d.us:d.ds);
    int fmt=which==0?d.gf:(which==1?d.uf:d.df);
    if(fmt!=2)return;
    wmma::fragment<wmma::accumulator,8,8,32,int> acc; wmma::fill_fragment(acc,0);
    const uint8_t *a=x+(size_t)d.offset*((K+1)/2);
    const uint8_t *b=(const uint8_t*)w+(size_t)(tile*8)*((K+1)/2);
    for(int k=0;k<K;k+=32){
        wmma::fragment<wmma::matrix_a,8,8,32,wmma::experimental::precision::s4,wmma::row_major> af;
        wmma::fragment<wmma::matrix_b,8,8,32,wmma::experimental::precision::s4,wmma::col_major> bf;
        wmma::load_matrix_sync(af,a+k/2,K);
        wmma::load_matrix_sync(bf,b+k/2,K);
        wmma::mma_sync(acc,af,bf,acc);
    }
    __shared__ int out[8][64]; wmma::store_matrix_sync(out[warp],acc,8,wmma::mem_row_major);
    for(int i=lane;i<64;i+=32){int s=i/8,o=tile*8+i%8;
        if(s<d.rows&&o<O)y[(size_t)(d.offset+s)*O+o]=(float)out[warp][i]*xscale[d.offset+s]*ws[o];}
#endif
}

__global__ static void grouped_hidden(float *y,const float *x,const GroupDesc *desc,
                                      int I,int D,int which){
    int o=blockIdx.x,s=blockIdx.y,c=blockIdx.z; GroupDesc d=desc[c];
    if(s>=d.rows) return;
    const void *w=which?d.u:d.g; const float *sc=which?d.us:d.gs; int fmt=which?d.uf:d.gf;
    size_t rb=row_bytes(fmt,D),row=(size_t)o*rb; const float *xs=x+(size_t)(d.offset+s)*D;
    float sum=0; for(int i=threadIdx.x;i<D;i+=blockDim.x) sum+=xs[i]*weight_at(w,fmt,row,i);
    __shared__ float p[256]; p[threadIdx.x]=sum; __syncthreads();
    for(int n=128;n;n>>=1){ if(threadIdx.x<n)p[threadIdx.x]+=p[threadIdx.x+n]; __syncthreads(); }
    if(!threadIdx.x) y[(size_t)(d.offset+s)*I+o]=p[0]*(fmt?sc[o]:1.f);
}

__global__ static void grouped_down(float *y,const float *x,const GroupDesc *desc,int D,int I){
    int o=blockIdx.x,s=blockIdx.y,c=blockIdx.z; GroupDesc d=desc[c];
    if(s>=d.rows) return;
    size_t rb=row_bytes(d.df,I),row=(size_t)o*rb; const float *xs=x+(size_t)(d.offset+s)*I;
    float sum=0; for(int i=threadIdx.x;i<I;i+=blockDim.x) sum+=xs[i]*weight_at(d.d,d.df,row,i);
    __shared__ float p[256]; p[threadIdx.x]=sum; __syncthreads();
    for(int n=128;n;n>>=1){ if(threadIdx.x<n)p[threadIdx.x]+=p[threadIdx.x+n]; __syncthreads(); }
    if(!threadIdx.x) y[(size_t)(d.offset+s)*D+o]=p[0]*(d.df?d.ds[o]:1.f);
}

__device__ static void unpack_s4(uint8_t v,float *lo,float *hi){
    int a=v&15,b=v>>4; *lo=(float)(a&8?a-16:a); *hi=(float)(b&8?b-16:b);
}

/* Exact low-row W4A32 path. It consumes each packed weight byte once instead
 * of routing both nibbles through weight_at(), preserving FP32 activations. */
__global__ static void grouped_hidden_w4(float *y,const float *x,const GroupDesc *desc,
                                         int I,int D,int which){
    int o=blockIdx.x,s=blockIdx.y,c=blockIdx.z;GroupDesc d=desc[c];if(s>=d.rows)return;
    const uint8_t *w=(const uint8_t*)(which?d.u:d.g);const float *sc=which?d.us:d.gs;
    const uint8_t *row=w+(size_t)o*((D+1)/2);const float *xs=x+(size_t)(d.offset+s)*D;
    float sum=0;for(int b=threadIdx.x;b<(D+1)/2;b+=blockDim.x){float a,z;unpack_s4(row[b],&a,&z);
        int i=b*2;sum+=xs[i]*a;if(i+1<D)sum+=xs[i+1]*z;}
    __shared__ float p[256];p[threadIdx.x]=sum;__syncthreads();
    for(int n=128;n;n>>=1){if(threadIdx.x<n)p[threadIdx.x]+=p[threadIdx.x+n];__syncthreads();}
    if(!threadIdx.x)y[(size_t)(d.offset+s)*I+o]=p[0]*sc[o];
}

__global__ static void grouped_hidden_w4_dual(float *gate,float *up,const float *x,
                                               const GroupDesc *desc,int I,int D){
    int o=blockIdx.x,s=blockIdx.y,c=blockIdx.z;GroupDesc d=desc[c];if(s>=d.rows)return;
    const uint8_t *gr=(const uint8_t*)d.g+(size_t)o*((D+1)/2);
    const uint8_t *ur=(const uint8_t*)d.u+(size_t)o*((D+1)/2);
    const float *xs=x+(size_t)(d.offset+s)*D;float ga=0,ua=0;
    for(int b=threadIdx.x;b<(D+1)/2;b+=blockDim.x){float g0,g1,u0,u1;unpack_s4(gr[b],&g0,&g1);unpack_s4(ur[b],&u0,&u1);
        int i=b*2;ga+=xs[i]*g0;ua+=xs[i]*u0;if(i+1<D){ga+=xs[i+1]*g1;ua+=xs[i+1]*u1;}}
    __shared__ float gp[256],upv[256];gp[threadIdx.x]=ga;upv[threadIdx.x]=ua;__syncthreads();
    for(int n=128;n;n>>=1){if(threadIdx.x<n){gp[threadIdx.x]+=gp[threadIdx.x+n];upv[threadIdx.x]+=upv[threadIdx.x+n];}__syncthreads();}
    if(!threadIdx.x){size_t z=(size_t)(d.offset+s)*I+o;gate[z]=gp[0]*d.gs[o];up[z]=upv[0]*d.us[o];}
}

__global__ static void grouped_down_w4(float *y,const float *x,const GroupDesc *desc,int D,int I){
    int o=blockIdx.x,s=blockIdx.y,c=blockIdx.z;GroupDesc d=desc[c];if(s>=d.rows)return;
    const uint8_t *row=(const uint8_t*)d.d+(size_t)o*((I+1)/2);
    const float *xs=x+(size_t)(d.offset+s)*I;float sum=0;
    for(int b=threadIdx.x;b<(I+1)/2;b+=blockDim.x){float a,z;unpack_s4(row[b],&a,&z);
        int i=b*2;sum+=xs[i]*a;if(i+1<I)sum+=xs[i+1]*z;}
    __shared__ float p[256];p[threadIdx.x]=sum;__syncthreads();
    for(int n=128;n;n>>=1){if(threadIdx.x<n)p[threadIdx.x]+=p[threadIdx.x+n];__syncthreads();}
    if(!threadIdx.x)y[(size_t)(d.offset+s)*D+o]=p[0]*d.ds[o];
}

/* Threads per block for the MLA absorb kernels. GB10 SMs hold ~2048 threads; 256
 * left occupancy ~12%%. 1024 improves it. Power of two (softmax reductions halve
 * blockDim); red[] sizing follows ATTN_TPB. */
#define ATTN_TPB 1024

__global__ static void attention_absorb_kernel(float *ctx,const float *q,const float *latent,
                                                const float *rope,const void *weights,const float *wscale,
                                                int fmt,int H,int Q,int R,int V,int K,int T,float scale){
    int h=blockIdx.x,tid=threadIdx.x,rbase=h*(Q+V);extern __shared__ float sm[];
    float *qa=sm,*cl=qa+K,*scores=cl+K;
    for(int k=tid;k<K;k+=blockDim.x){float a=0;for(int d=0;d<Q;d++)
        a+=q[(size_t)h*(Q+R)+d]*weight_at(weights,fmt,(size_t)(rbase+d)*row_bytes(fmt,K),k)*(fmt?wscale[rbase+d]:1.f);qa[k]=a;}
    __syncthreads();
    for(int t=tid;t<T;t+=blockDim.x){float a=0;const float *lt=latent+(size_t)t*K,*rt=rope+(size_t)t*R;
        for(int k=0;k<K;k++)a+=qa[k]*lt[k];for(int d=0;d<R;d++)a+=q[(size_t)h*(Q+R)+Q+d]*rt[d];scores[t]=a*scale;}
    __syncthreads();
    if(!tid){float mx=scores[0];for(int t=1;t<T;t++)mx=fmaxf(mx,scores[t]);float z=0;
        for(int t=0;t<T;t++){scores[t]=expf(scores[t]-mx);z+=scores[t];}for(int t=0;t<T;t++)scores[t]/=z;}
    __syncthreads();
    for(int k=tid;k<K;k+=blockDim.x){float a=0;for(int t=0;t<T;t++)a+=scores[t]*latent[(size_t)t*K+k];cl[k]=a;}
    __syncthreads();
    for(int v=tid;v<V;v+=blockDim.x){int row=rbase+Q+v;float a=0;size_t rb=row_bytes(fmt,K);
        for(int k=0;k<K;k++)a+=cl[k]*weight_at(weights,fmt,(size_t)row*rb,k);ctx[(size_t)h*V+v]=a*(fmt?wscale[row]:1.f);}
}

__global__ static void attention_absorb_batch_kernel(float *ctx,const float *q,
        const float *latent,const float *rope,const void *weights,const float *wscale,
        int fmt,int S,int H,int Q,int R,int V,int K,int T,float scale){
    int s=blockIdx.y,h=blockIdx.x,tid=threadIdx.x,nt=T-S+s+1,rbase=h*(Q+V);
    if(s>=S||nt<1)return;
    extern __shared__ float sm[];float *qa=sm,*cl=qa+K,*scores=cl+K,*red=scores+T;
    const float *qs=q+((size_t)s*H+h)*(Q+R);
    for(int k=tid;k<K;k+=blockDim.x){float a=0;for(int d=0;d<Q;d++)
        a+=qs[d]*weight_at(weights,fmt,(size_t)(rbase+d)*row_bytes(fmt,K),k)*
          (fmt?wscale[rbase+d]:1.f);qa[k]=a;}
    __syncthreads();
    for(int t=tid;t<nt;t+=blockDim.x){float a=0;const float *lt=latent+(size_t)t*K;
        const float *rt=rope+(size_t)t*R;for(int k=0;k<K;k++)a+=qa[k]*lt[k];
        for(int d=0;d<R;d++)a+=qs[Q+d]*rt[d];scores[t]=a*scale;}
    __syncthreads();
    float local=-3.402823466e+38F;for(int t=tid;t<nt;t+=blockDim.x)local=fmaxf(local,scores[t]);
    red[tid]=local;__syncthreads();
    for(int n=blockDim.x>>1;n;n>>=1){if(tid<n)red[tid]=fmaxf(red[tid],red[tid+n]);__syncthreads();}
    float mx=red[0];local=0;for(int t=tid;t<nt;t+=blockDim.x){float e=expf(scores[t]-mx);scores[t]=e;local+=e;}
    red[tid]=local;__syncthreads();
    for(int n=blockDim.x>>1;n;n>>=1){if(tid<n)red[tid]+=red[tid+n];__syncthreads();}
    float inv=1.f/red[0];for(int t=tid;t<nt;t+=blockDim.x)scores[t]*=inv;
    __syncthreads();
    for(int k=tid;k<K;k+=blockDim.x){float a=0;for(int t=0;t<nt;t++)
        a+=scores[t]*latent[(size_t)t*K+k];cl[k]=a;}
    __syncthreads();
    for(int v=tid;v<V;v+=blockDim.x){int row=rbase+Q+v;float a=0;size_t rb=row_bytes(fmt,K);
        for(int k=0;k<K;k++)a+=cl[k]*weight_at(weights,fmt,(size_t)row*rb,k);
        ctx[((size_t)s*H+h)*V+v]=a*(fmt?wscale[row]:1.f);}
}

/* DSA sparse prefill attention. Identical to attention_absorb_batch_kernel except
 * each query attends only to its indexer selection instead of all `nt` causal
 * positions: `sel_idx[s*maxsel + j]` (j < sel_cnt[s]) are the chosen cache rows.
 * An empty selection (sel_cnt[s] <= 0) is the is_dense case — attend causally to
 * 0..nt, which is guaranteed <= maxsel there (is_dense holds only when nk <=
 * index_topk = maxsel), so `scores[]` sized to maxsel is always sufficient. */
__global__ static void attention_absorb_sparse_kernel(float *ctx,const float *q,
        const float *latent,const float *rope,const void *weights,const float *wscale,
        const int *sel_idx,const int *sel_cnt,int maxsel,
        int fmt,int H0,int S,int H,int Q,int R,int V,int K,int T,float scale){
    // Tensor-parallel head slice: this launch covers heads [H0, H0+gridDim.x); the
    // global head index is H0+blockIdx.x while H stays the full head count so every
    // `*H` stride (q, ctx) keeps the full [S,H,·] layout. Columns outside the slice
    // are left untouched — the caller zeroes dc->ac first when the slice is partial.
    int s=blockIdx.y,h=H0+blockIdx.x,tid=threadIdx.x,nt=T-S+s+1,rbase=h*(Q+V);
    if(s>=S||nt<1)return;
    int cnt=sel_cnt[s],dense=(cnt<=0),n=dense?nt:cnt;
    const int *sidx=sel_idx+(size_t)s*maxsel;
    extern __shared__ float sm[];float *qa=sm,*cl=qa+K,*scores=cl+K,*red=scores+maxsel;
    const float *qs=q+((size_t)s*H+h)*(Q+R);
    for(int k=tid;k<K;k+=blockDim.x){float a=0;for(int d=0;d<Q;d++)
        a+=qs[d]*weight_at(weights,fmt,(size_t)(rbase+d)*row_bytes(fmt,K),k)*
          (fmt?wscale[rbase+d]:1.f);qa[k]=a;}
    __syncthreads();
    for(int j=tid;j<n;j+=blockDim.x){int t=dense?j:sidx[j];float a=0;
        const float *lt=latent+(size_t)t*K,*rt=rope+(size_t)t*R;
        for(int k=0;k<K;k++)a+=qa[k]*lt[k];for(int d=0;d<R;d++)a+=qs[Q+d]*rt[d];scores[j]=a*scale;}
    __syncthreads();
    float local=-3.402823466e+38F;for(int j=tid;j<n;j+=blockDim.x)local=fmaxf(local,scores[j]);
    red[tid]=local;__syncthreads();
    for(int m=blockDim.x>>1;m;m>>=1){if(tid<m)red[tid]=fmaxf(red[tid],red[tid+m]);__syncthreads();}
    float mx=red[0];local=0;for(int j=tid;j<n;j+=blockDim.x){float e=expf(scores[j]-mx);scores[j]=e;local+=e;}
    red[tid]=local;__syncthreads();
    for(int m=blockDim.x>>1;m;m>>=1){if(tid<m)red[tid]+=red[tid+m];__syncthreads();}
    float inv=1.f/red[0];for(int j=tid;j<n;j+=blockDim.x)scores[j]*=inv;
    __syncthreads();
    for(int k=tid;k<K;k+=blockDim.x){float a=0;for(int j=0;j<n;j++){int t=dense?j:sidx[j];
        a+=scores[j]*latent[(size_t)t*K+k];}cl[k]=a;}
    __syncthreads();
    for(int v=tid;v<V;v+=blockDim.x){int row=rbase+Q+v;float a=0;size_t rb=row_bytes(fmt,K);
        for(int k=0;k<K;k++)a+=cl[k]*weight_at(weights,fmt,(size_t)row*rb,k);
        ctx[((size_t)s*H+h)*V+v]=a*(fmt?wscale[row]:1.f);}
}

/* ==== DSA lightning-indexer scores ===========================================
 * score[s][t] = (1/sqrt(nh)) * sum_h hw[s][h] * relu((1/sqrt(hd)) * dot(qi[s][h], key[t]))
 * where key[t] is [hd], SHARED across all nh heads. This was the indexer's CPU hot
 * loop (~25.8 GFLOP per FULL layer). One block per query; `i` outer / `h` inner so
 * each key element is read once from global and every head's dot accumulates in the
 * same ascending-i order as the CPU reference — the selection must not shift. */
__global__ static void dsa_indexer_scores(float *scores,const float *qi,const float *hw,
        const float *keys,int nsp,int s0,int nh,int hd,int T,int pos_base){
    int si=blockIdx.x; if(si>=nsp)return;
    int s=s0+si, nk=pos_base+s+1; if(nk>T)nk=T;
    extern __shared__ float sm[];
    float *q=sm, *w=q+(size_t)nh*hd;
    for(int z=threadIdx.x;z<nh*hd;z+=blockDim.x)q[z]=qi[(size_t)si*nh*hd+z];
    for(int z=threadIdx.x;z<nh;z+=blockDim.x)w[z]=hw[(size_t)si*nh+z];
    __syncthreads();
    float rs=rsqrtf((float)hd), wsc=rsqrtf((float)nh);
    for(int t=threadIdx.x;t<nk;t+=blockDim.x){
        const float *kt=keys+(size_t)t*hd;
        float acc[32];                     /* nh <= 32 (GLM: 32); larger falls back to CPU */
        for(int h=0;h<nh;h++)acc[h]=0.f;
        for(int i=0;i<hd;i++){float ki=kt[i];const float *qi_i=q+i;
            for(int h=0;h<nh;h++)acc[h]+=qi_i[(size_t)h*hd]*ki;}
        float a=0.f;
        for(int h=0;h<nh;h++){float d0=acc[h]*rs; if(d0>0.f)a+=w[h]*d0;}
        scores[(size_t)si*T+t]=a*wsc;
    }
}

/* ==== Tensor-core (WMMA) DSA sparse-attention prefill core ====================
 * The scalar attention_absorb_sparse_kernel is ~4 GFLOP/s (75% of prefill attn).
 * MLA-absorb attention is two GEMMs per head in latent space:
 *   Scores[S,T] = QA[S,K+R] @ KB[T,K+R]^T ;  Ctx_lat[S,K] = P[S,T] @ Latent[T,K]
 * with QA=[scale*qabs | scale*qrope], KB=[latent | rope]. WMMA does the GEMMs;
 * a per-query DSA mask (unselected key -> -inf) keeps the sparse result exact;
 * flash online-softmax tiles over T; causal tiling skips the future. ~3x the
 * scalar core at GLM dims (microbench). Behind COLI_TC_ATTN. */
#define ATC_QT 16

/* KB[T,K+R] fp16 = [latent | rope]. */
__global__ static void tc_build_kb(__half *KB,const float *latent,const float *rope,int K,int R,int T){
    int t=blockIdx.x,tid=threadIdx.x,KR=K+R;
    for(int c=tid;c<KR;c+=blockDim.x)
        KB[(size_t)t*KR+c]=__float2half(c<K?latent[(size_t)t*K+c]:rope[(size_t)t*R+(c-K)]);
}

/* QA[S,H,K+R] fp16 = scale*[qabs | qrope] for the head slice [H0,H0+gridDim.x).
 * qabs[k]=sum_d q_nope[d]*W_K[rbase+d][k]*(fmt?wscale:1). Scale folded so Scores come out scaled. */
__global__ static void tc_build_qa(__half *QA,const float *q,const void *weights,const float *wscale,
        int fmt,int H0,int S,int H,int Q,int R,int V,int K,float scale){
    int s=blockIdx.y,h=H0+blockIdx.x,tid=threadIdx.x,KR=K+R,rbase=h*(Q+V);
    const float *qs=q+((size_t)s*H+h)*(Q+R);
    __half *dst=QA+((size_t)s*H+h)*KR; size_t rb=row_bytes(fmt,K);
    for(int k=tid;k<K;k+=blockDim.x){float a=0;
        for(int d=0;d<Q;d++)a+=qs[d]*weight_at(weights,fmt,(size_t)(rbase+d)*rb,k)*(fmt?wscale[rbase+d]:1.f);
        dst[k]=__float2half(a*scale);}
    for(int d=tid;d<R;d+=blockDim.x)dst[K+d]=__float2half(qs[Q+d]*scale);
}

/* Per-query key bitmask [S][ceil(T/8)]: for sparse queries (cnt>0) set the selected
 * keys' bits. Dense queries (cnt<=0) leave the row zero — the flash kernel uses causal
 * only there. One thread owns a whole query row (no atomics). Mask must be pre-zeroed. */
__global__ static void tc_build_mask(uint8_t *mask,const int *sel_idx,const int *sel_cnt,int maxsel,int S,int T){
    int s=blockIdx.x*blockDim.x+threadIdx.x; if(s>=S)return;
    int cnt=sel_cnt[s]; if(cnt<=0)return;
    size_t mr=(T+7)/8; uint8_t *row=mask+(size_t)s*mr; const int *sidx=sel_idx+(size_t)s*maxsel;
    for(int j=0;j<cnt;j++){int t=sidx[j]; if(t>=0&&t<T) row[t>>3]|=(uint8_t)(1<<(t&7));}
}

/* Flash MLA attention. Block=(head-slice index, query-tile of 16). Both GEMMs run
 * across all 8 warps (scores split-K; P@Latent by kn-tile). Dynamic shared: QA+KB
 * (fp16) + acc (f32) = QT*(4*(K+R)+4*K) bytes. */
__global__ static void tc_sparse_attn(float *ctx,const __half *QAh,const __half *KBh,
        const float *latent,const void *weights,const float *wscale,const uint8_t *mask,const int *sel_cnt,
        int fmt,int H0,int S,int H,int Q,int R,int V,int K,int T){
#if __CUDA_ARCH__ >= 700
    using namespace nvcuda;
    int h=H0+blockIdx.x, qt=blockIdx.y, tid=threadIdx.x, warp=tid>>5, lane=tid&31;
    int q0=qt*ATC_QT, rbase=h*(Q+V), KR=K+R, nwarp=blockDim.x>>5; size_t mr=(T+7)/8;
    extern __shared__ char smem[];
    __half *QA=(__half*)smem; __half *KB=QA+ATC_QT*KR; float *acc=(float*)(KB+ATC_QT*KR);
    __shared__ __half Pt[ATC_QT*ATC_QT];
    __shared__ __half ah[256], ah8[8][256], bh8[8][256];
    __shared__ float scpart[8*256], sc[ATC_QT*ATC_QT], mrow[ATC_QT], lrow[ATC_QT], corr[ATC_QT];
    for(int z=tid;z<ATC_QT*KR;z+=blockDim.x){int r=z/KR,c=z%KR;int s=q0+r;
        QA[z]=(s<S)?QAh[((size_t)s*H+h)*KR+c]:__float2half(0.f);}
    for(int r=tid;r<ATC_QT;r+=blockDim.x){mrow[r]=-3.4e38f;lrow[r]=0.f;}
    for(int z=tid;z<ATC_QT*K;z+=blockDim.x)acc[z]=0.f;
    __syncthreads();
    int ktmax=q0+ATC_QT; if(ktmax>T)ktmax=T;                 // causal (single-shot prefill T==S)
    for(int kt=0;kt<ktmax;kt+=ATC_QT){
        for(int z=tid;z<ATC_QT*KR;z+=blockDim.x){int r=z/KR,c=z%KR;int t=kt+r;
            KB[z]=(t<T)?KBh[(size_t)t*KR+c]:__float2half(0.f);}
        __syncthreads();
        // Scores[16,16]=QA@KB^T, split-K across warps.
        { __half *myah=ah8[warp],*mybh=bh8[warp];
          wmma::fragment<wmma::accumulator,16,16,16,float> accS; wmma::fill_fragment(accS,0.f);
          for(int k0=warp*16;k0<KR;k0+=nwarp*16){
            for(int z=lane;z<256;z+=32){int m=z/16,k=z%16;myah[z]=QA[m*KR+(k0+k)];mybh[z]=KB[m*KR+(k0+k)];}
            __syncwarp();
            wmma::fragment<wmma::matrix_a,16,16,16,__half,wmma::row_major> af;
            wmma::fragment<wmma::matrix_b,16,16,16,__half,wmma::col_major> bf;
            wmma::load_matrix_sync(af,myah,16);wmma::load_matrix_sync(bf,mybh,16);
            wmma::mma_sync(accS,af,bf,accS);__syncwarp(); }
          wmma::store_matrix_sync(&scpart[warp*256],accS,16,wmma::mem_row_major); }
        __syncthreads();
        for(int z=tid;z<ATC_QT*ATC_QT;z+=blockDim.x){float a=0;for(int wr=0;wr<nwarp;wr++)a+=scpart[wr*256+z];sc[z]=a;}
        __syncthreads();
        // mask + online softmax per query row (one warp per row)
        for(int r=warp;r<ATC_QT;r+=nwarp){int s=q0+r;int sp=(s<S)?sel_cnt[s]:0;int pos=T-S+s;
            float tmax=-3.4e38f;
            for(int c=lane;c<ATC_QT;c+=32){int t=kt+c;
                int keep=(s<S&&t<T&&t<=pos&&(sp>0?((mask[(size_t)s*mr+(t>>3)]>>(t&7))&1):1));
                float v=keep?sc[r*ATC_QT+c]:-3.4e38f; sc[r*ATC_QT+c]=v; tmax=fmaxf(tmax,v);}
            for(int o=16;o;o>>=1)tmax=fmaxf(tmax,__shfl_down_sync(0xffffffff,tmax,o));
            tmax=__shfl_sync(0xffffffff,tmax,0);
            float mold=mrow[r],mnew=fmaxf(mold,tmax),cr=expf(mold-mnew),lsum=0.f;
            for(int c=lane;c<ATC_QT;c+=32){float e=(sc[r*ATC_QT+c]>-1e30f)?expf(sc[r*ATC_QT+c]-mnew):0.f;Pt[r*ATC_QT+c]=__float2half(e);lsum+=e;}
            for(int o=16;o;o>>=1)lsum+=__shfl_down_sync(0xffffffff,lsum,o);
            lsum=__shfl_sync(0xffffffff,lsum,0);
            if(lane==0){mrow[r]=mnew;corr[r]=cr;lrow[r]=lrow[r]*cr+lsum;}}
        __syncthreads();
        // acc = acc*corr + P@Latent (accumulate into the WMMA fragment loaded from acc)
        for(int z=tid;z<ATC_QT*K;z+=blockDim.x){int r=z/K;acc[z]*=corr[r];}
        for(int z=tid;z<256;z+=blockDim.x)ah[z]=Pt[z];
        __syncthreads();
        { __half *mybh=bh8[warp];
          for(int kn=warp*16;kn<K;kn+=nwarp*16){
            wmma::fragment<wmma::accumulator,16,16,16,float> accP;
            wmma::load_matrix_sync(accP,&acc[kn],K,wmma::mem_row_major);
            for(int z=lane;z<256;z+=32){int n=z/16,key=z%16;int t=kt+key;
                mybh[z]=(t<T)?__float2half(latent[(size_t)t*K+(kn+n)]):__float2half(0.f);}
            __syncwarp();
            wmma::fragment<wmma::matrix_a,16,16,16,__half,wmma::row_major> af;
            wmma::fragment<wmma::matrix_b,16,16,16,__half,wmma::col_major> bf;
            wmma::load_matrix_sync(af,ah,16);wmma::load_matrix_sync(bf,mybh,16);
            wmma::mma_sync(accP,af,bf,accP);
            wmma::store_matrix_sync(&acc[kn],accP,K,wmma::mem_row_major);__syncwarp(); } }
        __syncthreads();
    }
    // Ctx[s,v] = (acc/l) @ W_V^T
    size_t rb=row_bytes(fmt,K);
    for(int r=0;r<ATC_QT;r++){int s=q0+r;if(s>=S)continue;float inv=1.f/lrow[r];
        for(int v=tid;v<V;v+=blockDim.x){int row=rbase+Q+v;float a=0;
            for(int k=0;k<K;k++)a+=(acc[r*K+k]*inv)*weight_at(weights,fmt,(size_t)row*rb,k);
            ctx[((size_t)s*H+h)*V+v]=a*(fmt?wscale[row]:1.f);}
        __syncthreads(); }
#endif
}

/* ---- Flash-attention decode absorb (S=1): T-parallel with online softmax ----
 * The per-head kernel above serializes the whole context in one block (64 blocks
 * for H=64 → low parallelism on the GB10). Flash splits the key dimension across
 * blocks: kernel 1 precomputes the absorbed query, kernel 2 runs H×nTiles blocks
 * each reducing one T-tile to a partial (m, l, acc[K]) with online softmax, and
 * kernel 3 combines the tiles per head and applies W_V. FLASH_TILE tokens/block. */
#define FLASH_TILE 512

__global__ static void flash_qabs(float *qabs,const float *q,const void *weights,
        const float *wscale,int fmt,int H,int Q,int R,int V,int K){
    int h=blockIdx.x,tid=threadIdx.x,rbase=h*(Q+V);size_t rb=row_bytes(fmt,K);
    const float *qs=q+(size_t)h*(Q+R);
    for(int k=tid;k<K;k+=blockDim.x){float a=0;
        for(int d=0;d<Q;d++)a+=qs[d]*weight_at(weights,fmt,(size_t)(rbase+d)*rb,k)*(fmt?wscale[rbase+d]:1.f);
        qabs[(size_t)h*K+k]=a;}
}

/* One block per (head, tile). Emits partial[(h*nTiles+tile)] = {m, l, acc[K]}. */
__global__ static void flash_partial(float *partials,const float *qabs,const float *q,
        const float *latent,const float *rope,int H,int Q,int R,int K,int T,int nTiles,float scale){
    int h=blockIdx.x,tile=blockIdx.y,tid=threadIdx.x;
    int t0=tile*FLASH_TILE,t1=t0+FLASH_TILE; if(t1>T)t1=T; int n=t1-t0; if(n<=0)return;
    extern __shared__ float sm[];float *scores=sm,*acc=scores+FLASH_TILE,*red=acc+K;
    const float *qa=qabs+(size_t)h*K,*qr=q+(size_t)h*(Q+R)+Q;
    for(int i=tid;i<n;i+=blockDim.x){int t=t0+i;const float *lt=latent+(size_t)t*K,*rt=rope+(size_t)t*R;
        float a=0;for(int k=0;k<K;k++)a+=qa[k]*lt[k];for(int d=0;d<R;d++)a+=qr[d]*rt[d];scores[i]=a*scale;}
    __syncthreads();
    float local=-3.402823466e+38F;for(int i=tid;i<n;i+=blockDim.x)local=fmaxf(local,scores[i]);
    red[tid]=local;__syncthreads();
    for(int s=blockDim.x>>1;s;s>>=1){if(tid<s)red[tid]=fmaxf(red[tid],red[tid+s]);__syncthreads();}
    float m=red[0];__syncthreads();
    local=0;for(int i=tid;i<n;i+=blockDim.x){float e=expf(scores[i]-m);scores[i]=e;local+=e;}
    red[tid]=local;__syncthreads();
    for(int s=blockDim.x>>1;s;s>>=1){if(tid<s)red[tid]+=red[tid+s];__syncthreads();}
    float l=red[0];__syncthreads();
    for(int k=tid;k<K;k+=blockDim.x){float a=0;for(int i=0;i<n;i++)a+=scores[i]*latent[(size_t)(t0+i)*K+k];acc[k]=a;}
    __syncthreads();
    float *p=partials+(size_t)(h*nTiles+tile)*(K+2);
    if(tid==0){p[0]=m;p[1]=l;}
    for(int k=tid;k<K;k+=blockDim.x)p[2+k]=acc[k];
}

/* One block per head: combine nTiles partials (online softmax) -> clat, apply W_V. */
__global__ static void flash_combine(float *ctx,const float *partials,const void *weights,
        const float *wscale,int fmt,int H,int Q,int V,int K,int nTiles){
    int h=blockIdx.x,tid=threadIdx.x,rbase=h*(Q+V);size_t rb=row_bytes(fmt,K);
    extern __shared__ float sm[];float *clat=sm;__shared__ float M,L;
    const float *base=partials+(size_t)h*nTiles*(K+2);
    if(tid==0){float mx=-3.402823466e+38F;for(int i=0;i<nTiles;i++)mx=fmaxf(mx,base[(size_t)i*(K+2)]);M=mx;}
    __syncthreads();
    if(tid==0){float s=0;for(int i=0;i<nTiles;i++)s+=expf(base[(size_t)i*(K+2)]-M)*base[(size_t)i*(K+2)+1];L=s;}
    __syncthreads();
    for(int k=tid;k<K;k+=blockDim.x){float a=0;
        for(int i=0;i<nTiles;i++)a+=expf(base[(size_t)i*(K+2)]-M)*base[(size_t)i*(K+2)+2+k];
        clat[k]=a/L;}
    __syncthreads();
    for(int v=tid;v<V;v+=blockDim.x){int row=rbase+Q+v;float a=0;
        for(int k=0;k<K;k++)a+=clat[k]*weight_at(weights,fmt,(size_t)row*rb,k);
        ctx[(size_t)h*V+v]=a*(fmt?wscale[row]:1.f);}
}

static int reserve(float **ptr, size_t *cap, size_t bytes) {
    if (*cap >= bytes) return 1;
    if (*ptr) cudaFree(*ptr);
    *ptr = nullptr;
    *cap = 0;
    if (!cuda_ok(cudaMalloc(ptr, bytes), "scratch allocation")) return 0;
    *cap = bytes;
    return 1;
}

static int reserve_bytes(void **ptr,size_t *cap,size_t bytes){
    if(*cap>=bytes) return 1; if(*ptr) cudaFree(*ptr); *ptr=nullptr; *cap=0;
    if(!cuda_ok(cudaMalloc(ptr,bytes),"descriptor allocation")) return 0; *cap=bytes; return 1;
}

static int reserve_pinned(float **ptr,size_t *cap,size_t bytes){
    if(*cap>=bytes)return 1;if(*ptr)cudaFreeHost(*ptr);*ptr=nullptr;*cap=0;
    if(!cuda_ok(cudaMallocHost(ptr,bytes),"pinned staging allocation"))return 0;*cap=bytes;return 1;
}

extern "C" int coli_cuda_init(const int *devices, int count) {
    int available = 0;
    if (!devices || count < 1 || count > COLI_CUDA_MAX_DEVICES) return 0;
    if (!cuda_ok(cudaGetDeviceCount(&available), "device discovery")) return 0;
    g_nctx = 0;
    for (int i = 0; i < count; i++) {
        int device = devices[i];
        if (device < 0 || device >= available) {
            std::fprintf(stderr, "[CUDA] invalid device %d (available: 0..%d)\n", device, available - 1);
            g_nctx = 0;
            return 0;
        }
        if (find_ctx(device)) {
            std::fprintf(stderr, "[CUDA] duplicate device %d\n", device);
            g_nctx = 0;
            return 0;
        }
        DeviceContext *ctx = &g_ctx[g_nctx];
        *ctx = {};
        ctx->device = device;
        if (!select_ctx(ctx)) { g_nctx = 0; return 0; }
        cudaDeviceProp prop{};
        if (!cuda_ok(cudaGetDeviceProperties(&prop, device), "device properties")) { g_nctx = 0; return 0; }
        ctx->compute_major=prop.major;ctx->compute_minor=prop.minor;
        if(!cuda_ok(cudaStreamCreateWithFlags(&ctx->stream,cudaStreamNonBlocking),"stream creation")){
            g_nctx=0;return 0;
        }
        g_nctx++;
        std::fprintf(stderr, "[CUDA] device %d: %s, %.1f GB VRAM, sm_%d%d\n",
                     device, prop.name, prop.totalGlobalMem / 1e9, prop.major, prop.minor);
    }
    return 1;
}

extern "C" void coli_cuda_shutdown(void) {
    for (int i = 0; i < g_nctx; i++) {
        DeviceContext *ctx = &g_ctx[i];
        if (!select_ctx(ctx)) continue;
        if (ctx->x) cudaFree(ctx->x);
        if (ctx->y) cudaFree(ctx->y);
        if (ctx->gate) cudaFree(ctx->gate);
        if (ctx->up) cudaFree(ctx->up);
        if (ctx->qx) cudaFree(ctx->qx);
        if (ctx->qscale) cudaFree(ctx->qscale);
        if(ctx->aq)cudaFree(ctx->aq);if(ctx->al)cudaFree(ctx->al);if(ctx->ar)cudaFree(ctx->ar);if(ctx->ac)cudaFree(ctx->ac);
        if(ctx->asel)cudaFree(ctx->asel);if(ctx->acnt)cudaFree(ctx->acnt);
        if(ctx->aqa)cudaFree(ctx->aqa);if(ctx->akb)cudaFree(ctx->akb);if(ctx->amsk)cudaFree(ctx->amsk);
        for(int b=0;b<24;b++) if(ctx->pipe_buf[b]) cudaFree(ctx->pipe_buf[b]);
        if (ctx->host_x) cudaFreeHost(ctx->host_x);
        if (ctx->host_y) cudaFreeHost(ctx->host_y);
        if (ctx->stream) cudaStreamDestroy(ctx->stream);
        if (ctx->group_desc) cudaFree(ctx->group_desc);
        ctx->x = ctx->y = ctx->gate = ctx->up = nullptr;
        ctx->qx=nullptr; ctx->qscale=nullptr;
        ctx->aq=ctx->al=ctx->ar=ctx->ac=nullptr;
        ctx->asel=ctx->acnt=nullptr;
        ctx->aqa=ctx->akb=ctx->amsk=nullptr;
        ctx->aqa_cap=ctx->akb_cap=ctx->amsk_cap=0;
        ctx->host_x=ctx->host_y=nullptr;ctx->stream=nullptr;
        ctx->ewg=ctx->ewu=ctx->ewd=nullptr;ctx->esg=ctx->esu=ctx->esd=nullptr;
        ctx->ewg_cap=ctx->ewu_cap=ctx->ewd_cap=ctx->esg_cap=ctx->esu_cap=ctx->esd_cap=0;
        ctx->x_cap = ctx->y_cap = ctx->gate_cap = ctx->up_cap = 0;
        ctx->qx_cap=ctx->qscale_cap=0;
        ctx->aq_cap=ctx->al_cap=ctx->ar_cap=ctx->ac_cap=0;
        ctx->asel_cap=ctx->acnt_cap=0;
        ctx->host_x_cap=ctx->host_y_cap=0;
        ctx->group_desc=nullptr; ctx->group_desc_cap=0;
    }
    g_nctx = 0;
}

extern "C" int coli_cuda_device_count(void) { return g_nctx; }

extern "C" int coli_cuda_device_at(int index) {
    return index >= 0 && index < g_nctx ? g_ctx[index].device : -1;
}

extern "C" int coli_cuda_mem_info(int device, size_t *free_bytes, size_t *total_bytes) {
    DeviceContext *ctx = find_ctx(device);
    if (!free_bytes || !total_bytes || !select_ctx(ctx)) return 0;
    return cuda_ok(cudaMemGetInfo(free_bytes, total_bytes), "memory info");
}

// Whether the device can read pageable host memory directly (coherent unified
// memory). 1 → the zero-copy `coli_cuda_tensor_wrap` path is usable.
extern "C" int coli_cuda_pageable_access(int device) {
    int v = 0;
    if (cudaDeviceGetAttribute(&v, cudaDevAttrPageableMemoryAccess, device) != cudaSuccess)
        return 0;
    return v;
}

extern "C" void coli_cuda_stats(int device, size_t *tensor_count, size_t *tensor_bytes) {
    size_t count = 0, bytes = 0;
    for (int i = 0; i < g_nctx; i++) if (device < 0 || g_ctx[i].device == device) {
        count += g_ctx[i].tensor_count;
        bytes += g_ctx[i].tensor_bytes;
    }
    if (tensor_count) *tensor_count = count;
    if (tensor_bytes) *tensor_bytes = bytes;
}

extern "C" void coli_cuda_group_stats(uint64_t *calls, uint64_t *experts, uint64_t *rows,
                                        double *h2d_ms, double *kernel_ms, double *d2h_ms) {
    if(calls) *calls=g_group_calls; if(experts) *experts=g_group_experts; if(rows) *rows=g_group_rows;
    if(h2d_ms) *h2d_ms=g_group_h2d_ms; if(kernel_ms) *kernel_ms=g_group_kernel_ms;
    if(d2h_ms) *d2h_ms=g_group_d2h_ms;
}

extern "C" int coli_cuda_tensor_upload(ColiCudaTensor **tensor,
                                        const void *weights, const float *scales,
                                        int fmt, int I, int O, int device) {
    DeviceContext *ctx = find_ctx(device);
    if (!tensor || !weights || I < 1 || O < 1 || !select_ctx(ctx)) return 0;
    size_t rb = row_bytes(fmt, I);
    if (!rb || (fmt && !scales)) return 0;
    if (*tensor) {
        ColiCudaTensor *t = *tensor;
        return t->fmt == fmt && t->I == I && t->O == O && t->device == device;
    }
    ColiCudaTensor *t = static_cast<ColiCudaTensor *>(std::calloc(1, sizeof(*t)));
    if (!t) return 0;
    t->fmt = fmt; t->I = I; t->O = O; t->device = device; t->weight_bytes = rb * (size_t)O;
    if (!cuda_ok(cudaMalloc(&t->weights, t->weight_bytes), "tensor allocation") ||
        !cuda_ok(cudaMemcpy(t->weights, weights, t->weight_bytes, cudaMemcpyHostToDevice), "tensor upload")) {
        coli_cuda_tensor_free(t);
        return 0;
    }
    if(fmt==2){offset_to_signed_s4<<<(unsigned)((t->weight_bytes+255)/256),256>>>((uint8_t*)t->weights,t->weight_bytes);
        if(!cuda_ok(cudaGetLastError(),"int4 weight conversion")){coli_cuda_tensor_free(t);return 0;}}
    if (fmt) {
        if (!cuda_ok(cudaMalloc(&t->scales, (size_t)O * sizeof(float)), "scale allocation") ||
            !cuda_ok(cudaMemcpy(t->scales, scales, (size_t)O * sizeof(float), cudaMemcpyHostToDevice), "scale upload")) {
            coli_cuda_tensor_free(t);
            return 0;
        }
    }
    t->tracked = 1;
    ctx->tensor_count++;
    ctx->tensor_bytes += t->weight_bytes + (fmt ? (size_t)O * sizeof(float) : 0);
    *tensor = t;
    return 1;
}

extern "C" int coli_cuda_tensor_update(ColiCudaTensor *tensor,
                                          const void *weights,
                                          const float *scales) {
    if (!tensor || !weights || (tensor->fmt && !scales)) return 0;
    DeviceContext *ctx=find_ctx(tensor->device);
    if (!select_ctx(ctx)) return 0;
    if (!cuda_ok(cudaMemcpy(tensor->weights,weights,tensor->weight_bytes,
                            cudaMemcpyHostToDevice),"tensor refresh")) return 0;
    if(tensor->fmt==2){
        offset_to_signed_s4<<<(unsigned)((tensor->weight_bytes+255)/256),256>>>(
            (uint8_t*)tensor->weights,tensor->weight_bytes);
        if(!cuda_ok(cudaGetLastError(),"int4 weight refresh")) return 0;
    }
    return !tensor->fmt || cuda_ok(cudaMemcpy(tensor->scales,scales,
        (size_t)tensor->O*sizeof(float),cudaMemcpyHostToDevice),"scale refresh");
}

extern "C" int coli_cuda_matmul(ColiCudaTensor **tensor,
                                 float *y, const float *x,
                                 const void *weights, const float *scales,
                                 int fmt, int S, int I, int O, int device) {
    if (S < 1 || !coli_cuda_tensor_upload(tensor, weights, scales, fmt, I, O, device)) return 0;
    ColiCudaTensor *t = *tensor;
    DeviceContext *ctx = find_ctx(t->device);
    if (!select_ctx(ctx)) return 0;
    size_t rb = row_bytes(fmt, I);
    size_t xb = (size_t)S * I * sizeof(float), yb = (size_t)S * O * sizeof(float);
    if (!reserve(&ctx->x, &ctx->x_cap, xb) || !reserve(&ctx->y, &ctx->y_cap, yb)) return 0;
    if (!cuda_ok(cudaMemcpy(ctx->x, x, xb, cudaMemcpyHostToDevice), "input upload")) return 0;
    // Tiled tensor-core path for the resident matmuls (attention q/kv/o/kv_b proj):
    // reads each weight once per 16-row tile vs quant_matmul's S-fold re-read.
    const char *tile_env = getenv("COLI_TILE_I8");
    int tile = (!tile_env || strcmp(tile_env, "0") != 0) && ctx->compute_major >= 7;
    if (tile && (fmt == 1 || fmt == 4)) {
        dim3 tg((unsigned)((O + 63) / 64), (unsigned)((S + 15) / 16));
        if (fmt == 4)
            fp8a16_matmul<<<tg, 128>>>(ctx->y, ctx->x, (const uint8_t *)t->weights, t->scales, S, I, O);
        else
            i8a16_matmul<<<tg, 128>>>(ctx->y, ctx->x, (const uint8_t *)t->weights, t->scales, S, I, O);
    } else {
        dim3 grid((unsigned)O, (unsigned)S);
        quant_matmul<<<grid, 256>>>(ctx->y, ctx->x, t->weights, t->scales, fmt, S, I, O, rb, t->wrapped);
    }
    if (!cuda_ok(cudaGetLastError(), "matmul launch") ||
        !cuda_ok(cudaMemcpy(y, ctx->y, yb, cudaMemcpyDeviceToHost), "output download")) return 0;
    return 1;
}

extern "C" int coli_cuda_expert_mlp(ColiCudaTensor *gate, ColiCudaTensor *up,
                                      ColiCudaTensor *down, float *y,
                                      const float *x, int S) {
    if (!gate || !up || !down || !x || !y || S < 1 ||
        gate->device != up->device || gate->device != down->device ||
        gate->I != up->I || gate->O != up->O ||
        down->I != gate->O || down->O != gate->I) return 0;
    DeviceContext *ctx = find_ctx(gate->device);
    if (!select_ctx(ctx)) return 0;
    int D = gate->I, I = gate->O;
    size_t xb=(size_t)S*D*sizeof(float), ib=(size_t)S*I*sizeof(float);
    size_t yb=(size_t)S*D*sizeof(float);
    if (!reserve(&ctx->x,&ctx->x_cap,xb) || !reserve(&ctx->y,&ctx->y_cap,yb) ||
        !reserve(&ctx->gate,&ctx->gate_cap,ib) || !reserve(&ctx->up,&ctx->up_cap,ib)) return 0;
    if (!cuda_ok(cudaMemcpy(ctx->x,x,xb,cudaMemcpyHostToDevice),"expert input upload")) return 0;
    dim3 hidden_grid((unsigned)I,(unsigned)S), output_grid((unsigned)D,(unsigned)S);
    quant_matmul<<<hidden_grid,256>>>(ctx->gate,ctx->x,gate->weights,gate->scales,
        gate->fmt,S,D,I,row_bytes(gate->fmt,D),gate->wrapped);
    quant_matmul<<<hidden_grid,256>>>(ctx->up,ctx->x,up->weights,up->scales,
        up->fmt,S,D,I,row_bytes(up->fmt,D),up->wrapped);
    size_t n=(size_t)S*I;
    silu_mul<<<(unsigned)((n+255)/256),256>>>(ctx->gate,ctx->up,n);
    quant_matmul<<<output_grid,256>>>(ctx->y,ctx->gate,down->weights,down->scales,
        down->fmt,S,I,D,row_bytes(down->fmt,I),down->wrapped);
    if (!cuda_ok(cudaGetLastError(),"expert MLP launch") ||
        !cuda_ok(cudaMemcpy(y,ctx->y,yb,cudaMemcpyDeviceToHost),"expert output download")) return 0;
    return 1;
}

extern "C" int coli_cuda_shared_mlp_w4a16(ColiCudaTensor *gate,ColiCudaTensor *up,
        ColiCudaTensor *down,float *y,const float *x,int S){
    if(!gate||!up||!down||!x||!y||S<1||gate->fmt!=2||up->fmt!=2||down->fmt!=2||
       gate->device!=up->device||gate->device!=down->device||gate->I!=up->I||
       gate->O!=up->O||down->I!=gate->O||down->O!=gate->I)return 0;
    DeviceContext *ctx=find_ctx(gate->device);if(!select_ctx(ctx)||ctx->compute_major<7)return 0;
    int D=gate->I,I=gate->O;size_t xb=(size_t)S*D*sizeof(float),ib=(size_t)S*I*sizeof(float);
    if(!reserve(&ctx->x,&ctx->x_cap,xb)||!reserve(&ctx->gate,&ctx->gate_cap,ib)||
       !reserve(&ctx->up,&ctx->up_cap,ib)||!reserve(&ctx->y,&ctx->y_cap,xb)||
       !reserve_pinned(&ctx->host_x,&ctx->host_x_cap,xb)||
       !reserve_pinned(&ctx->host_y,&ctx->host_y_cap,xb))return 0;
    std::memcpy(ctx->host_x,x,xb);
    if(!cuda_ok(cudaMemcpyAsync(ctx->x,ctx->host_x,xb,cudaMemcpyHostToDevice,ctx->stream),
                               "shared w4a16 input upload"))return 0;
    dim3 hidden((unsigned)((I+63)/64),(unsigned)((S+15)/16));
    dim3 output((unsigned)((D+63)/64),(unsigned)((S+15)/16));
    w4a16_gate_up<<<hidden,256,0,ctx->stream>>>(ctx->gate,ctx->up,ctx->x,
        (const uint8_t*)gate->weights,(const uint8_t*)up->weights,gate->scales,up->scales,S,D,I);
    silu_mul<<<(unsigned)(((size_t)S*I+255)/256),256,0,ctx->stream>>>(ctx->gate,ctx->up,(size_t)S*I);
    w4a16_matmul<<<output,128,0,ctx->stream>>>(ctx->y,ctx->gate,(const uint8_t*)down->weights,down->scales,S,I,D);
    if(!cuda_ok(cudaGetLastError(),"shared w4a16 launch")||
       !cuda_ok(cudaMemcpyAsync(ctx->host_y,ctx->y,xb,cudaMemcpyDeviceToHost,ctx->stream),
                               "shared w4a16 output download")||
       !cuda_ok(cudaStreamSynchronize(ctx->stream),"shared w4a16 synchronize"))return 0;
    std::memcpy(y,ctx->host_y,xb);
    return 1;
}

/* Tiled FP8 (e4m3 weights, fp16 activations) expert FFN — the tensor-core replacement
 * for coli_cuda_expert_mlp/quant_matmul. Same signature; requires fmt==4 on all three
 * projections and compute>=7. Weights read ONCE per 16-row tile (vs quant_matmul's
 * S-fold re-read), so it is a strict prefill win that grows with S. */
extern "C" int coli_cuda_expert_mlp_fp8(ColiCudaTensor *gate,ColiCudaTensor *up,
        ColiCudaTensor *down,float *y,const float *x,int S){
    if(!gate||!up||!down||!x||!y||S<1||gate->fmt!=4||up->fmt!=4||down->fmt!=4||
       gate->device!=up->device||gate->device!=down->device||gate->I!=up->I||
       gate->O!=up->O||down->I!=gate->O||down->O!=gate->I)return 0;
    DeviceContext *ctx=find_ctx(gate->device);if(!select_ctx(ctx)||ctx->compute_major<7)return 0;
    int D=gate->I,I=gate->O;size_t xb=(size_t)S*D*sizeof(float),ib=(size_t)S*I*sizeof(float);
    if(!reserve(&ctx->x,&ctx->x_cap,xb)||!reserve(&ctx->gate,&ctx->gate_cap,ib)||
       !reserve(&ctx->up,&ctx->up_cap,ib)||!reserve(&ctx->y,&ctx->y_cap,xb)||
       !reserve_pinned(&ctx->host_x,&ctx->host_x_cap,xb)||
       !reserve_pinned(&ctx->host_y,&ctx->host_y_cap,xb))return 0;
    // Optional per-call GPU-time accounting (COLI_FFN_EVT=1): times just the kernel
    // trio via events, accumulates, and prints running totals + row count to compare
    // against the CPU-side wall-time (GPUFFN_US). Diagnostic only.
    static int s_evt=-1; static cudaEvent_t s_e0=0,s_e1=0;
    static double s_kms=0; static long s_calls=0,s_rows=0;
    if(s_evt<0){ const char*e=getenv("COLI_FFN_EVT"); s_evt=e&&atoi(e); if(s_evt){cudaEventCreate(&s_e0);cudaEventCreate(&s_e1);} }
    std::memcpy(ctx->host_x,x,xb);
    if(!cuda_ok(cudaMemcpyAsync(ctx->x,ctx->host_x,xb,cudaMemcpyHostToDevice,ctx->stream),
                               "expert fp8 input upload"))return 0;
    dim3 hidden((unsigned)((I+63)/64),(unsigned)((S+15)/16));
    dim3 output((unsigned)((D+63)/64),(unsigned)((S+15)/16));
    // Expert weights live in pool-recycled host buffers that `pread` just wrote, so a
    // zero-copy GPU read pays a cache-coherence penalty on every (dirty) weight line —
    // measured ~2.8x/matmul slower than reading clean device memory. Stage them through
    // one streaming H2D copy per weight (resolves coherence in bulk), then run the
    // kernels on device pointers. Prefill-gated (COLI_FFN_DEVCOPY_MIN, default 16): at
    // small S the copy can't amortize. COLI_FFN_DEVCOPY=0 forces the old zero-copy path.
    const uint8_t *gw=(const uint8_t*)gate->weights,*uw=(const uint8_t*)up->weights,*dw=(const uint8_t*)down->weights;
    const float *gsc=gate->scales,*usc=up->scales,*dsc=down->scales;
    static int s_dc=-1,s_dcmin=16;
    if(s_dc<0){const char*e=getenv("COLI_FFN_DEVCOPY");s_dc=(!e||atoi(e));const char*m=getenv("COLI_FFN_DEVCOPY_MIN");if(m)s_dcmin=atoi(m);}
    if(s_dc&&S>=s_dcmin){
        size_t gwb=(size_t)I*D,dwb=(size_t)D*I;
        if(reserve_bytes((void**)&ctx->ewg,&ctx->ewg_cap,gwb)&&reserve_bytes((void**)&ctx->ewu,&ctx->ewu_cap,gwb)&&
           reserve_bytes((void**)&ctx->ewd,&ctx->ewd_cap,dwb)&&reserve(&ctx->esg,&ctx->esg_cap,(size_t)I*sizeof(float))&&
           reserve(&ctx->esu,&ctx->esu_cap,(size_t)I*sizeof(float))&&reserve(&ctx->esd,&ctx->esd_cap,(size_t)D*sizeof(float))){
            cudaMemcpyAsync(ctx->ewg,gw,gwb,cudaMemcpyHostToDevice,ctx->stream);
            cudaMemcpyAsync(ctx->ewu,uw,gwb,cudaMemcpyHostToDevice,ctx->stream);
            cudaMemcpyAsync(ctx->ewd,dw,dwb,cudaMemcpyHostToDevice,ctx->stream);
            cudaMemcpyAsync(ctx->esg,gsc,(size_t)I*sizeof(float),cudaMemcpyHostToDevice,ctx->stream);
            cudaMemcpyAsync(ctx->esu,usc,(size_t)I*sizeof(float),cudaMemcpyHostToDevice,ctx->stream);
            cudaMemcpyAsync(ctx->esd,dsc,(size_t)D*sizeof(float),cudaMemcpyHostToDevice,ctx->stream);
            gw=ctx->ewg;uw=ctx->ewu;dw=ctx->ewd;gsc=ctx->esg;usc=ctx->esu;dsc=ctx->esd;
        }
    }
    if(s_evt) cudaEventRecord(s_e0,ctx->stream);
    fp8a16_gate_up<<<hidden,256,0,ctx->stream>>>(ctx->gate,ctx->up,ctx->x,gw,uw,gsc,usc,S,D,I);
    silu_mul<<<(unsigned)(((size_t)S*I+255)/256),256,0,ctx->stream>>>(ctx->gate,ctx->up,(size_t)S*I);
    fp8a16_matmul<<<output,128,0,ctx->stream>>>(ctx->y,ctx->gate,dw,dsc,S,I,D);
    if(s_evt) cudaEventRecord(s_e1,ctx->stream);
    if(!cuda_ok(cudaGetLastError(),"expert fp8 launch")||
       !cuda_ok(cudaMemcpyAsync(ctx->host_y,ctx->y,xb,cudaMemcpyDeviceToHost,ctx->stream),
                               "expert fp8 output download")||
       !cuda_ok(cudaStreamSynchronize(ctx->stream),"expert fp8 synchronize"))return 0;
    if(s_evt){ float km=0; cudaEventElapsedTime(&km,s_e0,s_e1); s_kms+=km; s_calls++; s_rows+=S;
        if(s_calls%3000==0) fprintf(stderr,"[ffn-evt] calls=%ld rows=%ld kernel_gpu=%.1fs avg_kernel=%.3fms avg_rows=%.1f\n",
            s_calls,s_rows,s_kms/1e3,s_kms/s_calls,(double)s_rows/s_calls); }
    std::memcpy(y,ctx->host_y,xb);
    return 1;
}

/* Tiled int8 (W8A16) expert/MLP FFN — the tensor-core replacement for quant_matmul on
 * resident int8 weights (the shared expert). Same contract as coli_cuda_expert_mlp but
 * requires fmt==1 (int8) and compute>=7; weights read once per 16-row tile. */
extern "C" int coli_cuda_expert_mlp_i8a16(ColiCudaTensor *gate,ColiCudaTensor *up,
        ColiCudaTensor *down,float *y,const float *x,int S){
    if(!gate||!up||!down||!x||!y||S<1||gate->fmt!=1||up->fmt!=1||down->fmt!=1||
       gate->device!=up->device||gate->device!=down->device||gate->I!=up->I||
       gate->O!=up->O||down->I!=gate->O||down->O!=gate->I)return 0;
    DeviceContext *ctx=find_ctx(gate->device);if(!select_ctx(ctx)||ctx->compute_major<7)return 0;
    int D=gate->I,I=gate->O;size_t xb=(size_t)S*D*sizeof(float),ib=(size_t)S*I*sizeof(float);
    if(!reserve(&ctx->x,&ctx->x_cap,xb)||!reserve(&ctx->gate,&ctx->gate_cap,ib)||
       !reserve(&ctx->up,&ctx->up_cap,ib)||!reserve(&ctx->y,&ctx->y_cap,xb)||
       !reserve_pinned(&ctx->host_x,&ctx->host_x_cap,xb)||
       !reserve_pinned(&ctx->host_y,&ctx->host_y_cap,xb))return 0;
    std::memcpy(ctx->host_x,x,xb);
    if(!cuda_ok(cudaMemcpyAsync(ctx->x,ctx->host_x,xb,cudaMemcpyHostToDevice,ctx->stream),
                               "expert i8 input upload"))return 0;
    dim3 hidden((unsigned)((I+63)/64),(unsigned)((S+15)/16));
    dim3 output((unsigned)((D+63)/64),(unsigned)((S+15)/16));
    i8a16_gate_up<<<hidden,256,0,ctx->stream>>>(ctx->gate,ctx->up,ctx->x,
        (const uint8_t*)gate->weights,(const uint8_t*)up->weights,gate->scales,up->scales,S,D,I);
    silu_mul<<<(unsigned)(((size_t)S*I+255)/256),256,0,ctx->stream>>>(ctx->gate,ctx->up,(size_t)S*I);
    i8a16_matmul<<<output,128,0,ctx->stream>>>(ctx->y,ctx->gate,(const uint8_t*)down->weights,down->scales,S,I,D);
    if(!cuda_ok(cudaGetLastError(),"expert i8 launch")||
       !cuda_ok(cudaMemcpyAsync(ctx->host_y,ctx->y,xb,cudaMemcpyDeviceToHost,ctx->stream),
                               "expert i8 output download")||
       !cuda_ok(cudaStreamSynchronize(ctx->stream),"expert i8 synchronize"))return 0;
    std::memcpy(y,ctx->host_y,xb);
    return 1;
}

extern "C" int coli_cuda_expert_group(ColiCudaTensor *const *gates,
                                        ColiCudaTensor *const *ups,
                                        ColiCudaTensor *const *downs,
                                        const int *rows, int count,
                                        float *y, const float *x) {
    if (!gates || !ups || !downs || !rows || !x || !y || count < 1) return 0;
    ColiCudaTensor *first=gates[0];
    if (!first) return 0;
    int device=first->device,D=first->I,I=first->O,total=0,max_rows=0;
    GroupDesc host[64]; if(count>64) return 0;
    int all_s4=1, all_fp8=1;
    for(int c=0;c<count;c++){
        ColiCudaTensor *g=gates[c],*u=ups[c],*d=downs[c];
        if(!g||!u||!d||rows[c]<1||g->device!=device||u->device!=device||d->device!=device||
           g->I!=D||u->I!=D||g->O!=I||u->O!=I||d->I!=I||d->O!=D) return 0;
        host[c]={g->weights,u->weights,d->weights,g->scales,u->scales,d->scales,
                 g->fmt,u->fmt,d->fmt,rows[c],total,g->wrapped};
        all_s4&=g->fmt==2&&u->fmt==2&&d->fmt==2;
        all_fp8&=g->fmt==4&&u->fmt==4&&d->fmt==4;
        total+=rows[c]; if(rows[c]>max_rows) max_rows=rows[c];
    }
    DeviceContext *ctx=find_ctx(device); if(!select_ctx(ctx)) return 0;
    size_t xb=(size_t)total*D*sizeof(float), ib=(size_t)total*I*sizeof(float);
    if(!reserve(&ctx->x,&ctx->x_cap,xb)||!reserve(&ctx->y,&ctx->y_cap,xb)||
       !reserve(&ctx->gate,&ctx->gate_cap,ib)||!reserve(&ctx->up,&ctx->up_cap,ib)||
       !reserve_bytes(&ctx->group_desc,&ctx->group_desc_cap,(size_t)count*sizeof(GroupDesc))) return 0;
    int async=!getenv("COLI_CUDA_ASYNC")||atoi(getenv("COLI_CUDA_ASYNC"));
    if(async&&(!reserve_pinned(&ctx->host_x,&ctx->host_x_cap,xb)||
               !reserve_pinned(&ctx->host_y,&ctx->host_y_cap,xb)))return 0;
    cudaError_t copy_desc=async?cudaMemcpyAsync(ctx->group_desc,host,(size_t)count*sizeof(GroupDesc),
                                                cudaMemcpyHostToDevice,ctx->stream)
                               :cudaMemcpy(ctx->group_desc,host,(size_t)count*sizeof(GroupDesc),cudaMemcpyHostToDevice);
    if(!cuda_ok(copy_desc,"expert group descriptors"))return 0;
    int profile=getenv("COLI_CUDA_PROFILE")&&atoi(getenv("COLI_CUDA_PROFILE"));
    cudaEvent_t ev[4]={};
    if(profile) for(int i=0;i<4;i++) if(!cuda_ok(cudaEventCreate(&ev[i]),"profile event")) profile=0;
    if(profile) cudaEventRecord(ev[0],ctx->stream);
    if(async)std::memcpy(ctx->host_x,x,xb);
    cudaError_t copy_x=async?cudaMemcpyAsync(ctx->x,ctx->host_x,xb,cudaMemcpyHostToDevice,ctx->stream)
                            :cudaMemcpy(ctx->x,x,xb,cudaMemcpyHostToDevice);
    if(!cuda_ok(copy_x,"expert group input upload")) return 0;
    if(profile) cudaEventRecord(ev[1],ctx->stream);
    GroupDesc *dev=(GroupDesc*)ctx->group_desc;
    int tc=getenv("COLI_CUDA_TC_INT4")&&atoi(getenv("COLI_CUDA_TC_INT4"));
    tc=tc&&all_s4&&D%32==0&&I%32==0&&D%8==0&&I%8==0;
    int tc_min=getenv("COLI_CUDA_TC_MIN_ROWS")?atoi(getenv("COLI_CUDA_TC_MIN_ROWS")):8;
    for(int c=0;c<count&&tc;c++)tc=rows[c]>=tc_min;
    if(tc){
        size_t qb=(size_t)(total+7)*(size_t)(D>I?D:I)/2;
        if(!reserve_bytes((void**)&ctx->qx,&ctx->qx_cap,qb)||
           !reserve(&ctx->qscale,&ctx->qscale_cap,(size_t)(total+7)*sizeof(float)))return 0;
        cudaMemsetAsync(ctx->qx,0,qb,ctx->stream);
        quantize_s4_rows<<<total,256,0,ctx->stream>>>(ctx->qx,ctx->qscale,ctx->x,total,D);
        grouped_s4_wmma<<<dim3((unsigned)((I+63)/64),(unsigned)count),256,0,ctx->stream>>>(ctx->gate,ctx->qx,ctx->qscale,dev,D,I,0);
        grouped_s4_wmma<<<dim3((unsigned)((I+63)/64),(unsigned)count),256,0,ctx->stream>>>(ctx->up,ctx->qx,ctx->qscale,dev,D,I,1);
        silu_mul<<<(unsigned)(((size_t)total*I+255)/256),256,0,ctx->stream>>>(ctx->gate,ctx->up,(size_t)total*I);
        quantize_s4_rows<<<total,256,0,ctx->stream>>>(ctx->qx,ctx->qscale,ctx->gate,total,I);
        grouped_s4_wmma<<<dim3((unsigned)((D+63)/64),(unsigned)count),256,0,ctx->stream>>>(ctx->y,ctx->qx,ctx->qscale,dev,I,D,2);
    }else if(all_fp8&&ctx->compute_major>=7){
        /* FP8 (e4m3) tiled Tensor Core, one launch trio per expert on the stream —
         * the whole group shares ONE H2D + ONE D2H, so the per-expert synchronous
         * upload/download round-trip (which dominates moe-compute) is paid once for
         * the layer instead of once per expert. */
        int off8=0;
        for(int c=0;c<count;c++){
            int r=rows[c];
            float *g8=ctx->gate+(size_t)off8*I,*u8=ctx->up+(size_t)off8*I;
            float *x8=ctx->x+(size_t)off8*D,*y8=ctx->y+(size_t)off8*D;
            dim3 hg8((unsigned)((I+63)/64),(unsigned)((r+15)/16));
            dim3 og8((unsigned)((D+63)/64),(unsigned)((r+15)/16));
            fp8a16_gate_up<<<hg8,256,0,ctx->stream>>>(g8,u8,x8,
                (const uint8_t*)host[c].g,(const uint8_t*)host[c].u,host[c].gs,host[c].us,r,D,I);
            silu_mul<<<(unsigned)(((size_t)r*I+255)/256),256,0,ctx->stream>>>(g8,u8,(size_t)r*I);
            fp8a16_matmul<<<og8,128,0,ctx->stream>>>(y8,g8,
                (const uint8_t*)host[c].d,host[c].ds,r,I,D);
            off8+=r;
        }
    }else if(all_s4&&ctx->compute_major>=7&&getenv("COLI_CUDA_TC_W4A16")&&
             atoi(getenv("COLI_CUDA_TC_W4A16"))){
        /* W4A16 Tensor Core per gruppo: attivazioni fp16 per tile (lossless al
         * contrario del path W4A4), un lancio per expert dentro lo stream —
         * l'overhead di lancio e' trascurabile rispetto ai GEMM. */
        int tc16_min=getenv("COLI_CUDA_TC_W4A16_MIN")?atoi(getenv("COLI_CUDA_TC_W4A16_MIN")):16;
        int off16=0;
        for(int c=0;c<count;c++){
            int r=rows[c];
            float *g16=ctx->gate+(size_t)off16*I,*u16=ctx->up+(size_t)off16*I;
            float *x16=ctx->x+(size_t)off16*D,*y16=ctx->y+(size_t)off16*D;
            if(r>=tc16_min){
                dim3 hg16((unsigned)((I+63)/64),(unsigned)((r+15)/16));
                dim3 og16((unsigned)((D+63)/64),(unsigned)((r+15)/16));
                w4a16_gate_up<<<hg16,256,0,ctx->stream>>>(g16,u16,x16,
                    (const uint8_t*)host[c].g,(const uint8_t*)host[c].u,host[c].gs,host[c].us,r,D,I);
                silu_mul<<<(unsigned)(((size_t)r*I+255)/256),256,0,ctx->stream>>>(g16,u16,(size_t)r*I);
                w4a16_matmul<<<og16,128,0,ctx->stream>>>(y16,g16,
                    (const uint8_t*)host[c].d,host[c].ds,r,I,D);
            }else{
                /* piccoli batch: tile TC quasi vuoti + overhead di lancio — il
                 * kernel naive per-elemento resta piu' veloce (misurato in decode) */
                quant_matmul<<<dim3((unsigned)I,(unsigned)r),256,0,ctx->stream>>>(g16,x16,
                    host[c].g,host[c].gs,host[c].gf,r,D,I,row_bytes(host[c].gf,D),host[c].wrapped);
                quant_matmul<<<dim3((unsigned)I,(unsigned)r),256,0,ctx->stream>>>(u16,x16,
                    host[c].u,host[c].us,host[c].uf,r,D,I,row_bytes(host[c].uf,D),host[c].wrapped);
                silu_mul<<<(unsigned)(((size_t)r*I+255)/256),256,0,ctx->stream>>>(g16,u16,(size_t)r*I);
                quant_matmul<<<dim3((unsigned)D,(unsigned)r),256,0,ctx->stream>>>(y16,g16,
                    host[c].d,host[c].ds,host[c].df,r,I,D,row_bytes(host[c].df,I),host[c].wrapped);
            }
            off16+=r;
        }
    }else if(all_s4&&(!getenv("COLI_CUDA_W4_PACKED")||atoi(getenv("COLI_CUDA_W4_PACKED")))){
        dim3 hg((unsigned)I,(unsigned)max_rows,(unsigned)count),og((unsigned)D,(unsigned)max_rows,(unsigned)count);
        int dual=!getenv("COLI_CUDA_DUAL_PROJ")||atoi(getenv("COLI_CUDA_DUAL_PROJ"));
        if(dual)grouped_hidden_w4_dual<<<hg,256,0,ctx->stream>>>(ctx->gate,ctx->up,ctx->x,dev,I,D);
        else{
            grouped_hidden_w4<<<hg,256,0,ctx->stream>>>(ctx->gate,ctx->x,dev,I,D,0);
            grouped_hidden_w4<<<hg,256,0,ctx->stream>>>(ctx->up,ctx->x,dev,I,D,1);
        }
        silu_mul<<<(unsigned)(((size_t)total*I+255)/256),256,0,ctx->stream>>>(ctx->gate,ctx->up,(size_t)total*I);
        grouped_down_w4<<<og,256,0,ctx->stream>>>(ctx->y,ctx->gate,dev,D,I);
    }else{
        dim3 hg((unsigned)I,(unsigned)max_rows,(unsigned)count),og((unsigned)D,(unsigned)max_rows,(unsigned)count);
        grouped_hidden<<<hg,256,0,ctx->stream>>>(ctx->gate,ctx->x,dev,I,D,0);
        grouped_hidden<<<hg,256,0,ctx->stream>>>(ctx->up,ctx->x,dev,I,D,1);
        silu_mul<<<(unsigned)(((size_t)total*I+255)/256),256,0,ctx->stream>>>(ctx->gate,ctx->up,(size_t)total*I);
        grouped_down<<<og,256,0,ctx->stream>>>(ctx->y,ctx->gate,dev,D,I);
    }
    if(profile) cudaEventRecord(ev[2],ctx->stream);
    if(!async&&!cuda_ok(cudaStreamSynchronize(ctx->stream),"expert group synchronize"))return 0;
    cudaError_t copy_y=async?cudaMemcpyAsync(ctx->host_y,ctx->y,xb,cudaMemcpyDeviceToHost,ctx->stream)
                            :cudaMemcpy(y,ctx->y,xb,cudaMemcpyDeviceToHost);
    if(!cuda_ok(cudaGetLastError(),"expert group launch")||!cuda_ok(copy_y,"expert group output download"))return 0;
    if(async){if(!cuda_ok(cudaStreamSynchronize(ctx->stream),"expert group synchronize"))return 0;
        std::memcpy(y,ctx->host_y,xb);}
    if(profile){
        cudaEventRecord(ev[3],ctx->stream); cudaEventSynchronize(ev[3]); float a=0,b=0,c=0;
        cudaEventElapsedTime(&a,ev[0],ev[1]); cudaEventElapsedTime(&b,ev[1],ev[2]);
        cudaEventElapsedTime(&c,ev[2],ev[3]);
        { std::lock_guard<std::mutex> lock(g_group_stats_mu);
          g_group_h2d_ms+=a; g_group_kernel_ms+=b; g_group_d2h_ms+=c; }
        for(int i=0;i<4;i++) cudaEventDestroy(ev[i]);
    }
    { std::lock_guard<std::mutex> lock(g_group_stats_mu);
      g_group_calls++; g_group_experts+=(uint64_t)count; g_group_rows+=(uint64_t)total; }
    return 1;
}


extern "C" int coli_cuda_attention_absorb(ColiCudaTensor *w,float *ctx,const float *q,
                                            const float *latent,const float *rope,int H,int Q,
                                            int R,int V,int K,int T,float scale){
    if(!w||!ctx||!q||!latent||!rope||H<1||Q<1||R<1||V<1||K<1||K>512||T<1||T>4096||
       w->I!=K||w->O!=H*(Q+V))return 0;
    DeviceContext *dc=find_ctx(w->device);if(!select_ctx(dc))return 0;
    size_t qb=(size_t)H*(Q+R)*sizeof(float),lb=(size_t)T*K*sizeof(float);
    size_t rb=(size_t)T*R*sizeof(float),cb=(size_t)H*V*sizeof(float);
    if(!reserve(&dc->aq,&dc->aq_cap,qb)||!reserve(&dc->al,&dc->al_cap,lb)||
       !reserve(&dc->ar,&dc->ar_cap,rb)||!reserve(&dc->ac,&dc->ac_cap,cb))return 0;
    if(!cuda_ok(cudaMemcpyAsync(dc->aq,q,qb,cudaMemcpyHostToDevice,dc->stream),"attention q upload")||
       !cuda_ok(cudaMemcpyAsync(dc->al,latent,lb,cudaMemcpyHostToDevice,dc->stream),"attention latent upload")||
       !cuda_ok(cudaMemcpyAsync(dc->ar,rope,rb,cudaMemcpyHostToDevice,dc->stream),"attention rope upload"))return 0;
    size_t shared=(size_t)(2*K+T)*sizeof(float);
    attention_absorb_kernel<<<H,256,shared,dc->stream>>>(dc->ac,dc->aq,dc->al,dc->ar,w->weights,w->scales,
        w->fmt,H,Q,R,V,K,T,scale);
    if(!cuda_ok(cudaGetLastError(),"attention absorb launch")||
       !cuda_ok(cudaMemcpyAsync(ctx,dc->ac,cb,cudaMemcpyDeviceToHost,dc->stream),"attention context download")||
       !cuda_ok(cudaStreamSynchronize(dc->stream),"attention synchronize"))return 0;
    return 1;
}

static int attention_absorb_batch_run(ColiCudaTensor *w,ColiCudaTensor *proj,float *out,
        const float *q,const float *latent,const float *rope,int S,int H,int Q,int R,int V,
        int K,int T,float scale){
    if(!w||!out||!q||!latent||!rope||S<1||H<1||Q<1||R<1||V<1||K<1||K>512||
       T<S||T>8192||w->I!=K||w->O!=H*(Q+V))return 0;
    if(proj&&(proj->device!=w->device||proj->I!=H*V))return 0;
    DeviceContext *dc=find_ctx(w->device);if(!select_ctx(dc))return 0;
    size_t qb=(size_t)S*H*(Q+R)*sizeof(float),lb=(size_t)T*K*sizeof(float);
    size_t rb=(size_t)T*R*sizeof(float),cb=(size_t)S*H*V*sizeof(float);
    if(!reserve(&dc->aq,&dc->aq_cap,qb)||!reserve(&dc->al,&dc->al_cap,lb)||
       !reserve(&dc->ar,&dc->ar_cap,rb)||!reserve(&dc->ac,&dc->ac_cap,cb))return 0;
    if(!cuda_ok(cudaMemcpyAsync(dc->aq,q,qb,cudaMemcpyHostToDevice,dc->stream),"attention batch q upload")||
       !cuda_ok(cudaMemcpyAsync(dc->al,latent,lb,cudaMemcpyHostToDevice,dc->stream),"attention batch latent upload")||
       !cuda_ok(cudaMemcpyAsync(dc->ar,rope,rb,cudaMemcpyHostToDevice,dc->stream),"attention batch rope upload"))return 0;
    size_t shared=(size_t)(2*K+T+ATTN_TPB)*sizeof(float);
    attention_absorb_batch_kernel<<<dim3(H,S),ATTN_TPB,shared,dc->stream>>>(dc->ac,dc->aq,dc->al,
        dc->ar,w->weights,w->scales,w->fmt,S,H,Q,R,V,K,T,scale);
    if(!cuda_ok(cudaGetLastError(),"attention batch launch"))return 0;
    const float *src=dc->ac;size_t ob=cb;
    if(proj){
        ob=(size_t)S*proj->O*sizeof(float);if(!reserve(&dc->y,&dc->y_cap,ob))return 0;
        quant_matmul<<<dim3(proj->O,S),256,0,dc->stream>>>(dc->y,dc->ac,proj->weights,
            proj->scales,proj->fmt,S,proj->I,proj->O,row_bytes(proj->fmt,proj->I),proj->wrapped);
        if(!cuda_ok(cudaGetLastError(),"attention o_proj launch"))return 0;src=dc->y;
    }
    if(!cuda_ok(cudaMemcpyAsync(out,src,ob,cudaMemcpyDeviceToHost,dc->stream),
                               proj?"attention projected output download":"attention batch context download")||
       !cuda_ok(cudaStreamSynchronize(dc->stream),"attention batch synchronize"))return 0;
    return 1;
}

extern "C" int coli_cuda_attention_absorb_batch(ColiCudaTensor *w,float *ctx,const float *q,
        const float *latent,const float *rope,int S,int H,int Q,int R,int V,int K,int T,
        float scale){
    return attention_absorb_batch_run(w,nullptr,ctx,q,latent,rope,S,H,Q,R,V,K,T,scale);
}

/* DSA sparse prefill attention. Mirrors attention_absorb_batch_run but uploads the
 * per-query indexer selection (`sel_idx` is [S, maxsel] int, `sel_cnt` is [S] int)
 * and dispatches attention_absorb_sparse_kernel. `maxsel` must be `index_topk` (the
 * kernel's is_dense fallback relies on dense queries having nt <= maxsel). Larger T
 * than the dense path is fine — shared memory is sized to maxsel, not T. */
/* Host entry for the DSA indexer scores (declared after `reserve`). Reuses the
 * attention scratch — the indexer and the attention core run sequentially within a
 * layer and each uploads/downloads inside one synchronized call. */
extern "C" int coli_cuda_dsa_indexer_scores(float *scores,const float *qi,const float *hw,
        const float *keys,int nsp,int s0,int nh,int hd,int T,int pos_base,int device){
    if(!scores||!qi||!hw||!keys||nsp<1||nh<1||nh>32||hd<1||T<1)return 0;
    DeviceContext *dc=find_ctx(device);if(!select_ctx(dc))return 0;
    size_t qb=(size_t)nsp*nh*hd*sizeof(float),wb=(size_t)nsp*nh*sizeof(float);
    size_t kb=(size_t)T*hd*sizeof(float),sb=(size_t)nsp*T*sizeof(float);
    if(!reserve(&dc->aq,&dc->aq_cap,qb)||!reserve(&dc->ar,&dc->ar_cap,wb)||
       !reserve(&dc->al,&dc->al_cap,kb)||!reserve(&dc->ac,&dc->ac_cap,sb))return 0;
    if(!cuda_ok(cudaMemcpyAsync(dc->aq,qi,qb,cudaMemcpyHostToDevice,dc->stream),"dsa qi")||
       !cuda_ok(cudaMemcpyAsync(dc->ar,hw,wb,cudaMemcpyHostToDevice,dc->stream),"dsa hw")||
       !cuda_ok(cudaMemcpyAsync(dc->al,keys,kb,cudaMemcpyHostToDevice,dc->stream),"dsa keys"))return 0;
    size_t sh=((size_t)nh*hd+nh)*sizeof(float);
    if(sh>96*1024)return 0;
    dsa_indexer_scores<<<(unsigned)nsp,256,sh,dc->stream>>>(dc->ac,dc->aq,dc->ar,dc->al,nsp,s0,nh,hd,T,pos_base);
    if(!cuda_ok(cudaGetLastError(),"dsa indexer scores launch"))return 0;
    if(!cuda_ok(cudaMemcpyAsync(scores,dc->ac,sb,cudaMemcpyDeviceToHost,dc->stream),"dsa scores download")||
       !cuda_ok(cudaStreamSynchronize(dc->stream),"dsa scores sync"))return 0;
    return 1;
}

/* Tensor-core sparse-attention path (COLI_TC_ATTN=1): build QA/KB (fp16) + the DSA key
 * bitmask, then run the WMMA flash kernel. Same [S,H,V] ctx output as the scalar run
 * (partial head slice zeroes the rest); no fused o_proj. */
static int tc_sparse_attn_run(ColiCudaTensor *w,float *out,const float *q,const float *latent,const float *rope,
        const int *sel_idx,const int *sel_cnt,int maxsel,int H0,int HC,int S,int H,int Q,int R,int V,int K,int T,float scale){
    if(H0<0||HC<1||H0+HC>H||K<1||K>512||T<S)return 0;
    DeviceContext *dc=find_ctx(w->device);if(!select_ctx(dc))return 0;
    int KR=K+R; size_t mr=(T+7)/8;
    size_t qb=(size_t)S*H*(Q+R)*4,lb=(size_t)T*K*4,rbb=(size_t)T*R*4,cb=(size_t)S*H*V*4;
    size_t sib=(size_t)S*maxsel*4,scb=(size_t)S*4;
    size_t qab=(size_t)S*H*KR*2,kbb=(size_t)T*KR*2,mskb=(size_t)S*mr;
    if(!reserve(&dc->aq,&dc->aq_cap,qb)||!reserve(&dc->al,&dc->al_cap,lb)||!reserve(&dc->ar,&dc->ar_cap,rbb)||
       !reserve(&dc->ac,&dc->ac_cap,cb)||!reserve_bytes(&dc->asel,&dc->asel_cap,sib)||!reserve_bytes(&dc->acnt,&dc->acnt_cap,scb)||
       !reserve_bytes(&dc->aqa,&dc->aqa_cap,qab)||!reserve_bytes(&dc->akb,&dc->akb_cap,kbb)||!reserve_bytes(&dc->amsk,&dc->amsk_cap,mskb))return 0;
    if(!cuda_ok(cudaMemcpyAsync(dc->aq,q,qb,cudaMemcpyHostToDevice,dc->stream),"tc attn q")||
       !cuda_ok(cudaMemcpyAsync(dc->al,latent,lb,cudaMemcpyHostToDevice,dc->stream),"tc attn latent")||
       !cuda_ok(cudaMemcpyAsync(dc->ar,rope,rbb,cudaMemcpyHostToDevice,dc->stream),"tc attn rope")||
       !cuda_ok(cudaMemcpyAsync(dc->asel,sel_idx,sib,cudaMemcpyHostToDevice,dc->stream),"tc attn sel")||
       !cuda_ok(cudaMemcpyAsync(dc->acnt,sel_cnt,scb,cudaMemcpyHostToDevice,dc->stream),"tc attn cnt"))return 0;
    if(!cuda_ok(cudaMemsetAsync(dc->amsk,0,mskb,dc->stream),"tc attn mask zero"))return 0;
    if((H0!=0||HC!=H)&&!cuda_ok(cudaMemsetAsync(dc->ac,0,cb,dc->stream),"tc attn ctx zero"))return 0;
    tc_build_mask<<<(unsigned)(S+255)/256,256,0,dc->stream>>>((uint8_t*)dc->amsk,(const int*)dc->asel,(const int*)dc->acnt,maxsel,S,T);
    tc_build_kb<<<(unsigned)T,256,0,dc->stream>>>((__half*)dc->akb,dc->al,dc->ar,K,R,T);
    tc_build_qa<<<dim3(HC,S),256,0,dc->stream>>>((__half*)dc->aqa,dc->aq,w->weights,w->scales,w->fmt,H0,S,H,Q,R,V,K,scale);
    if(!cuda_ok(cudaGetLastError(),"tc attn prep launch"))return 0;
    size_t shW=(size_t)ATC_QT*(4*KR+4*K);
    if(!cuda_ok(cudaFuncSetAttribute(tc_sparse_attn,cudaFuncAttributeMaxDynamicSharedMemorySize,(int)shW),"tc attn shared attr"))return 0;
    tc_sparse_attn<<<dim3(HC,(S+ATC_QT-1)/ATC_QT),256,shW,dc->stream>>>((float*)dc->ac,(const __half*)dc->aqa,
        (const __half*)dc->akb,dc->al,w->weights,w->scales,(const uint8_t*)dc->amsk,(const int*)dc->acnt,w->fmt,H0,S,H,Q,R,V,K,T);
    if(!cuda_ok(cudaGetLastError(),"tc attn launch"))return 0;
    if(!cuda_ok(cudaMemcpyAsync(out,dc->ac,cb,cudaMemcpyDeviceToHost,dc->stream),"tc attn ctx download")||
       !cuda_ok(cudaStreamSynchronize(dc->stream),"tc attn sync"))return 0;
    return 1;
}

static int attention_absorb_sparse_run(ColiCudaTensor *w,ColiCudaTensor *proj,float *out,
        const float *q,const float *latent,const float *rope,
        const int *sel_idx,const int *sel_cnt,int maxsel,
        int H0,int HC,int S,int H,int Q,int R,int V,int K,int T,float scale){
    if(!w||!out||!q||!latent||!rope||!sel_idx||!sel_cnt||S<1||H<1||Q<1||R<1||V<1||K<1||K>512||
       T<S||T>65536||maxsel<1||maxsel>T||w->I!=K||w->O!=H*(Q+V))return 0;
    // Head slice [H0, H0+HC) of the full H heads (tensor-parallel attention). Full
    // range is H0=0, HC=H. A partial slice writes only its ctx columns, so zero the
    // pooled context buffer first (stale from a prior call) — needed for the copy-back
    // and for the fused GPU o_proj, which contracts over all H*V ctx columns.
    if(H0<0||HC<1||H0+HC>H)return 0;
    // Tensor-core WMMA path (opt-in). Only the non-fused case (no o_proj); ~3x the scalar core.
    { static int tc=-1; if(tc<0){const char*e=getenv("COLI_TC_ATTN");tc=(e&&atoi(e))?1:0;}
      if(tc && !proj) return tc_sparse_attn_run(w,out,q,latent,rope,sel_idx,sel_cnt,maxsel,H0,HC,S,H,Q,R,V,K,T,scale); }
    if(proj&&(proj->device!=w->device||proj->I!=H*V))return 0;
    DeviceContext *dc=find_ctx(w->device);if(!select_ctx(dc))return 0;
    size_t qb=(size_t)S*H*(Q+R)*sizeof(float),lb=(size_t)T*K*sizeof(float);
    size_t rb=(size_t)T*R*sizeof(float),cb=(size_t)S*H*V*sizeof(float);
    size_t sib=(size_t)S*maxsel*sizeof(int),scb=(size_t)S*sizeof(int);
    if(!reserve(&dc->aq,&dc->aq_cap,qb)||!reserve(&dc->al,&dc->al_cap,lb)||
       !reserve(&dc->ar,&dc->ar_cap,rb)||!reserve(&dc->ac,&dc->ac_cap,cb)||
       !reserve_bytes(&dc->asel,&dc->asel_cap,sib)||!reserve_bytes(&dc->acnt,&dc->acnt_cap,scb))return 0;
    if(!cuda_ok(cudaMemcpyAsync(dc->aq,q,qb,cudaMemcpyHostToDevice,dc->stream),"sparse attn q upload")||
       !cuda_ok(cudaMemcpyAsync(dc->al,latent,lb,cudaMemcpyHostToDevice,dc->stream),"sparse attn latent upload")||
       !cuda_ok(cudaMemcpyAsync(dc->ar,rope,rb,cudaMemcpyHostToDevice,dc->stream),"sparse attn rope upload")||
       !cuda_ok(cudaMemcpyAsync(dc->asel,sel_idx,sib,cudaMemcpyHostToDevice,dc->stream),"sparse attn sel upload")||
       !cuda_ok(cudaMemcpyAsync(dc->acnt,sel_cnt,scb,cudaMemcpyHostToDevice,dc->stream),"sparse attn cnt upload"))return 0;
    if((H0!=0||HC!=H)&&!cuda_ok(cudaMemsetAsync(dc->ac,0,cb,dc->stream),"sparse attn ctx zero"))return 0;
    size_t shared=(size_t)(2*K+maxsel+ATTN_TPB)*sizeof(float);
    attention_absorb_sparse_kernel<<<dim3(HC,S),ATTN_TPB,shared,dc->stream>>>(dc->ac,dc->aq,dc->al,
        dc->ar,w->weights,w->scales,(const int*)dc->asel,(const int*)dc->acnt,maxsel,w->fmt,H0,S,H,Q,R,V,K,T,scale);
    if(!cuda_ok(cudaGetLastError(),"sparse attn launch"))return 0;
    const float *src=dc->ac;size_t ob=cb;
    if(proj){
        ob=(size_t)S*proj->O*sizeof(float);if(!reserve(&dc->y,&dc->y_cap,ob))return 0;
        quant_matmul<<<dim3(proj->O,S),256,0,dc->stream>>>(dc->y,dc->ac,proj->weights,
            proj->scales,proj->fmt,S,proj->I,proj->O,row_bytes(proj->fmt,proj->I),proj->wrapped);
        if(!cuda_ok(cudaGetLastError(),"sparse attn o_proj launch"))return 0;src=dc->y;
    }
    if(!cuda_ok(cudaMemcpyAsync(out,src,ob,cudaMemcpyDeviceToHost,dc->stream),
                               proj?"sparse attn projected output download":"sparse attn context download")||
       !cuda_ok(cudaStreamSynchronize(dc->stream),"sparse attn synchronize"))return 0;
    return 1;
}

extern "C" int coli_cuda_attention_absorb_sparse(ColiCudaTensor *w,float *ctx,const float *q,
        const float *latent,const float *rope,const int *sel_idx,const int *sel_cnt,int maxsel,
        int H0,int HC,int S,int H,int Q,int R,int V,int K,int T,float scale){
    return attention_absorb_sparse_run(w,nullptr,ctx,q,latent,rope,sel_idx,sel_cnt,maxsel,H0,HC,S,H,Q,R,V,K,T,scale);
}

extern "C" int coli_cuda_attention_project_batch(ColiCudaTensor *w,ColiCudaTensor *proj,
        float *out,const float *q,const float *latent,const float *rope,int S,int H,int Q,
        int R,int V,int K,int T,float scale){
    return attention_absorb_batch_run(w,proj,out,q,latent,rope,S,H,Q,R,V,K,T,scale);
}

extern "C" void coli_cuda_tensor_free(ColiCudaTensor *tensor) {
    if (!tensor) return;
    // Wrapped tensors borrow host memory (owned by the Rust QTensor) — free only
    // the descriptor, never the buffers.
    if (tensor->wrapped) { std::free(tensor); return; }
    DeviceContext *ctx = find_ctx(tensor->device);
    if (ctx) select_ctx(ctx);
    if (tensor->tracked && ctx) {
        size_t bytes = tensor->weight_bytes + (tensor->fmt ? (size_t)tensor->O * sizeof(float) : 0);
        if (ctx->tensor_count) ctx->tensor_count--;
        if (ctx->tensor_bytes >= bytes) ctx->tensor_bytes -= bytes;
    }
    if (tensor->weights) cudaFree(tensor->weights);
    if (tensor->scales) cudaFree(tensor->scales);
    std::free(tensor);
}

// Zero-copy tensor: wrap host (RAM) buffers so the GPU reads them in place. Only
// valid where the device can access pageable host memory directly
// (cudaDevAttrPageableMemoryAccess — true on the GB10's coherent unified memory).
// `weights` stays in its on-disk layout: int4 is offset-binary (kernels pass off=1),
// int8 is already signed. No cudaMalloc, no memcpy, no conversion, no device memory.
extern "C" int coli_cuda_tensor_wrap(ColiCudaTensor **tensor,
                                     const void *weights, const float *scales,
                                     int fmt, int I, int O, int device) {
    if (!tensor || !weights || I < 1 || O < 1) return 0;
    size_t rb = row_bytes(fmt, I);
    if (!rb || (fmt && !scales)) return 0;
    if (*tensor) {
        ColiCudaTensor *t = *tensor;
        return t->fmt == fmt && t->I == I && t->O == O && t->device == device;
    }
    ColiCudaTensor *t = static_cast<ColiCudaTensor *>(std::calloc(1, sizeof(*t)));
    if (!t) return 0;
    t->fmt = fmt; t->I = I; t->O = O; t->device = device;
    t->weight_bytes = rb * (size_t)O;
    t->weights = const_cast<void *>(weights);
    t->scales = const_cast<float *>(scales);
    t->wrapped = 1;
    *tensor = t;
    return 1;
}

extern "C" size_t coli_cuda_tensor_bytes(const ColiCudaTensor *tensor) {
    return tensor ? tensor->weight_bytes + (tensor->fmt ? (size_t)tensor->O * sizeof(float) : 0) : 0;
}

extern "C" int coli_cuda_tensor_device(const ColiCudaTensor *tensor) {
    return tensor ? tensor->device : -1;
}

/* ==== resident-pipeline primitives (Inc.0, 2026-07-13) ====
 * Device-side building blocks so the residual stream can stay on the layer's
 * home device across a whole layer. Control flow stays on CPU; only the data
 * plane lives here. All entry points take DEVICE pointers (no transfers) —
 * the caller owns staging via the pipe buffer API below. */

__global__ static void pipe_rmsnorm_rows(float *y,const float *x,const float *w,
                                         int D,float eps,int xstride,int ystride){
    const float *xr=x+(size_t)blockIdx.x*xstride; float *yr=y+(size_t)blockIdx.x*ystride;
    __shared__ double sh[256];
    double a=0; for(int i=threadIdx.x;i<D;i+=blockDim.x){ double v=xr[i]; a+=v*v; }
    sh[threadIdx.x]=a; __syncthreads();
    for(int s=blockDim.x/2;s>0;s>>=1){ if(threadIdx.x<s) sh[threadIdx.x]+=sh[threadIdx.x+s]; __syncthreads(); }
    float r=rsqrtf((float)(sh[0]/D)+eps);
    for(int i=threadIdx.x;i<D;i+=blockDim.x) yr[i]=xr[i]*r*w[i];
}

/* RoPE interleaved, identical math to glm.c rope_interleave. One block per row;
 * row layout: v + row*stride + offset holds R floats. pos index = row/heads
 * (heads=1 for k_rot rows, heads=H for [S,H,qh] query rows). */
__global__ static void pipe_rope_rows(float *v,const int *pos,int pos_base,int stride,
                                      int offset,int R,int heads,float theta){
    float *p=v+(size_t)blockIdx.x*stride+offset;
    int half=R/2, ps=pos?pos[blockIdx.x/heads]:pos_base+(int)(blockIdx.x/heads);
    __shared__ float in[256];
    for(int j=threadIdx.x;j<R;j+=blockDim.x) in[j]=p[j];
    __syncthreads();
    for(int j=threadIdx.x;j<half;j+=blockDim.x){
        float inv=__powf(theta,-2.0f*j/R);
        float ang=ps*inv, cs=__cosf(ang), sn=__sinf(ang);
        float a=in[2*j], b=in[2*j+1];
        p[j]=a*cs-b*sn; p[half+j]=b*cs+a*sn;
    }
}

__global__ static void pipe_add_n(float *x,const float *t,size_t n){
    size_t i=(size_t)blockIdx.x*blockDim.x+threadIdx.x;
    if(i<n) x[i]+=t[i];
}

/* Fixed-order partial merge: block b adds partial row b into x row rows[b].
 * Target rows are unique by construction (CPU pre-sums per token), so no
 * atomics — the 9.20.7 lesson. */
__global__ static void pipe_rows_add(float *x,const float *partial,const int *rows,
                                     int D){
    float *xr=x+(size_t)rows[blockIdx.x]*D;
    const float *pr=partial+(size_t)blockIdx.x*D;
    for(int i=threadIdx.x;i<D;i+=blockDim.x) xr[i]+=pr[i];
}

/* scratch persistente per (device,slot): cresce e resta — niente cudaMalloc/Free
 * per layer (78 x ~10 alloc/richiesta erano puro churn). */
extern "C" float *coli_cuda_pipe_scratch(int device,int slot,size_t bytes){
    DeviceContext *ctx=find_ctx(device);
    if(slot<0||slot>=24||!select_ctx(ctx)) return NULL;
    if(!reserve(&ctx->pipe_buf[slot],&ctx->pipe_cap[slot],bytes)) return NULL;
    return ctx->pipe_buf[slot];
}
extern "C" void *coli_cuda_pipe_alloc(int device,size_t bytes){
    DeviceContext *ctx=find_ctx(device); if(!select_ctx(ctx)) return NULL;
    void *p=NULL;
    if(!cuda_ok(cudaMalloc(&p,bytes),"pipe alloc")) return NULL;
    return p;
}
extern "C" void coli_cuda_pipe_free(int device,void *p){
    DeviceContext *ctx=find_ctx(device); if(!p||!select_ctx(ctx)) return;
    cudaFree(p);
}
extern "C" int coli_cuda_pipe_upload(int device,void *dst,const void *src,size_t bytes){
    DeviceContext *ctx=find_ctx(device); if(!select_ctx(ctx)) return 0;
    return cuda_ok(cudaMemcpy(dst,src,bytes,cudaMemcpyHostToDevice),"pipe upload");
}
extern "C" int coli_cuda_pipe_download(int device,const void *src,void *dst,size_t bytes){
    DeviceContext *ctx=find_ctx(device); if(!select_ctx(ctx)) return 0;
    return cuda_ok(cudaMemcpy(dst,src,bytes,cudaMemcpyDeviceToHost),"pipe download");
}
extern "C" int coli_cuda_pipe_rmsnorm(int device,float *y_dev,const float *x_dev,
                                      const float *w_dev,int S,int D,float eps){
    DeviceContext *ctx=find_ctx(device);
    if(S<1||D<1||!select_ctx(ctx)) return 0;
    pipe_rmsnorm_rows<<<S,256>>>(y_dev,x_dev,w_dev,D,eps,D,D);
    return cuda_ok(cudaGetLastError(),"pipe rmsnorm");
}
extern "C" int coli_cuda_pipe_rmsnorm_s(int device,float *y_dev,const float *x_dev,
                                        const float *w_dev,int S,int D,float eps,
                                        int xstride,int ystride){
    DeviceContext *ctx=find_ctx(device);
    if(S<1||D<1||xstride<D||ystride<D||!select_ctx(ctx)) return 0;
    pipe_rmsnorm_rows<<<S,256>>>(y_dev,x_dev,w_dev,D,eps,xstride,ystride);
    return cuda_ok(cudaGetLastError(),"pipe rmsnorm strided");
}
extern "C" int coli_cuda_pipe_rope(int device,float *v_dev,const int *pos_dev,
                                   int rows,int stride,int offset,int R,int heads,
                                   float theta){
    DeviceContext *ctx=find_ctx(device);
    if(rows<1||R<2||R>256||heads<1||!select_ctx(ctx)) return 0;
    pipe_rope_rows<<<rows,128>>>(v_dev,pos_dev,0,stride,offset,R,heads,theta);
    return cuda_ok(cudaGetLastError(),"pipe rope");
}
extern "C" int coli_cuda_pipe_rope_base(int device,float *v_dev,int pos_base,int rows,
                                        int stride,int offset,int R,int heads,float theta){
    DeviceContext *ctx=find_ctx(device);
    if(rows<1||R<2||R>256||heads<1||!select_ctx(ctx)) return 0;
    pipe_rope_rows<<<rows,128>>>(v_dev,NULL,pos_base,stride,offset,R,heads,theta);
    return cuda_ok(cudaGetLastError(),"pipe rope base");
}
extern "C" int coli_cuda_pipe_copy2d(int device,float *dst,int dpitch,const float *src,
                                     int spitch,int width,int height){
    DeviceContext *ctx=find_ctx(device); if(!select_ctx(ctx)) return 0;
    return cuda_ok(cudaMemcpy2D(dst,(size_t)dpitch*4,src,(size_t)spitch*4,
        (size_t)width*4,height,cudaMemcpyDeviceToDevice),"pipe copy2d");
}
/* attention batch + fused o_proj with DEVICE-resident q/latent/rope: the whole
 * upstream projection chain stayed on this device, so nothing is uploaded here.
 * Only the final [S,O] projection is downloaded to host. */
extern "C" int coli_cuda_attention_project_batch_dev(ColiCudaTensor *w,ColiCudaTensor *proj,
        float *out,const float *q_dev,const float *latent_dev,const float *rope_dev,
        int S,int H,int Q,int R,int V,int K,int T,float scale){
    if(!w||!proj||!out||!q_dev||!latent_dev||!rope_dev||S<1||H<1||Q<1||R<1||V<1||
       K<1||K>512||T<S||T>8192||w->I!=K||w->O!=H*(Q+V)||
       proj->device!=w->device||proj->I!=H*V)return 0;
    DeviceContext *dc=find_ctx(w->device);if(!select_ctx(dc))return 0;
    size_t cb=(size_t)S*H*V*sizeof(float);
    if(!reserve(&dc->ac,&dc->ac_cap,cb))return 0;
    size_t shared=(size_t)(2*K+T+ATTN_TPB)*sizeof(float);
    attention_absorb_batch_kernel<<<dim3(H,S),ATTN_TPB,shared,dc->stream>>>(dc->ac,q_dev,latent_dev,
        rope_dev,w->weights,w->scales,w->fmt,S,H,Q,R,V,K,T,scale);
    if(!cuda_ok(cudaGetLastError(),"pipe attention launch"))return 0;
    size_t ob=(size_t)S*proj->O*sizeof(float);
    if(!reserve(&dc->y,&dc->y_cap,ob))return 0;
    quant_matmul<<<dim3(proj->O,S),256,0,dc->stream>>>(dc->y,dc->ac,proj->weights,
        proj->scales,proj->fmt,S,proj->I,proj->O,row_bytes(proj->fmt,proj->I),proj->wrapped);
    if(!cuda_ok(cudaGetLastError(),"pipe o_proj launch"))return 0;
    if(!cuda_ok(cudaMemcpyAsync(out,dc->y,ob,cudaMemcpyDeviceToHost,dc->stream),"pipe attention download")||
       !cuda_ok(cudaStreamSynchronize(dc->stream),"pipe attention sync"))return 0;
    return 1;
}
extern "C" int coli_cuda_pipe_silu_mul(int device,float *gate_dev,const float *up_dev,
                                       size_t n){
    DeviceContext *ctx=find_ctx(device); if(!n||!select_ctx(ctx)) return 0;
    silu_mul<<<(unsigned)((n+255)/256),256>>>(gate_dev,up_dev,n);
    return cuda_ok(cudaGetLastError(),"pipe silu mul");
}
extern "C" int coli_cuda_pipe_add(int device,float *x_dev,const float *t_dev,size_t n){
    DeviceContext *ctx=find_ctx(device); if(!n||!select_ctx(ctx)) return 0;
    pipe_add_n<<<(unsigned)((n+255)/256),256>>>(x_dev,t_dev,n);
    return cuda_ok(cudaGetLastError(),"pipe add");
}
extern "C" int coli_cuda_pipe_rows_add(int device,float *x_dev,const float *partial_dev,
                                       const int *rows_dev,int nrows,int D){
    DeviceContext *ctx=find_ctx(device); if(nrows<1||D<1||!select_ctx(ctx)) return 0;
    pipe_rows_add<<<nrows,256>>>(x_dev,partial_dev,rows_dev,D);
    return cuda_ok(cudaGetLastError(),"pipe rows add");
}
/* GEMM with device-resident activations: same quant_matmul kernel as
 * coli_cuda_matmul, zero host transfers. */
extern "C" int coli_cuda_pipe_gemm(ColiCudaTensor *t,float *y_dev,const float *x_dev,
                                   int S){
    if(!t||S<1) return 0;
    DeviceContext *ctx=find_ctx(t->device); if(!select_ctx(ctx)) return 0;
    // Tile only when S is large enough to amortize the 16-row tile (decode S=1 stays
    // on the naive kernel, which is better for a single row).
    const char *tile_env=getenv("COLI_TILE_I8");
    int tile=(!tile_env||strcmp(tile_env,"0")!=0)&&ctx->compute_major>=7&&S>=16;
    if(tile&&(t->fmt==1||t->fmt==4)){
        dim3 tg((unsigned)((t->O+63)/64),(unsigned)((S+15)/16));
        if(t->fmt==4)
            fp8a16_matmul<<<tg,128>>>(y_dev,x_dev,(const uint8_t*)t->weights,t->scales,S,t->I,t->O);
        else
            i8a16_matmul<<<tg,128>>>(y_dev,x_dev,(const uint8_t*)t->weights,t->scales,S,t->I,t->O);
    }else{
        dim3 grid((unsigned)t->O,(unsigned)S);
        quant_matmul<<<grid,256>>>(y_dev,x_dev,t->weights,t->scales,t->fmt,S,t->I,t->O,
            row_bytes(t->fmt,t->I),t->wrapped);
    }
    return cuda_ok(cudaGetLastError(),"pipe gemm");
}
/* copia diretta scheda->scheda (P2P se disponibile, altrimenti staging driver) */
extern "C" int coli_cuda_pipe_peer_copy(int dst_dev,float *dst,int src_dev,
                                        const float *src,size_t bytes){
    if(!dst||!src) return 0;
    if(dst_dev==src_dev){ DeviceContext *c=find_ctx(dst_dev); if(!select_ctx(c)) return 0;
        return cuda_ok(cudaMemcpy(dst,src,bytes,cudaMemcpyDeviceToDevice),"pipe intra copy"); }
    return cuda_ok(cudaMemcpyPeer(dst,dst_dev,src,src_dev,bytes),"pipe peer copy");
}
/* come attention_project_batch_dev ma l'uscita di o_proj RESTA sul device (out_dev). */
extern "C" int coli_cuda_attention_project_batch_dev_out(ColiCudaTensor *w,ColiCudaTensor *proj,
        float *out_dev,const float *q_dev,const float *latent_dev,const float *rope_dev,
        int S,int H,int Q,int R,int V,int K,int T,float scale){
    if(!w||!proj||!out_dev||!q_dev||!latent_dev||!rope_dev||S<1||H<1||Q<1||R<1||V<1||
       K<1||K>512||T<S||T>8192||w->I!=K||w->O!=H*(Q+V)||
       proj->device!=w->device||proj->I!=H*V)return 0;
    DeviceContext *dc=find_ctx(w->device);if(!select_ctx(dc))return 0;
    size_t cb=(size_t)S*H*V*sizeof(float);
    if(!reserve(&dc->ac,&dc->ac_cap,cb))return 0;
    size_t shared=(size_t)(2*K+T+ATTN_TPB)*sizeof(float);
    attention_absorb_batch_kernel<<<dim3(H,S),ATTN_TPB,shared,dc->stream>>>(dc->ac,q_dev,latent_dev,
        rope_dev,w->weights,w->scales,w->fmt,S,H,Q,R,V,K,T,scale);
    if(!cuda_ok(cudaGetLastError(),"pipe attention launch (dev out)"))return 0;
    quant_matmul<<<dim3(proj->O,S),256,0,dc->stream>>>(out_dev,dc->ac,proj->weights,
        proj->scales,proj->fmt,S,proj->I,proj->O,row_bytes(proj->fmt,proj->I),proj->wrapped);
    if(!cuda_ok(cudaGetLastError(),"pipe o_proj launch (dev out)"))return 0;
    return cuda_ok(cudaStreamSynchronize(dc->stream),"pipe attention sync (dev out)");
}
/* absorb batch con TUTTO su device (q/latent/rope gia' residenti sulla scheda
 * dello shard, ctx resta sul device): il cuore della attention head-shardata
 * dentro il pipeline. Nessun trasferimento host. */
extern "C" int coli_cuda_attention_absorb_batch_dev(ColiCudaTensor *w,float *ctx_dev,
        const float *q_dev,const float *latent_dev,const float *rope_dev,
        int S,int H,int Q,int R,int V,int K,int T,float scale){
    if(!w||!ctx_dev||!q_dev||!latent_dev||!rope_dev||S<1||H<1||Q<1||R<1||V<1||
       K<1||K>512||T<S||T>8192||w->I!=K||w->O!=H*(Q+V))return 0;
    DeviceContext *dc=find_ctx(w->device);if(!select_ctx(dc))return 0;
    size_t shared=(size_t)(2*K+T+ATTN_TPB)*sizeof(float);
    attention_absorb_batch_kernel<<<dim3(H,S),ATTN_TPB,shared,dc->stream>>>(ctx_dev,q_dev,latent_dev,
        rope_dev,w->weights,w->scales,w->fmt,S,H,Q,R,V,K,T,scale);
    if(!cuda_ok(cudaGetLastError(),"pipe shard attention launch"))return 0;
    return cuda_ok(cudaStreamSynchronize(dc->stream),"pipe shard attention sync");
}
/* absorb per il DECODE con KV gia' residente: carica solo q (poche KB),
 * latent/rope arrivano dall'ombra device. ctx torna a host (S piccolo). */
extern "C" int coli_cuda_attention_absorb_kvdev(ColiCudaTensor *w,float *ctx,const float *q,
        const float *latent_dev,const float *rope_dev,int H,int Q,int R,int V,int K,int T,
        float scale){
    if(!w||!ctx||!q||!latent_dev||!rope_dev||H<1||Q<1||R<1||V<1||K<1||K>512||T<1||T>8192||
       w->I!=K||w->O!=H*(Q+V))return 0;
    DeviceContext *dc=find_ctx(w->device);if(!select_ctx(dc))return 0;
    size_t qb=(size_t)H*(Q+R)*sizeof(float),cb=(size_t)H*V*sizeof(float);
    if(!reserve(&dc->aq,&dc->aq_cap,qb)||!reserve(&dc->ac,&dc->ac_cap,cb))return 0;
    if(!cuda_ok(cudaMemcpyAsync(dc->aq,q,qb,cudaMemcpyHostToDevice,dc->stream),"kvdev q upload"))return 0;
    /* Flash decode: T-parallel absorb (qabs -> per-tile partials -> combine+W_V). */
    int nTiles=(T+FLASH_TILE-1)/FLASH_TILE;
    float *qabs=coli_cuda_pipe_scratch(w->device,22,(size_t)H*K*sizeof(float));
    float *partials=coli_cuda_pipe_scratch(w->device,23,(size_t)H*nTiles*(K+2)*sizeof(float));
    if(!qabs||!partials)return 0;
    flash_qabs<<<H,ATTN_TPB,0,dc->stream>>>(qabs,dc->aq,w->weights,w->scales,w->fmt,H,Q,R,V,K);
    size_t sh1=(size_t)(FLASH_TILE+K+ATTN_TPB)*sizeof(float);
    flash_partial<<<dim3(H,nTiles),ATTN_TPB,sh1,dc->stream>>>(partials,qabs,dc->aq,latent_dev,
        rope_dev,H,Q,R,K,T,nTiles,scale);
    size_t sh2=(size_t)K*sizeof(float);
    flash_combine<<<H,ATTN_TPB,sh2,dc->stream>>>(dc->ac,partials,w->weights,w->scales,w->fmt,H,Q,V,K,nTiles);
    if(!cuda_ok(cudaGetLastError(),"kvdev flash launch")||
       !cuda_ok(cudaMemcpyAsync(ctx,dc->ac,cb,cudaMemcpyDeviceToHost,dc->stream),"kvdev ctx download")||
       !cuda_ok(cudaStreamSynchronize(dc->stream),"kvdev absorb sync"))return 0;
    return 1;
}
extern "C" int coli_cuda_pipe_sync(int device){
    DeviceContext *ctx=find_ctx(device); if(!select_ctx(ctx)) return 0;
    return cuda_ok(cudaDeviceSynchronize(),"pipe sync");
}
