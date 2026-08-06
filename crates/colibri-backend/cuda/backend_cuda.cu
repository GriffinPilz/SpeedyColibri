#include "backend_cuda.h"

#include <cuda_runtime.h>
#include <mma.h>
#include <cuda_fp8.h>

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>
#include <mutex>
#include <chrono>
#include <thread>
#include <atomic>
#include <vector>

struct ColiCudaTensor {
    void *weights;
    float *scales;
    size_t weight_bytes;
    int fmt, I, O, device;
    int tracked;
    // Zero-copy on unified memory (GB10): `weights`/`scales` point directly at the
    // host (RAM) buffers — no cudaMalloc, no memcpy. The weights stay in their
    // on-disk layout.
    int wrapped;
    // NVFP4 (fmt==5) only: `weights` holds packed e2m1 nibbles [O, ceil(I/2)], `bscale`
    // holds ue4m3 per-16 block scales [O, ceil(I/16)], `gscale` is the per-tensor global.
    // `scales` is unused for NVFP4. Zero for every other format.
    const void *bscale;
    float gscale;
};

typedef struct {
    int device;
    int compute_major,compute_minor;
    float *x, *y, *gate, *up;
    size_t x_cap, y_cap, gate_cap, up_cap;
    uint8_t *qx; float *qscale;
    size_t qx_cap, qscale_cap;
    float *host_x,*host_y; size_t host_x_cap,host_y_cap;
    /* Pinned staging for the segmented-GEMV descriptor array. It is only ~1 KB, but it
     * is uploaded once per MoE layer per token, and a PAGEABLE async copy is not async:
     * the driver bounces it through an internal buffer and synchronizes. */
    float *host_seg; size_t host_seg_cap;
    /* Segmented-expert descriptors (see coli_cuda_expert_seg_nvfp4_relu2): per-row-tile
     * expert index + local row base, and per-expert offsets/rows/weight pointers/scales. */
    int *sg_tile_e,*sg_tile_r0,*sg_off,*sg_rows; float *sg_ug,*sg_dg; void *sg_uw,*sg_ubs,*sg_dw,*sg_dbs;
    size_t sg_tile_e_cap,sg_tile_r0_cap,sg_off_cap,sg_rows_cap,sg_ug_cap,sg_dg_cap,
           sg_uw_cap,sg_ubs_cap,sg_dw_cap,sg_dbs_cap;
    /* Device scratch for expert weights, so the kernel reads clean device memory
     * instead of zero-copy from freshly-pread (dirty, coherence-heavy) host pages. */
    uint8_t *ewg,*ewu,*ewd; size_t ewg_cap,ewu_cap,ewd_cap;
    float *esg,*esu,*esd; size_t esg_cap,esu_cap,esd_cap;
    uint8_t *ebsg,*ebsu,*ebsd; size_t ebsg_cap,ebsu_cap,ebsd_cap;  /* NVFP4 block-scale device scratch (devcopy) */
    /* Per-layer bulk residency (COLI_LAYER_RESIDENT): one device arena holding a whole
     * group's expert weights, filled by a single transfer out of `host_lres`, which is
     * PINNED. The per-expert `ewg/ewu/...` scratch above stages one expert at a time out of
     * pageable memory, which is what measured 1.0-2.2 GB/s and 93.6% of expert-call time. */
    uint8_t *lres; size_t lres_cap;
    float *host_lres; size_t host_lres_cap;   /* `float*` to match reserve_pinned; bytes underneath */
    float *aq,*al,*ar,*ac; size_t aq_cap,al_cap,ar_cap,ac_cap;
    /* Nemotron-H Mamba2 selective-scan decode scratch (state in/out + per-step inputs). */
    float *ms_state,*ms_x,*ms_y,*ms_b,*ms_c,*ms_dth,*ms_dah,*ms_d;
    size_t ms_state_cap,ms_x_cap,ms_y_cap,ms_b_cap,ms_c_cap,ms_dth_cap,ms_dah_cap,ms_d_cap;
    /* Pinned host staging for the prefill scan's bulk transfers (hidden in, y+state out).
     * Pageable async copies fall back to a synchronous bounce buffer (~146 MB/s measured);
     * pinned lets the DMA engine run and a plain host memcpy covers the rest. */
    float *ms_pin_x,*ms_pin_y,*ms_pin_state;
    size_t ms_pin_x_cap,ms_pin_y_cap,ms_pin_state_cap;
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
/* One mutex per DeviceContext slot. Every compute entry point holds it across the whole
 * reserve -> upload -> launch -> download -> synchronize sequence, so two threads issuing
 * work on the same device can't clobber that context's shared scratch (aq/al/ar/ac,
 * x/y/gate/up, qx/qscale, ewg..., asel/acnt, aqa/akb/amsk). Production forward() is
 * single-threaded per device, so this is uncontended there; it exists to make the
 * multi-threaded `cargo test` harness deterministic. Kept out of DeviceContext itself
 * because that struct is reset with `*ctx = {}` (line ~936), which a std::mutex member
 * would forbid. Indexed by slot (ctx - g_ctx), always in [0, COLI_CUDA_MAX_DEVICES). */
static std::mutex g_scratch_mu[COLI_CUDA_MAX_DEVICES];
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

/* Mutex guarding `ctx`'s shared scratch. `ctx` must point into g_ctx (find_ctx only ever
 * returns that or nullptr, and callers bail on nullptr before locking). */
static inline std::mutex &scratch_mu(DeviceContext *ctx) { return g_scratch_mu[ctx - g_ctx]; }

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
    if (fmt == 3) return (size_t)(I + 3) / 4;
    if (fmt == 4) return (size_t)I;          // e4m3 fp8: 1 byte/weight
    if (fmt == 5) return (size_t)(I + 1) / 2; // nvfp4: packed e2m1 nibbles, 2/byte
    // mxfp4: the SAME packed e2m1 nibbles as nvfp4 — the formats differ only in the
    // separate block-scale array (E8M0 per 32 vs ue4m3 per 16), which is not counted here.
    if (fmt == 6) return (size_t)(I + 1) / 2;
    return 0;
}

// Decode one e4m3 (fp8) byte to float via the hardware conversion.
__device__ __forceinline__ static float e4m3f(uint8_t b) {
    __half_raw hr = __nv_cvt_fp8_to_halfraw((__nv_fp8_storage_t)b, __NV_E4M3);
    return __half2float(*reinterpret_cast<__half *>(&hr));
}

// `off` is vestigial — it selected int4's offset-binary vs signed representation,
// which has been removed. Kept in the signature so existing call sites need no change.
__device__ static float weight_at(const void *weights, int fmt, size_t row, int i, int off=0) {
    (void)off;
    const uint8_t *base = static_cast<const uint8_t *>(weights) + row;
    if (fmt == 0) return reinterpret_cast<const float *>(base)[i];
    if (fmt == 1) return static_cast<float>(reinterpret_cast<const int8_t *>(base)[i]);
    if (fmt == 4) return e4m3f(base[i]);      // e4m3 fp8; per-row scale applied by caller
    const uint8_t *q = base;
    // int2 (fmt 3): 4 values/byte, value = field − 2
    uint8_t v = q[i >> 2];
    return static_cast<float>(((v >> ((i & 3) * 2)) & 3) - 2);
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

/* Clamped OpenAI-SwiGLU (MiniMax-M3 "swigluoai"): gate clamped to <= limit, up
 * clamped to [-limit, limit], out = (up + 1) * gate * sigmoid(alpha * gate).
 * Mirrors the CPU `swiglu_oai` reference so the GPU expert path is token-identical. */
__global__ static void swiglu_oai_mul(float *gate, const float *up, size_t n,
                                      float alpha, float limit) {
    size_t i = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) {
        float g = gate[i];
        if (g > limit) g = limit;                    // clamp upper only
        float u = fminf(fmaxf(up[i], -limit), limit); // clamp [-limit, limit]
        float gated = g / (1.0f + expf(-alpha * g));  // g * sigmoid(alpha * g)
        gate[i] = gated * (u + 1.0f);
    }
}

/* FFN gate/up activation-combine variant, set once from the host by
 * coli_cuda_set_activation: 0 = SiLU-SwiGLU (GLM, the default), 1 = clamped
 * OpenAI-SwiGLU (MiniMax-M3). Applies to every FFN kernel (routed experts, shared
 * expert, dense MLP) since the activation is a per-model constant. */
static int   g_act_oai   = 0;
static float g_act_alpha = 1.702f;
static float g_act_limit = 7.0f;

extern "C" void coli_cuda_set_activation(int oai, float alpha, float limit) {
    g_act_oai = oai;
    g_act_alpha = alpha;
    g_act_limit = limit;
}

/* Gateless ReLU² activation (Nemotron-H experts): t = relu(t)² in place over `n`
 * elements. The two-tensor expert has no gate to combine — it squares the ReLU of the
 * single up-projection between the up and down GEMMs. Mirrors the CPU reference
 * `r = u.max(0.0); u = r*r` (moe.rs `ffn_cpu`, relu2 branch) so the GPU path is
 * token-identical. */
__global__ static void relu2_inplace(float *t, size_t n) {
    size_t i = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) {
        float v = t[i];
        v = v > 0.f ? v : 0.f;
        t[i] = v * v;
    }
}

/* Launch the selected gate*up activation-combine over `n` elements on `stream`. */
/* DeepSeek-V4: plain SwiGLU under the reference's ASYMMETRIC clamp — `up` both sides,
 * `gate` only from above (silu already bounds it below).
 *
 * The clamps are byte-identical to `swiglu_oai_mul` above; the PRODUCT is not. That one
 * gates with `sigmoid(alpha*g)` and multiplies by `(u + 1)`, neither of which V4 does.
 * Matching clamps are exactly why reusing the oai kernel here would have looked correct
 * and been wrong everywhere the product saturates. Mirrors `dsv4::swiglu_clamped_one`. */
__global__ static void swiglu_clamped_mul(float *gate, const float *up, size_t n,
                                          float limit) {
    size_t i = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) {
        float g = fminf(gate[i], limit);
        float u = fminf(fmaxf(up[i], -limit), limit);
        gate[i] = (g / (1.0f + expf(-g))) * u;
    }
}

static inline void act_mul(float *gate, const float *up, size_t n, cudaStream_t stream) {
    unsigned blocks = (unsigned)((n + 255) / 256);
    if (g_act_oai)
        swiglu_oai_mul<<<blocks, 256, 0, stream>>>(gate, up, n, g_act_alpha, g_act_limit);
    else if (g_act_limit > 0.0f)
        swiglu_clamped_mul<<<blocks, 256, 0, stream>>>(gate, up, n, g_act_limit);
    else
        silu_mul<<<blocks, 256, 0, stream>>>(gate, up, n);
}

/* FP8 (e4m3) tiled tensor-core expert matmuls (1 byte/weight, direct K stride).
 * Weights are FP8, activations FP16, MMA runs in f16 (W8A16). This is the tiled
 * path that replaces the naive quant_matmul's M-fold weight re-reads. */
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

/* FP8 (e4m3) GEMV for the single-row decode case (M==1). The tiled `fp8a16_matmul`
 * is a 16x16x16 WMMA kernel: at M==1 it computes a 16-row MMA of which 15 rows are
 * padding, and only a subset of threads load each weight tile — measured ~51 GB/s
 * against the GPU's ~155 GB/s pageable-host read ceiling. This path instead assigns
 * one warp per output column and streams the weight row with all 32 lanes reading
 * consecutive bytes (a coalesced sweep), so the whole block is doing the
 * memory-bound read rather than a subset of threads.
 *
 *   y[n] = scale[n] * Σ_k x[k] · e4m3(w[n*K + k])          n ∈ [0,N)
 *
 * `x` (K floats) is loaded into shared once per block. Per-byte reads (not `uchar4`)
 * because expert weight offsets are not 4-byte aligned — 0 of 4326 sampled were even
 * 512-aligned — and a misaligned vector load is UB; the tiled kernels read per-byte
 * for the same reason. Consecutive lanes still read consecutive bytes, so the access
 * coalesces into cache-line transactions. */
__global__ static void fp8a16_gemv(float *y,const float *x,const uint8_t *w,
                                   const float *scale,int K,int N){
    extern __shared__ float xs[];
    for(int k=threadIdx.x;k<K;k+=blockDim.x) xs[k]=x[k];
    __syncthreads();
    int warp=threadIdx.x>>5,lane=threadIdx.x&31;
    int n=blockIdx.x*(blockDim.x>>5)+warp;
    if(n>=N) return;
    const uint8_t *wr=w+(size_t)n*K; float acc=0.f;
    for(int k=lane;k<K;k+=32) acc+=xs[k]*e4m3f(wr[k]);
    #pragma unroll
    for(int o=16;o>0;o>>=1) acc+=__shfl_down_sync(0xffffffff,acc,o);
    if(lane==0) y[n]=acc*scale[n];
}

/* ==== NVFP4 (e2m1 nibbles + ue4m3 per-16 block scale + f32 global) expert kernels ====
 * Weight decode: W[n,k] = e2m1(nib(n,k)) · e4m3(bscale[n*ceil(K/16) + k/16]) · gscale.
 * Nibbles are packed 2/byte (low = even k). At 0.5 B/wt + one ue4m3/16 (0.0625 B/wt) the
 * decode GEMV reads ~half the bytes of the fp8 GEMV — the bytes-bound decode win. Compute
 * mirrors the fp8a16 path (decode → f16 → WMMA), NOT native FP4 MMA (a later lever). */

// Decode one 4-bit e2m1 code (bit 3 = sign; low 3 bits pick a magnitude) to float.
__device__ __forceinline__ static float e2m1f(int nib) {
    const float mag[8] = {0.f, 0.5f, 1.f, 1.5f, 2.f, 3.f, 4.f, 6.f};
    float m = mag[nib & 7];
    return (nib & 8) ? -m : m;
}

/* Single-row decode GEMV (S==1): one warp per output column, all 32 lanes sweep the
 * nibble row coalesced. Mirror of `fp8a16_gemv` with the nvfp4 decode + global. */
__global__ static void nvfp4_gemv(float *y,const float *x,const uint8_t *w,
                                  const uint8_t *bs,float g,int K,int N){
    extern __shared__ float xs[];
    for(int k=threadIdx.x;k<K;k+=blockDim.x) xs[k]=x[k];
    __syncthreads();
    int warp=threadIdx.x>>5,lane=threadIdx.x&31;
    int n=blockIdx.x*(blockDim.x>>5)+warp;
    if(n>=N) return;
    int Kh=(K+1)>>1, nb=(K+15)>>4;
    const uint8_t *wr=w+(size_t)n*Kh;
    const uint8_t *br=bs+(size_t)n*nb;
    float acc=0.f;
    for(int k=lane;k<K;k+=32){
        uint8_t byte=wr[k>>1];
        int nib=(k&1)?(byte>>4):(byte&0xF);
        acc += xs[k]*e2m1f(nib)*e4m3f(br[k>>4]);
    }
    #pragma unroll
    for(int o=16;o>0;o>>=1) acc+=__shfl_down_sync(0xffffffff,acc,o);
    if(lane==0) y[n]=acc*g;
}

/* Widest-read NVFP4 GEMV: one uint4 (16 B) per lane, 512 B per warp transaction.
 *
 * `nvfp4_gemv` gives one k to each lane and indexes `wr[k>>1]`, so lanes 2j/2j+1 fetch the
 * SAME byte and a warp covers 16 B. `nvfp4_gemv_wide` gives each lane a byte: 32 B. Both
 * are far under what the memory wants. Measured on GB10 with a pure read kernel over a
 * real shard (mapbench2), bandwidth against access width:
 *
 *   4 B/lane (128 B/warp)   144.7 GB/s heap
 *   16 B/lane (512 B/warp)  172.9 GB/s heap, 248.3 GB/s device-resident
 *
 * and the routed-expert gemv was measured at only ~110 GB/s, i.e. the KERNEL was the
 * limit, not the memory. Hence 16 B/lane here.
 *
 * Layout: a uint4 spans 32 nibbles = 32 values of k, so it covers exactly TWO NVFP4
 * block-16 scales — `v.x`/`v.y` take `br[2i]`, `v.z`/`v.w` take `br[2i+1]` — and the
 * scale decode drops to 2 per 16 B instead of 1 per value.
 *
 * `x` IS staged in shared memory. A first version read it from global to free up
 * occupancy, which was a mistake: each lane then pulls 32 consecutive x values (128 B) so
 * a warp scatters over ~4 KB per step instead of broadcasting from shared, and the
 * activation stream becomes the bottleneck the weight stream just stopped being. Measured
 * on Nemotron decode: gpu-ffn flat but total moe 1271 -> 1503 ms, an 18% REGRESSION.
 * Widen the weights, keep x shared.
 *
 * NOT bit-identical to `nvfp4_gemv`: per-lane k assignment changes the f32 reduction
 * order, so results differ in the last ULPs. The caller must keep using the narrow kernel
 * whenever `exact` is set (MTP verify), which is why this is a separate kernel and not a
 * replacement. Requires `wr` 16 B aligned, i.e. `Kh % 16 == 0` and a 16 B aligned base;
 * the caller checks both and falls back otherwise. */
__global__ static void nvfp4_gemv_u4(float *y,const float *x,const uint8_t *w,
                                     const uint8_t *bs,float g,int K,int N){
    extern __shared__ float xs[];
    for(int k=threadIdx.x;k<K;k+=blockDim.x) xs[k]=x[k];
    __syncthreads();
    int warp=threadIdx.x>>5,lane=threadIdx.x&31;
    int n=blockIdx.x*(blockDim.x>>5)+warp;
    if(n>=N) return;
    int Kh=(K+1)>>1, nb=(K+15)>>4;
    const uint8_t *wr=w+(size_t)n*Kh;
    const uint8_t *br=bs+(size_t)n*nb;
    const uint4 *wq=(const uint4*)wr;
    int Kq=Kh>>4;                       /* whole uint4 groups; caller guarantees Kh%16==0 */
    float acc=0.f;
    for(int i=lane;i<Kq;i+=32){
        uint4 v=wq[i];
        int k0=i<<5;                    /* 32 values of k per uint4 */
        float s0=e4m3f(br[k0>>4]);      /* k0 .. k0+15  */
        float s1=e4m3f(br[(k0+16)>>4]); /* k0+16 .. k0+31 */
        uint32_t word[4]={v.x,v.y,v.z,v.w};
        #pragma unroll
        for(int wi=0;wi<4;wi++){
            float sc=(wi<2)?s0:s1;
            int kb=k0+(wi<<3);
            uint32_t wd=word[wi];
            float part=0.f;
            #pragma unroll
            for(int bj=0;bj<4;bj++){
                uint32_t byte=(wd>>(bj<<3))&0xFFu;
                int kk=kb+(bj<<1);
                part += xs[kk]*e2m1f(byte&0xF) + xs[kk+1]*e2m1f(byte>>4);
            }
            acc+=part*sc;
        }
    }
    #pragma unroll
    for(int o=16;o>0;o>>=1) acc+=__shfl_down_sync(0xffffffff,acc,o);
    if(lane==0) y[n]=acc*g;
}

/* One expert's two projections, as the segmented GEMV kernels see them. Both are packed
 * into a single descriptor so the whole layer's set is ONE pinned upload per call rather
 * than six pageable ones. `which` selects up (0) or down (1) at launch. */
struct NvSegDesc {
    const uint8_t *uw,*ubs,*dw,*dbs;
    float ug,dg;
};

/* The same expert set, but delivered in KERNEL PARAMETER space instead of through a
 * device buffer.
 *
 * This distinction is worth 2.5x and is not obvious. Expert weights here are ZERO-COPY
 * HOST pointers. Handed to a kernel as a parameter, the driver sees the address at launch
 * and the access is a normal mapped read. Handed to it as opaque bytes inside a device
 * buffer, the driver cannot know which host pages the kernel will touch, and the reads
 * degrade to fault-driven mapping. Measured on Nemotron-H decode, everything else held
 * identical (same grid, same inner loop, same host-side block, tokens bit-identical):
 *
 *     weight pointer as kernel parameter   11.31 tok/s
 *     weight pointer read from a device buffer    4.02 tok/s
 *
 * This also CORRECTS the explanation recorded on `expert_seg_decode_enabled`, which
 * attributed that path's 2.4x regression to `nvfp4_matmul_seg` wasting 15/16 of a 16-row
 * MMA tile on a 1-row expert. That cannot be the cause: the GEMV here tiles nothing and
 * reproduces the same penalty. What both paths share is reading weight pointers out of
 * `sg_uw`/`sg_dw` device arrays.
 *
 * Sized for the largest routed top-k in the fleet with headroom; the caller declines
 * above it rather than silently truncating the expert set. 640 B, far under the 4 KB
 * parameter limit. */
#define SEGP_MAX 32
struct SegP {
    const uint8_t *w[SEGP_MAX];
    const uint8_t *bs[SEGP_MAX];
    float g[SEGP_MAX];
};

/* ---- SEGMENTED decode GEMV: one warp per (expert, output row) -------------------
 *
 * The decode expert path issues a launch trio per expert — 22 experts x 40 MoE layers =
 * 2640 launches per token on Nemotron-H — and each one is a GEMV over a single row. An
 * nsys profile measured 26,400 `nvfp4_gemv` launches averaging 15.4 us, which was the
 * whole of the routed-expert time. One expert's up-projection is only (2688+7)/8 = 337
 * blocks, so each launch leaves the machine mostly idle and they serialize on the stream.
 *
 * These kernels add an EXPERT AXIS to the grid (`blockIdx.y`) so a whole layer's experts
 * go out in one launch: same warps, same arithmetic, ~22x the blocks in flight and 3
 * launches per layer instead of 66.
 *
 * This is the "true segmented GEMV" that `expert_seg_decode_enabled` asks for and does
 * NOT provide: that path reuses `nvfp4_matmul_seg`, whose 16-row MMA tiles waste 15/16 of
 * their work on a 1-row expert and measured a 2.4x REGRESSION. Nothing here tiles rows.
 *
 * Layout: at decode every expert owns exactly one row, so expert `e` reads `x + e*K` and
 * writes `y + e*N` — which is already how the caller packs its pooled buffers.
 *
 * BIT-EXACTNESS: the inner loop, the per-lane `k` assignment and the shuffle reduction
 * are copied unchanged from the per-expert kernels above, so each output is bit-identical
 * to the unsegmented path. That makes the token-identity gate a real regression test for
 * this change rather than a formality. (The wide and narrow variants still differ from
 * *each other* in the last ULPs, exactly as they do unsegmented.)
 *
 * `w`/`bs` are device arrays of host (zero-copy) weight pointers, one per expert. */
__global__ static void nvfp4_gemv_seg(float *y,const float *x,
                                      const NvSegDesc *segs,int which,int K,int N){
    extern __shared__ float xs[];
    int e=blockIdx.y;
    const float *xe=x+(size_t)e*K;
    for(int k=threadIdx.x;k<K;k+=blockDim.x) xs[k]=xe[k];
    __syncthreads();
    int warp=threadIdx.x>>5,lane=threadIdx.x&31;
    int n=blockIdx.x*(blockDim.x>>5)+warp;
    if(n>=N) return;
    const NvSegDesc s=segs[e];
    const uint8_t *wb=which?s.dw:s.uw;
    const uint8_t *bb=which?s.dbs:s.ubs;
    float g=which?s.dg:s.ug;
    int Kh=(K+1)>>1, nb=(K+15)>>4;
    const uint8_t *wr=wb+(size_t)n*Kh;
    const uint8_t *br=bb+(size_t)n*nb;
    float acc=0.f;
    for(int k=lane;k<K;k+=32){
        uint8_t byte=wr[k>>1];
        int nib=(k&1)?(byte>>4):(byte&0xF);
        acc += xs[k]*e2m1f(nib)*e4m3f(br[k>>4]);
    }
    #pragma unroll
    for(int o=16;o>0;o>>=1) acc+=__shfl_down_sync(0xffffffff,acc,o);
    if(lane==0) y[(size_t)e*N+n]=acc*g;
}

/* Wide (16 B/lane) twin of `nvfp4_gemv_seg`, mirroring `nvfp4_gemv_u4`. Every expert's
 * weight must satisfy `nvfp4_u4_ok`, or the caller must use the narrow kernel above —
 * a misaligned `uint4` load is UB, not merely slow. */
__global__ static void nvfp4_gemv_u4_seg(float *y,const float *x,
                                         const NvSegDesc *segs,int which,int K,int N){
    extern __shared__ float xs[];
    int e=blockIdx.y;
    const float *xe=x+(size_t)e*K;
    for(int k=threadIdx.x;k<K;k+=blockDim.x) xs[k]=xe[k];
    __syncthreads();
    int warp=threadIdx.x>>5,lane=threadIdx.x&31;
    int n=blockIdx.x*(blockDim.x>>5)+warp;
    if(n>=N) return;
    const NvSegDesc s=segs[e];
    const uint8_t *wb=which?s.dw:s.uw;
    const uint8_t *bb=which?s.dbs:s.ubs;
    float g=which?s.dg:s.ug;
    int Kh=(K+1)>>1, nb=(K+15)>>4;
    const uint8_t *wr=wb+(size_t)n*Kh;
    const uint8_t *br=bb+(size_t)n*nb;
    const uint4 *wq=(const uint4*)wr;
    int Kq=Kh>>4;
    float acc=0.f;
    for(int i=lane;i<Kq;i+=32){
        uint4 v=wq[i];
        int k0=i<<5;
        float s0=e4m3f(br[k0>>4]);
        float s1=e4m3f(br[(k0+16)>>4]);
        uint32_t word[4]={v.x,v.y,v.z,v.w};
        #pragma unroll
        for(int wi=0;wi<4;wi++){
            float sc=(wi<2)?s0:s1;
            int kb=k0+(wi<<3);
            uint32_t wd=word[wi];
            float part=0.f;
            #pragma unroll
            for(int bj=0;bj<4;bj++){
                uint32_t byte=(wd>>(bj<<3))&0xFFu;
                int kk=kb+(bj<<1);
                part += xs[kk]*e2m1f(byte&0xF) + xs[kk+1]*e2m1f(byte>>4);
            }
            acc+=part*sc;
        }
    }
    #pragma unroll
    for(int o=16;o>0;o>>=1) acc+=__shfl_down_sync(0xffffffff,acc,o);
    if(lane==0) y[(size_t)e*N+n]=acc*g;
}

/* Parameter-space twin of `nvfp4_gemv_u4_seg` — see `SegP` for why this exists.
 * Identical inner loop and lane->k mapping, so it is bit-identical to `nvfp4_gemv_u4`. */
__global__ static void nvfp4_gemv_u4_segp(float *y,const float *x,SegP p,int K,int N){
    extern __shared__ float xs[];
    int e=blockIdx.y;
    const float *xe=x+(size_t)e*K;
    for(int k=threadIdx.x;k<K;k+=blockDim.x) xs[k]=xe[k];
    __syncthreads();
    int warp=threadIdx.x>>5,lane=threadIdx.x&31;
    int n=blockIdx.x*(blockDim.x>>5)+warp;
    if(n>=N) return;
    int Kh=(K+1)>>1, nb=(K+15)>>4;
    const uint8_t *wr=p.w[e]+(size_t)n*Kh;
    const uint8_t *br=p.bs[e]+(size_t)n*nb;
    const uint4 *wq=(const uint4*)wr;
    int Kq=Kh>>4;
    float acc=0.f;
    for(int i=lane;i<Kq;i+=32){
        uint4 v=wq[i];
        int k0=i<<5;
        float s0=e4m3f(br[k0>>4]);
        float s1=e4m3f(br[(k0+16)>>4]);
        uint32_t word[4]={v.x,v.y,v.z,v.w};
        #pragma unroll
        for(int wi=0;wi<4;wi++){
            float sc=(wi<2)?s0:s1;
            int kb=k0+(wi<<3);
            uint32_t wd=word[wi];
            float part=0.f;
            #pragma unroll
            for(int bj=0;bj<4;bj++){
                uint32_t byte=(wd>>(bj<<3))&0xFFu;
                int kk=kb+(bj<<1);
                part += xs[kk]*e2m1f(byte&0xF) + xs[kk+1]*e2m1f(byte>>4);
            }
            acc+=part*sc;
        }
    }
    #pragma unroll
    for(int o=16;o>0;o>>=1) acc+=__shfl_down_sync(0xffffffff,acc,o);
    if(lane==0) y[(size_t)e*N+n]=acc*p.g[e];
}

/* Can `nvfp4_gemv_u4` read this weight? Needs every row 16 B aligned.
 * `COLI_NVFP4_U4=0` forces the narrow kernel, so the two can be A/B'd in one binary. */
__host__ static inline int nvfp4_u4_ok(const uint8_t *w,int K){
    static int s_on=-1;
    if(s_on<0){const char*e=getenv("COLI_NVFP4_U4");s_on=e?atoi(e):1;}
    if(!s_on) return 0;
    int Kh=(K+1)>>1;
    int ok = ((Kh & 15)==0) && ((((uintptr_t)w) & 15)==0);
    /* Diagnostic (COLI_U4_REPORT=1): a silently-ineligible weight is indistinguishable
     * from "the wide kernel is no faster", and expert weights are views at arbitrary
     * safetensors offsets, so alignment is NOT a given. */
    static int s_rep=-1; if(s_rep<0){const char*e=getenv("COLI_U4_REPORT");s_rep=e?atoi(e):0;}
    if(s_rep){
        static long long yes=0,no=0,no_kh=0,no_ptr=0;
        if(ok) yes++; else { no++; if(Kh&15) no_kh++; if(((uintptr_t)w)&15) no_ptr++; }
        if(((yes+no)%500)==0)
            fprintf(stderr,"[u4] eligible=%lld ineligible=%lld (Kh%%16: %lld, ptr%%16: %lld)\n",
                    yes,no,no_kh,no_ptr);
    }
    return ok;
}

/* Wide-read NVFP4 GEMV — RESIDENT weights only (S==1 decode).
 *
 * `nvfp4_gemv` above assigns one k per lane and indexes `wr[k>>1]`, so lanes 2j and 2j+1
 * fetch the SAME byte: a warp covers just 16 contiguous bytes per step, a fraction of a
 * 128 B line. The int8 GEMV gives each lane its own byte and covers 32. That read width —
 * not the decode arithmetic — is the suspected reason resident NVFP4 achieved ~65 GB/s
 * against int8's ~89 on the same weights.
 *
 * Here each lane owns one BYTE and unpacks both of its nibbles, doubling bytes per warp
 * transaction and halving the trip count. A byte's two nibbles are k0=2*kb and k0+1 with
 * k0 even, so `k0>>4 == (k0+1)>>4` always: they share a block scale, which also saves one
 * ue4m3 decode per byte.
 *
 * Kept SEPARATE from `nvfp4_gemv` on purpose. Per-lane k assignment changes the f32
 * accumulation order, so this is a few ULP from the original — fine for resident weights
 * (gated on quality), but the expert path stays on the proven bit-exact kernel so its
 * token-identity gates keep meaning what they say. */
__global__ static void nvfp4_gemv_wide(float *y,const float *x,const uint8_t *w,
                                       const uint8_t *bs,float g,int K,int N){
    extern __shared__ float xs[];
    for(int k=threadIdx.x;k<K;k+=blockDim.x) xs[k]=x[k];
    __syncthreads();
    int warp=threadIdx.x>>5,lane=threadIdx.x&31;
    int n=blockIdx.x*(blockDim.x>>5)+warp;
    if(n>=N) return;
    int Kh=(K+1)>>1, nb=(K+15)>>4;
    const uint8_t *wr=w+(size_t)n*Kh;
    const uint8_t *br=bs+(size_t)n*nb;
    float acc=0.f;
    for(int kb=lane;kb<Kh;kb+=32){
        uint8_t byte=wr[kb];
        int k0=kb<<1;
        float sc=e4m3f(br[k0>>4]);          /* both nibbles share this block scale */
        float a=xs[k0]*e2m1f(byte&0xF);
        float b=(k0+1<K)?xs[k0+1]*e2m1f(byte>>4):0.f;
        acc+=(a+b)*sc;
    }
    #pragma unroll
    for(int o=16;o>0;o>>=1) acc+=__shfl_down_sync(0xffffffff,acc,o);
    if(lane==0) y[n]=acc*g;
}

/* Wide-read NVFP4 GEMV with NO shared staging of x — the occupancy variant.
 *
 * `nvfp4_gemv{,_wide}` stage x in shared memory: K floats, i.e. 16 KB per block at K=4096
 * and 32 KB at K=8192. That is a hard occupancy cap — at 32 KB an SM holds only ~3 blocks
 * (24 warps) where it could hold 8-16 — so far fewer memory requests are in flight than
 * the memory system can track. Resident NVFP4 decode achieves ~65 GB/s against a measured
 * 146 GB/s zero-copy ceiling, and too few concurrent readers is the likeliest reason.
 *
 * The trade is favourable: the WEIGHTS are the traffic (GB per token), while x is only
 * K floats (16-32 KB) shared by every warp in the grid — small enough to sit in L2, so
 * re-reading it from global costs little. Dropping the shared allocation lets many more
 * warps be resident, which is what actually raises memory-level parallelism.
 *
 * Same per-lane byte assignment (and shared block-scale) as `nvfp4_gemv_wide`. */
__global__ static void nvfp4_gemv_wide_g(float *y,const float *x,const uint8_t *w,
                                         const uint8_t *bs,float g,int K,int N){
    int warp=threadIdx.x>>5,lane=threadIdx.x&31;
    int n=blockIdx.x*(blockDim.x>>5)+warp;
    if(n>=N) return;
    int Kh=(K+1)>>1, nb=(K+15)>>4;
    const uint8_t *wr=w+(size_t)n*Kh;
    const uint8_t *br=bs+(size_t)n*nb;
    float acc=0.f;
    for(int kb=lane;kb<Kh;kb+=32){
        uint8_t byte=wr[kb];
        int k0=kb<<1;
        float sc=e4m3f(br[k0>>4]);
        float a=x[k0]*e2m1f(byte&0xF);
        float b=(k0+1<K)?x[k0+1]*e2m1f(byte>>4):0.f;
        acc+=(a+b)*sc;
    }
    #pragma unroll
    for(int o=16;o>0;o>>=1) acc+=__shfl_down_sync(0xffffffff,acc,o);
    if(lane==0) y[n]=acc*g;
}

/* Full-line NVFP4 GEMV — resident, S==1. Each lane loads a uint32 (4 bytes = 8 nibbles),
 * so a warp fetches 128 B: one whole cache line per step, versus 32 B for the byte version
 * and 16 B for the original.
 *
 * The 8 nibbles also share ONE block scale. Their k range starts at k0 = 8*kb4, so
 * k0 mod 16 is 0 or 8 and k0..k0+7 never straddles a 16-wide scale block — one ue4m3
 * decode per 8 values instead of per 2.
 *
 * Rows are Kh = ceil(K/2) bytes apart, so 4-byte alignment holds whenever Kh % 4 == 0;
 * the caller checks that and falls back to the byte version otherwise rather than issuing
 * a misaligned uint32 load. */
__global__ static void nvfp4_gemv_u32(float *y,const float *x,const uint8_t *w,
                                      const uint8_t *bs,float g,int K,int N){
    int warp=threadIdx.x>>5,lane=threadIdx.x&31;
    int n=blockIdx.x*(blockDim.x>>5)+warp;
    if(n>=N) return;
    int Kh=(K+1)>>1, nb=(K+15)>>4, Kw=Kh>>2;
    const uint32_t *wr=(const uint32_t*)(w+(size_t)n*Kh);
    const uint8_t *br=bs+(size_t)n*nb;
    float acc=0.f;
    for(int kw=lane;kw<Kw;kw+=32){
        uint32_t v=wr[kw];
        int k0=kw<<3;                       /* 8 nibbles per uint32 */
        float sc=e4m3f(br[k0>>4]);          /* k0..k0+7 share one block scale */
        float p=0.f;
        #pragma unroll
        for(int j=0;j<8;j++){
            int nib=(v>>(j<<2))&0xF;
            p+=x[k0+j]*e2m1f(nib);
        }
        acc+=p*sc;
    }
    /* tail: whatever bytes the uint32 sweep could not cover */
    for(int kb=(Kw<<2)+lane;kb<Kh;kb+=32){
        uint8_t byte=w[(size_t)n*Kh+kb];
        int k0=kb<<1;
        float sc=e4m3f(br[k0>>4]);
        float a=x[k0]*e2m1f(byte&0xF);
        float b=(k0+1<K)?x[k0+1]*e2m1f(byte>>4):0.f;
        acc+=(a+b)*sc;
    }
    #pragma unroll
    for(int o=16;o>0;o>>=1) acc+=__shfl_down_sync(0xffffffff,acc,o);
    if(lane==0) y[n]=acc*g;
}

/* ---------------------------------------------------------------------------------
 * MXFP4 (Kimi-K3) expert kernels.
 *
 * Same e2m1 nibbles as NVFP4 with exactly two differences, both localized:
 *   - one OCP E8M0 (power-of-two) block scale per 32 inputs, not a ue4m3 per 16;
 *   - no per-tensor global (a natively-MXFP4 tensor carries g == 1.0), kept as a
 *     parameter so both formats share one call shape.
 *
 * Written as separate kernels rather than by templating the NVFP4 ones: those are the
 * hot path for four shipped models, and changing their codegen to serve a fifth is not
 * a trade worth making for ~110 lines. K3's experts are natively MXFP4 and pass through
 * convert bit-exact, so this is the only kernel that can read them.
 * --------------------------------------------------------------------------------- */

/* Decode one OCP E8M0 byte: a bare power of two, 2^(b-127).
 *
 * `b` IS an IEEE-754 biased exponent, so 2^(b-127) is exactly the float whose exponent
 * field is `b` and whose mantissa is zero — one shift and a bit-reinterpret, and exact
 * for every b in 1..254.
 *
 * DELIBERATELY UNGUARDED at the two endpoints, which is a real (if unreachable)
 * deviation from the OCP spec:
 *   b = 0xFF -> +inf here, NaN per spec.
 *   b = 0    -> +0.0 here, 2^-127 (a SUBNORMAL) per spec.
 *
 * The guards cost more than everything else in the decode put together. `coli gpubench
 * 1 300` on the v4-expert triple (4096x2048), mode 2, isolating one thing at a time:
 *
 *   e8m0f with both endpoint guards        116.6 us      <- was the shipped form
 *   ... delegating to e4m3f (wrong values)  99.1
 *   ... branchless shift, no guards        100.0         <- this
 *   ... ablated to a constant (no load)     91.4
 *   nvfp4 baseline, same shape              97.5
 *
 * So the two comparisons cost 16.6 us — 14% of the whole triple — and removing them puts
 * MXFP4 at parity with NVFP4, which is where it should be given it reads 6% FEWER bytes.
 * (Branch form vs predicated-select form measured identically, so it is the comparisons,
 * not the control flow.) This is the entire MXFP4-vs-NVFP4 gap; nothing else moved it.
 *
 * Safe because neither value occurs, and could not matter if it did:
 *   - Scanned 177M real block-scale bytes — 600 routed-expert tensors sampled across
 *     DeepSeek-V4 (range 119..124, 6 distinct values) and Kimi-K3 (113..123, 11 values).
 *     ZERO b=0 and ZERO b=255 in either.
 *   - b=0 means every weight in that block is at most 6*2^-127 ~ 3.5e-38, which is below
 *     the ULP of an f32 accumulator that has seen a single normal-magnitude term. It
 *     cannot change a dot product it participates in.
 *   - b=255 is NaN, i.e. a corrupt checkpoint; +inf destroys the output just as loudly.
 *
 * `colibri_core::f8e8m0_to_f32` (the CPU reference) stays exact, so
 * `gpu::tests::mxfp4_expert_ffn_gpu_matches_cpu_at_every_s` would catch a divergence if a
 * checkpoint ever did carry those bytes. */
__device__ __forceinline__ static float e8m0f(uint8_t b) {
    return __int_as_float((int)b << 23);
}

/* Single-row decode GEMV (S==1 decode): one warp per output column. */
__global__ static void mxfp4_gemv(float *y,const float *x,const uint8_t *w,
                                  const uint8_t *bs,float g,int K,int N){
    extern __shared__ float xs[];
    for(int k=threadIdx.x;k<K;k+=blockDim.x) xs[k]=x[k];
    __syncthreads();
    int warp=threadIdx.x>>5,lane=threadIdx.x&31;
    int n=blockIdx.x*(blockDim.x>>5)+warp;
    if(n>=N) return;
    int Kh=(K+1)>>1, nb=(K+31)>>5;
    const uint8_t *wr=w+(size_t)n*Kh;
    const uint8_t *br=bs+(size_t)n*nb;
    float acc=0.f;
    for(int k=lane;k<K;k+=32){
        uint8_t byte=wr[k>>1];
        int nib=(k&1)?(byte>>4):(byte&0xF);
        acc += xs[k]*e2m1f(nib)*e8m0f(br[k>>5]);
    }
    #pragma unroll
    for(int o=16;o>0;o>>=1) acc+=__shfl_down_sync(0xffffffff,acc,o);
    if(lane==0) y[n]=acc*g;
}

/* Tiled WMMA matmul (S>1 prefill). Mirror of `nvfp4_matmul` with the MXFP4 decode. */
__global__ static void mxfp4_matmul(float *y,const float *x,const uint8_t *w,
                                    const uint8_t *bs,float g,int M,int K,int N){
#if __CUDA_ARCH__ >= 700
    using namespace nvcuda;int warp=threadIdx.x>>5,lane=threadIdx.x&31;
    int m0=blockIdx.y*16,n0=blockIdx.x*64+warp*16;
    int Kh=(K+1)>>1, nb=(K+31)>>5;
    __shared__ __half ah[256],bh[4][256];
    wmma::fragment<wmma::accumulator,16,16,16,float> acc;wmma::fill_fragment(acc,0.f);
    for(int k0=0;k0<K;k0+=16){
        for(int z=threadIdx.x;z<256;z+=blockDim.x){
            int m=z/16,k=z%16,gm=m0+m,gk=k0+k;
            ah[z]=(gm<M&&gk<K)?__float2half(x[(size_t)gm*K+gk]):__float2half(0.f);
        }
        for(int z=lane;z<256;z+=32){
            int nn=z/16,gk=k0+(z%16),gn=n0+nn;float v=0.f;
            if(gn<N&&gk<K){
                uint8_t byte=w[(size_t)gn*Kh+(gk>>1)];
                int nib=(gk&1)?(byte>>4):(byte&0xF);
                v=e2m1f(nib)*e8m0f(bs[(size_t)gn*nb+(gk>>5)])*g;
            }
            bh[warp][z]=__float2half(v);
        }
        __syncthreads();
        wmma::fragment<wmma::matrix_a,16,16,16,__half,wmma::row_major> af;
        wmma::fragment<wmma::matrix_b,16,16,16,__half,wmma::col_major> bf;
        wmma::load_matrix_sync(af,ah,16);wmma::load_matrix_sync(bf,bh[warp],16);
        wmma::mma_sync(acc,af,bf,acc);__syncthreads();
    }
    __shared__ float out[4][256];wmma::store_matrix_sync(out[warp],acc,16,wmma::mem_row_major);__syncwarp();
    for(int z=lane;z<256;z+=32){int m=z/16,nn=z%16;
        if(m0+m<M&&n0+nn<N)y[(size_t)(m0+m)*N+n0+nn]=out[warp][z];}
#endif
}

/* Tiled WMMA fused gate+up (S>1 prefill). Mirror of `nvfp4_gate_up` with the MXFP4 decode. */
__global__ static void mxfp4_gate_up(float *gate,float *up,const float *x,
        const uint8_t *gw,const uint8_t *uw,const uint8_t *gbs,const uint8_t *ubs,
        float gg,float ug,int M,int K,int N){
#if __CUDA_ARCH__ >= 700
    using namespace nvcuda;int warp=threadIdx.x>>5,lane=threadIdx.x&31,which=warp&1,tile=warp>>1;
    int m0=blockIdx.y*16,n0=blockIdx.x*64+tile*16;
    const uint8_t *w=which?uw:gw;const uint8_t *bs=which?ubs:gbs;
    float g=which?ug:gg;float *y=which?up:gate;
    int Kh=(K+1)>>1, nb=(K+31)>>5;
    __shared__ __half ah[256],bh[8][256];
    wmma::fragment<wmma::accumulator,16,16,16,float> acc;wmma::fill_fragment(acc,0.f);
    for(int k0=0;k0<K;k0+=16){
        for(int z=threadIdx.x;z<256;z+=blockDim.x){int m=z/16,k=z%16,gm=m0+m,gk=k0+k;
            ah[z]=(gm<M&&gk<K)?__float2half(x[(size_t)gm*K+gk]):__float2half(0.f);}
        for(int z=lane;z<256;z+=32){int nn=z/16,gk=k0+(z%16),gn=n0+nn;float v=0.f;
            if(gn<N&&gk<K){
                uint8_t byte=w[(size_t)gn*Kh+(gk>>1)];
                int nib=(gk&1)?(byte>>4):(byte&0xF);
                v=e2m1f(nib)*e8m0f(bs[(size_t)gn*nb+(gk>>5)])*g;
            }
            bh[warp][z]=__float2half(v);}
        __syncthreads();
        wmma::fragment<wmma::matrix_a,16,16,16,__half,wmma::row_major> af;
        wmma::fragment<wmma::matrix_b,16,16,16,__half,wmma::col_major> bf;
        wmma::load_matrix_sync(af,ah,16);wmma::load_matrix_sync(bf,bh[warp],16);
        wmma::mma_sync(acc,af,bf,acc);__syncthreads();
    }
    __shared__ float out[8][256];wmma::store_matrix_sync(out[warp],acc,16,wmma::mem_row_major);__syncwarp();
    for(int z=lane;z<256;z+=32){int m=z/16,nn=z%16;
        if(m0+m<M&&n0+nn<N)y[(size_t)(m0+m)*N+n0+nn]=out[warp][z];}
#endif
}

/* Kimi-K3 `situ`, fused into the gate/up combine:
 *     gate = beta*tanh(gate/beta)*sigmoid(gate) * linear_beta*tanh(up/linear_beta)
 * ASYMMETRIC — the gate half gets tanh*sigmoid, the up half only tanh — so the two
 * arguments are not interchangeable. `linear_beta <= 0` means "unset" (the reference's
 * `None`), leaving `up` a plain passthrough. Mirrors `math::situ` exactly. */
__global__ static void situ_mul(float *gate,const float *up,size_t n,
                                float beta,float linear_beta){
    size_t i=(size_t)blockIdx.x*blockDim.x+threadIdx.x;
    if(i>=n) return;
    float gv=gate[i], uv=up[i];
    float a = beta*tanhf(gv/beta)*(1.0f/(1.0f+__expf(-gv)));
    float u = (linear_beta>0.0f) ? linear_beta*tanhf(uv/linear_beta) : uv;
    gate[i]=a*u;
}

/* Tiled WMMA down-proj (S>1 prefill). Mirror of `fp8a16_matmul` with the nvfp4 decode. */
/* SEGMENTED NVFP4 matmul: one launch covers EVERY expert in a layer.
 *
 * Identical math and tiling to `nvfp4_matmul` below — 16 rows x 64 output columns per
 * block, WMMA over K in steps of 16 — but each block first looks up WHICH expert it
 * belongs to, so a layer's ~453 experts become one grid instead of ~453 launches.
 *
 * Why: the per-expert launch is OCCUPANCY-bound, not bandwidth- or compute-bound. At
 * I~2700 and ~25 rows/expert its grid is dim3((I+63)/64,(S+15)/16) = 43 x 2 = 86 blocks
 * (~11k threads), which cannot keep enough memory requests in flight to hide host-memory
 * latency: measured 6.2 GB/s, 2.3% of this chip's ~273 GB/s and 12% of the ~51 GB/s
 * zero-copy host ceiling, while compute sat at 0.26% of peak. Feeding the same kernel 4x
 * the rows per expert (2048- vs 512-token prompt) improved per-token cost 2.75x with no
 * code change, which is what this kernel buys at any prompt length: ~453x the blocks.
 *
 * Tile descriptors: `tile_e[t]` is the expert owning row-tile t and `tile_r0[t]` its
 * first row WITHIN that expert. `e_off[e]` is where expert e's rows start in the packed
 * x/y buffers, `e_rows[e]` how many it has. Weight pointers are per expert and may be
 * host (zero-copy) or device — the kernel does not care. */
__global__ static void nvfp4_matmul_seg(float *y, const float *x,
        const uint8_t *const *ws, const uint8_t *const *bss, const float *gs,
        const int *tile_e, const int *tile_r0, const int *e_off, const int *e_rows,
        int K, int N) {
#if __CUDA_ARCH__ >= 700
    using namespace nvcuda; int warp = threadIdx.x >> 5, lane = threadIdx.x & 31;
    int t = blockIdx.y, e = tile_e[t];
    const uint8_t *w = ws[e], *bs = bss[e];
    float g = gs[e];
    int M = e_rows[e];                       // rows this expert owns
    int m0 = tile_r0[t];                     // first row of this tile, expert-local
    size_t base = (size_t)e_off[e];          // expert's first row in the packed buffers
    int n0 = blockIdx.x * 64 + warp * 16;
    int Kh = (K + 1) >> 1, nb = (K + 15) >> 4;
    __shared__ __half ah[256], bh[4][256];
    wmma::fragment<wmma::accumulator,16,16,16,float> acc; wmma::fill_fragment(acc, 0.f);
    for (int k0 = 0; k0 < K; k0 += 16) {
        for (int z = threadIdx.x; z < 256; z += blockDim.x) {
            int m = z / 16, k = z % 16, gm = m0 + m, gk = k0 + k;
            ah[z] = (gm < M && gk < K)
                ? __float2half(x[(base + gm) * (size_t)K + gk]) : __float2half(0.f);
        }
        for (int z = lane; z < 256; z += 32) {
            int nn = z / 16, gk = k0 + (z % 16), gn = n0 + nn; float v = 0.f;
            if (gn < N && gk < K) {
                uint8_t byte = w[(size_t)gn * Kh + (gk >> 1)];
                int nib = (gk & 1) ? (byte >> 4) : (byte & 0xF);
                v = e2m1f(nib) * e4m3f(bs[(size_t)gn * nb + (gk >> 4)]) * g;
            }
            bh[warp][z] = __float2half(v);
        }
        __syncthreads();
        wmma::fragment<wmma::matrix_a,16,16,16,__half,wmma::row_major> af;
        wmma::fragment<wmma::matrix_b,16,16,16,__half,wmma::col_major> bf;
        wmma::load_matrix_sync(af, ah, 16); wmma::load_matrix_sync(bf, bh[warp], 16);
        wmma::mma_sync(acc, af, bf, acc); __syncthreads();
    }
    __shared__ float out[4][256];
    wmma::store_matrix_sync(out[warp], acc, 16, wmma::mem_row_major); __syncwarp();
    for (int z = lane; z < 256; z += 32) {
        int m = z / 16, nn = z % 16;
        if (m0 + m < M && n0 + nn < N)
            y[(base + m0 + m) * (size_t)N + n0 + nn] = out[warp][z];
    }
#endif
}

__global__ static void nvfp4_matmul(float *y,const float *x,const uint8_t *w,
                                    const uint8_t *bs,float g,int M,int K,int N){
#if __CUDA_ARCH__ >= 700
    using namespace nvcuda;int warp=threadIdx.x>>5,lane=threadIdx.x&31;
    int m0=blockIdx.y*16,n0=blockIdx.x*64+warp*16;
    int Kh=(K+1)>>1, nb=(K+15)>>4;
    __shared__ __half ah[256],bh[4][256];
    wmma::fragment<wmma::accumulator,16,16,16,float> acc;wmma::fill_fragment(acc,0.f);
    for(int k0=0;k0<K;k0+=16){
        for(int z=threadIdx.x;z<256;z+=blockDim.x){
            int m=z/16,k=z%16,gm=m0+m,gk=k0+k;
            ah[z]=(gm<M&&gk<K)?__float2half(x[(size_t)gm*K+gk]):__float2half(0.f);
        }
        for(int z=lane;z<256;z+=32){
            int nn=z/16,gk=k0+(z%16),gn=n0+nn;float v=0.f;
            if(gn<N&&gk<K){
                uint8_t byte=w[(size_t)gn*Kh+(gk>>1)];
                int nib=(gk&1)?(byte>>4):(byte&0xF);
                v=e2m1f(nib)*e4m3f(bs[(size_t)gn*nb+(gk>>4)])*g;
            }
            bh[warp][z]=__float2half(v);
        }
        __syncthreads();
        wmma::fragment<wmma::matrix_a,16,16,16,__half,wmma::row_major> af;
        wmma::fragment<wmma::matrix_b,16,16,16,__half,wmma::col_major> bf;
        wmma::load_matrix_sync(af,ah,16);wmma::load_matrix_sync(bf,bh[warp],16);
        wmma::mma_sync(acc,af,bf,acc);__syncthreads();
    }
    __shared__ float out[4][256];wmma::store_matrix_sync(out[warp],acc,16,wmma::mem_row_major);__syncwarp();
    for(int z=lane;z<256;z+=32){int m=z/16,nn=z%16;
        if(m0+m<M&&n0+nn<N)y[(size_t)(m0+m)*N+n0+nn]=out[warp][z];}
#endif
}

/* Tiled WMMA fused gate+up (S>1 prefill). Mirror of `fp8a16_gate_up`; each weight carries
 * its own block-scale array + global. */
__global__ static void nvfp4_gate_up(float *gate,float *up,const float *x,
        const uint8_t *gw,const uint8_t *uw,const uint8_t *gbs,const uint8_t *ubs,
        float gg,float ug,int M,int K,int N){
#if __CUDA_ARCH__ >= 700
    using namespace nvcuda;int warp=threadIdx.x>>5,lane=threadIdx.x&31,which=warp&1,tile=warp>>1;
    int m0=blockIdx.y*16,n0=blockIdx.x*64+tile*16;
    const uint8_t *w=which?uw:gw;const uint8_t *bs=which?ubs:gbs;
    float g=which?ug:gg;float *y=which?up:gate;
    int Kh=(K+1)>>1, nb=(K+15)>>4;
    __shared__ __half ah[256],bh[8][256];
    wmma::fragment<wmma::accumulator,16,16,16,float> acc;wmma::fill_fragment(acc,0.f);
    for(int k0=0;k0<K;k0+=16){
        for(int z=threadIdx.x;z<256;z+=blockDim.x){int m=z/16,k=z%16,gm=m0+m,gk=k0+k;
            ah[z]=(gm<M&&gk<K)?__float2half(x[(size_t)gm*K+gk]):__float2half(0.f);}
        for(int z=lane;z<256;z+=32){int nn=z/16,gk=k0+(z%16),gn=n0+nn;float v=0.f;
            if(gn<N&&gk<K){
                uint8_t byte=w[(size_t)gn*Kh+(gk>>1)];
                int nib=(gk&1)?(byte>>4):(byte&0xF);
                v=e2m1f(nib)*e4m3f(bs[(size_t)gn*nb+(gk>>4)])*g;
            }
            bh[warp][z]=__float2half(v);}
        __syncthreads();
        wmma::fragment<wmma::matrix_a,16,16,16,__half,wmma::row_major> af;
        wmma::fragment<wmma::matrix_b,16,16,16,__half,wmma::col_major> bf;
        wmma::load_matrix_sync(af,ah,16);wmma::load_matrix_sync(bf,bh[warp],16);
        wmma::mma_sync(acc,af,bf,acc);__syncthreads();
    }
    __shared__ float out[8][256];wmma::store_matrix_sync(out[warp],acc,16,wmma::mem_row_major);__syncwarp();
    for(int z=lane;z<256;z+=32){int m=z/16,nn=z%16;
        if(m0+m<M&&n0+nn<N)y[(size_t)(m0+m)*N+n0+nn]=out[warp][z];}
#endif
}

/* Weight-stationary small-M NVFP4 matmul: y[M,N] = x[M,K] @ dequant(W[N,K])^T, the
 * SAME contract as `nvfp4_matmul`. One warp per output column `n`; the 32 lanes split K,
 * and each lane holds MT per-row accumulators so a weight element is read + dequantized
 * EXACTLY ONCE and reused across all M rows. The WMMA path instead re-dequantizes the
 * weight once per 16-row m-tile and, at the ~26 rows/expert of a routed prefill, runs its
 * 16x16 MMA at ~1/8 utilization — measured 0.26% of tensor peak, weight-read bound (#90).
 * Reading the weight once amortizes the dequant over M rows, which is the whole cost.
 *
 * MT is a compile-time bucket so `acc[MT]` stays in registers (a runtime length spills to
 * local memory and erases the win); the caller dispatches the smallest bucket >= M and the
 * kernel zero-pads rows [M,MT). x is staged into shared per K-tile so it is read from
 * device once per block, not once per column. M > the largest bucket falls back to WMMA. */
template<int MT>
__global__ static void nvfp4_wsmm(float *y,const float *x,const uint8_t *w,
        const uint8_t *bs,float g,int M,int K,int N){
    const int KT=128;
    extern __shared__ float wsmm_xs[];      // [MT][KT], row-major m*KT+kk
    int warp=threadIdx.x>>5, lane=threadIdx.x&31, wpb=blockDim.x>>5;
    int n=blockIdx.x*wpb+warp;              // output column this warp owns
    int Kh=(K+1)>>1, nb=(K+15)>>4;
    const uint8_t *wr=w+(size_t)(n<N?n:0)*Kh;
    const uint8_t *br=bs+(size_t)(n<N?n:0)*nb;
    float acc[MT];
    #pragma unroll
    for(int m=0;m<MT;m++) acc[m]=0.f;
    for(int k0=0;k0<K;k0+=KT){
        int kt=min(KT,K-k0);
        for(int idx=threadIdx.x; idx<M*kt; idx+=blockDim.x){
            int m=idx/kt, kk=idx-m*kt;
            wsmm_xs[m*KT+kk]=x[(size_t)m*K+k0+kk];
        }
        __syncthreads();
        if(n<N){
            for(int kk=lane;kk<kt;kk+=32){
                int k=k0+kk;
                uint8_t byte=wr[k>>1];
                int nib=(k&1)?(byte>>4):(byte&0xF);
                float wv=e2m1f(nib)*e4m3f(br[k>>4]);
                #pragma unroll
                for(int m=0;m<MT;m++){
                    float xv=(m<M)?wsmm_xs[m*KT+kk]:0.f;
                    acc[m]+=xv*wv;
                }
            }
        }
        __syncthreads();
    }
    if(n<N){
        #pragma unroll
        for(int m=0;m<MT;m++){
            if(m>=M) continue;
            float a=acc[m];
            #pragma unroll
            for(int o=16;o>0;o>>=1) a+=__shfl_down_sync(0xffffffff,a,o);
            if(lane==0) y[(size_t)m*N+n]=a*g;
        }
    }
}

/* Dispatch the weight-stationary kernel at the smallest MT bucket >= M. Returns false if
 * M exceeds the largest bucket (caller keeps the WMMA path). blockDim = 128 (4 warps). */
static bool nvfp4_wsmm_launch(float *y,const float *x,const uint8_t *w,const uint8_t *bs,
        float g,int M,int K,int N,cudaStream_t s){
    const int TPB=128, wpb=TPB>>5;
    dim3 grid((unsigned)((N+wpb-1)/wpb));
    #define WSMM_CASE(MT) do{ size_t sh=(size_t)(MT)*128*sizeof(float); \
        nvfp4_wsmm<MT><<<grid,TPB,sh,s>>>(y,x,w,bs,g,M,K,N); }while(0)
    if(M<=8) WSMM_CASE(8);
    else if(M<=16) WSMM_CASE(16);
    else if(M<=32) WSMM_CASE(32);
    else return false;
    #undef WSMM_CASE
    return true;
}

/* Weight-stationary small-M MXFP4 matmul — the fmt-6 mirror of `nvfp4_wsmm` above, with
 * the block-16 e4m3 scale swapped for MXFP4's block-32 E8M0. Same contract, same MT
 * bucketing, same reason: at prefill a routed expert sees only S = tokens*top_k/n_experts
 * rows (DeepSeek-V4: 512*6/256 = 12; Kimi-K3 similar), so `mxfp4_matmul`'s 16x16 WMMA tile
 * runs at a fraction of its utilization AND re-dequantizes the whole weight once per 16-row
 * m-tile. #90 measured that shape at 0.26% of tensor peak on the NVFP4 side and fixed it
 * there; fmt 6 kept the WMMA tile because nothing enumerated it — the same closed-set
 * dispatch trap that left the decode GEMV narrow.
 *
 * NOT applicable to decode: V4 routes top-6 of 256, so a decode expert sees exactly ONE
 * row and there is no reuse to amortize the dequant over. Decode's lever is the GEMV read
 * pattern below, not this. */
template<int MT>
__global__ static void mxfp4_wsmm(float *y,const float *x,const uint8_t *w,
        const uint8_t *bs,float g,int M,int K,int N){
    const int KT=128;
    extern __shared__ float mxwsmm_xs[];    // [MT][KT], row-major m*KT+kk
    int warp=threadIdx.x>>5, lane=threadIdx.x&31, wpb=blockDim.x>>5;
    int n=blockIdx.x*wpb+warp;              // output column this warp owns
    int Kh=(K+1)>>1, nb=(K+31)>>5;
    const uint8_t *wr=w+(size_t)(n<N?n:0)*Kh;
    const uint8_t *br=bs+(size_t)(n<N?n:0)*nb;
    float acc[MT];
    #pragma unroll
    for(int m=0;m<MT;m++) acc[m]=0.f;
    for(int k0=0;k0<K;k0+=KT){
        int kt=min(KT,K-k0);
        for(int idx=threadIdx.x; idx<M*kt; idx+=blockDim.x){
            int m=idx/kt, kk=idx-m*kt;
            mxwsmm_xs[m*KT+kk]=x[(size_t)m*K+k0+kk];
        }
        __syncthreads();
        if(n<N){
            for(int kk=lane;kk<kt;kk+=32){
                int k=k0+kk;
                uint8_t byte=wr[k>>1];
                int nib=(k&1)?(byte>>4):(byte&0xF);
                float wv=e2m1f(nib)*e8m0f(br[k>>5]);
                #pragma unroll
                for(int m=0;m<MT;m++){
                    float xv=(m<M)?mxwsmm_xs[m*KT+kk]:0.f;
                    acc[m]+=xv*wv;
                }
            }
        }
        __syncthreads();
    }
    if(n<N){
        #pragma unroll
        for(int m=0;m<MT;m++){
            if(m>=M) continue;
            float a=acc[m];
            #pragma unroll
            for(int o=16;o>0;o>>=1) a+=__shfl_down_sync(0xffffffff,a,o);
            if(lane==0) y[(size_t)m*N+n]=a*g;
        }
    }
}

/* Dispatch the MXFP4 weight-stationary kernel at the smallest MT bucket >= M. Mirror of
 * `nvfp4_wsmm_launch`: returns false above the largest bucket so the caller keeps WMMA.
 *
 * `COLI_MXFP4_WSMM=0` forces every call to decline, i.e. restores the pre-port WMMA path.
 * The NVFP4 twin deliberately has no such knob — S is the only thing that should decide,
 * and an override can only disagree with the shape actually being computed. **This is NOT
 * a tuning knob and must not be used as one**; it exists because the WMMA path is otherwise
 * unreachable at these S, so without it there is no way to A/B this kernel against what it
 * replaced.
 *
 * It stays now the port is measured (1.19x, `97e8b86`), rather than being removed as that
 * commit implied. Reason, learned the hard way one commit later: `e8m0f` had no equivalent
 * knob, so A/B-ing it meant hand-patching the source, building a second binary, and
 * keeping the two straight — slower and easier to get wrong than `COLI_MXFP4_WSMM=0`.
 * Same standing as `COLI_NVFP4_GEMV`'s modes, which are kept for exactly this. */
static bool mxfp4_wsmm_launch(float *y,const float *x,const uint8_t *w,const uint8_t *bs,
        float g,int M,int K,int N,cudaStream_t s){
    static int s_on=-1;
    if(s_on<0){const char*e=getenv("COLI_MXFP4_WSMM");s_on=e?atoi(e):1;}
    if(!s_on) return false;
    const int TPB=128, wpb=TPB>>5;
    dim3 grid((unsigned)((N+wpb-1)/wpb));
    #define MXWSMM_CASE(MT) do{ size_t sh=(size_t)(MT)*128*sizeof(float); \
        mxfp4_wsmm<MT><<<grid,TPB,sh,s>>>(y,x,w,bs,g,M,K,N); }while(0)
    if(M<=8) MXWSMM_CASE(8);
    else if(M<=16) MXWSMM_CASE(16);
    else if(M<=32) MXWSMM_CASE(32);
    else return false;
    #undef MXWSMM_CASE
    return true;
}

/* Launch the S==1 NVFP4 GEMV under COLI_NVFP4_GEMV:
 *   0 = narrow (original: one nibble-pair byte shared by lanes 2j/2j+1, x staged in shared)
 *   1 = one byte per lane + shared x
 *   2 = one byte per lane, x read from device (frees the shared-mem occupancy cap)
 *   3 = uint32 per lane — a full 128 B/warp cache line (default; needs Kh % 4 == 0)
 *
 * ONE dispatcher for every NVFP4 decode GEMV. The read-pattern work (#46) originally lived
 * inline in `coli_cuda_matmul_nvfp4`, so it reached the *resident* weights only and both
 * expert MLPs kept launching the narrow kernel — the closed-set dispatch trap, and the
 * reason that win read as "+9.4%, rest is Amdahl" when routed experts are the larger half
 * of a decode step. Route every call site here so a read-pattern change lands everywhere.
 *
 * Mode is resolved once: within a call site K is fixed, so draft and verify pick the same
 * kernel and the MTP `exact` path stays reduction-order-identical to sequential decode. */
/* `nvfp4_gemv_u32` casts the weight row to `const uint32_t*`, so it needs BOTH a
 * 4-aligned row stride and a 4-aligned base. The stride test alone is not enough: a
 * resident weight comes from `cudaMalloc` (256-aligned, so the base is free), but an
 * EXPERT weight is a view at an arbitrary safetensors offset. Routing expert GEMVs through
 * a stride-only guard faults with `misaligned address`, and because a CUDA context error is
 * sticky that takes down every later call — the engine silently falls back to CPU and the
 * model gets ~3x slower with the GPU at 0%. Mirrors `nvfp4_u4_ok`, which learned this
 * first ("expert weights are views at arbitrary safetensors offsets"). */
__host__ static inline int nvfp4_u32_ok(const uint8_t *w,int K){
    int Kh=(K+1)>>1;
    return ((Kh&3)==0) && ((((uintptr_t)w)&3)==0);
}

/* Default is 2 (byte/lane, no shared x), NOT 3 (uint32/lane), and the reason is
 * correctness rather than speed.
 *
 * Mode 3 is selected per call by `nvfp4_u32_ok`, which tests the weight POINTER. Expert
 * weights come from a recycling buffer pool, so the same expert lands at a different heap
 * address on every run — mode 3 for one run, the mode-2 fallback for the next, two
 * different float summation orders, two different answers. MiniMax-M2.7 decode diverged at
 * token 7 across three identical back-to-back runs; forcing mode 2 made all three
 * byte-identical. (mmap-served spans are unaffected: page-aligned base plus a fixed file
 * offset is reproducible, which is why short warm runs looked deterministic.)
 *
 * It also costs nothing to fix. M2.7 decode, two pairs: mode 3 gave 6.57/6.69 tok/s and
 * mode 2 gave 7.03/6.87 — mode 2 is *faster*, matching the earlier finding that read width
 * stops paying past 32 B/warp. Mode 3 was buying non-determinism for no throughput.
 *
 * The modes remain selectable via COLI_NVFP4_GEMV for A/B; 3 is non-deterministic on any
 * pooled buffer and must not be used to produce reference output. */
static void nvfp4_gemv_dispatch(float *y,const float *x,const uint8_t *w,const uint8_t *bs,
        float g,int K,int N,cudaStream_t s){
    static int s_mode=-1;
    if(s_mode<0){const char*e=getenv("COLI_NVFP4_GEMV");s_mode=e?atoi(e):2;}
    const int tpb=256,wpb=tpb>>5;
    unsigned blocks=(unsigned)((N+wpb-1)/wpb);
    size_t shm=(size_t)K*sizeof(float);
    if(s_mode==0)      nvfp4_gemv     <<<blocks,tpb,shm,s>>>(y,x,w,bs,g,K,N);
    else if(s_mode==1) nvfp4_gemv_wide<<<blocks,tpb,shm,s>>>(y,x,w,bs,g,K,N);
    else if(s_mode==2) nvfp4_gemv_wide_g<<<blocks,tpb,0,s>>>(y,x,w,bs,g,K,N);
    else if(nvfp4_u32_ok(w,K)) nvfp4_gemv_u32<<<blocks,tpb,0,s>>>(y,x,w,bs,g,K,N);
    else               nvfp4_gemv_wide_g<<<blocks,tpb,0,s>>>(y,x,w,bs,g,K,N);
}

/* ---------------------------------------------------------------------------------
 * MXFP4 decode GEMV read patterns — the fmt-6 mirror of the NVFP4 family above.
 *
 * This exists because the read-pattern work (#46) was done ONE FORMAT OVER. It added
 * `nvfp4_gemv_wide_g`/`_u32` and routed every fmt-5 call site through a single
 * dispatcher, precisely so a read-pattern change would land everywhere — and fmt 6
 * kept `mxfp4_gemv`, whose loop is `for(k=lane;k<K;k+=32)` with `byte=wr[k>>1]`. Lanes
 * 2j and 2j+1 therefore load the SAME byte and a warp fetches 16 B per step where the
 * wide kernel fetches 32 B. That is the closed-set dispatch trap the comment above
 * warns about, one arm short.
 *
 * It matters more here than it did there: DeepSeek-V4 routes top-6 of 256 experts, so
 * every routed expert in decode is an S==1 GEMV, and `gpu-ffn` is the largest single
 * sub-phase of a V4 decode step.
 *
 * The narrow kernel scales per element (`x*e2m1*e8m0`); these scale per 2-nibble byte
 * (`(a+b)*sc`) exactly as the NVFP4 versions do. Same value, different summation order,
 * so output is NOT bit-identical to the old kernel — correctness is against the CPU
 * dequant reference, not against `mxfp4_gemv`.
 * --------------------------------------------------------------------------------- */

/* One byte (2 nibbles) per lane, x read straight from device — no shared staging, so no
 * shared-memory occupancy cap. Mirror of `nvfp4_gemv_wide_g` with block-32 E8M0. */
__global__ static void mxfp4_gemv_wide_g(float *y,const float *x,const uint8_t *w,
                                         const uint8_t *bs,float g,int K,int N){
    int warp=threadIdx.x>>5,lane=threadIdx.x&31;
    int n=blockIdx.x*(blockDim.x>>5)+warp;
    if(n>=N) return;
    int Kh=(K+1)>>1, nb=(K+31)>>5;
    const uint8_t *wr=w+(size_t)n*Kh;
    const uint8_t *br=bs+(size_t)n*nb;
    float acc=0.f;
    for(int kb=lane;kb<Kh;kb+=32){
        uint8_t byte=wr[kb];
        int k0=kb<<1;
        float sc=e8m0f(br[k0>>5]);
        float a=x[k0]*e2m1f(byte&0xF);
        float b=(k0+1<K)?x[k0+1]*e2m1f(byte>>4):0.f;
        acc+=(a+b)*sc;
    }
    #pragma unroll
    for(int o=16;o>0;o>>=1) acc+=__shfl_down_sync(0xffffffff,acc,o);
    if(lane==0) y[n]=acc*g;
}

/* Full-line MXFP4 GEMV: each lane loads a uint32 (4 bytes = 8 nibbles), so a warp fetches
 * one 128 B cache line per step. k0 = kw*8 is 8-aligned, so all 8 nibbles fall inside one
 * 32-element MXFP4 block and share `br[k0>>5]` — the block-32 layout makes this strictly
 * safer than the block-16 NVFP4 version, which needed the same argument at 16.
 *
 * Selected only under an explicit COLI_MXFP4_GEMV=3; see the dispatcher for why it is not
 * the default. */
__global__ static void mxfp4_gemv_u32(float *y,const float *x,const uint8_t *w,
                                      const uint8_t *bs,float g,int K,int N){
    int warp=threadIdx.x>>5,lane=threadIdx.x&31;
    int n=blockIdx.x*(blockDim.x>>5)+warp;
    if(n>=N) return;
    int Kh=(K+1)>>1, nb=(K+31)>>5, Kw=Kh>>2;
    const uint32_t *wr=(const uint32_t*)(w+(size_t)n*Kh);
    const uint8_t *br=bs+(size_t)n*nb;
    float acc=0.f;
    for(int kw=lane;kw<Kw;kw+=32){
        uint32_t v=wr[kw];
        int k0=kw<<3;                       /* 8 nibbles per uint32 */
        float sc=e8m0f(br[k0>>5]);          /* k0..k0+7 share one block scale */
        float p=0.f;
        #pragma unroll
        for(int j=0;j<8;j++){
            int nib=(v>>(j<<2))&0xF;
            p+=x[k0+j]*e2m1f(nib);
        }
        acc+=p*sc;
    }
    /* tail: whatever bytes the uint32 sweep could not cover */
    for(int kb=(Kw<<2)+lane;kb<Kh;kb+=32){
        uint8_t byte=w[(size_t)n*Kh+kb];
        int k0=kb<<1;
        float sc=e8m0f(br[k0>>5]);
        float a=x[k0]*e2m1f(byte&0xF);
        float b=(k0+1<K)?x[k0+1]*e2m1f(byte>>4):0.f;
        acc+=(a+b)*sc;
    }
    #pragma unroll
    for(int o=16;o>0;o>>=1) acc+=__shfl_down_sync(0xffffffff,acc,o);
    if(lane==0) y[n]=acc*g;
}

/* ONE dispatcher for every MXFP4 decode GEMV, under COLI_MXFP4_GEMV:
 *   0 = narrow (the original `mxfp4_gemv`: one byte shared by lanes 2j/2j+1, x in shared)
 *   2 = one byte per lane, x from device  (default)
 *   3 = uint32 per lane — a full 128 B/warp cache line (needs Kh % 4 == 0 and a 4-aligned
 *       base; `nvfp4_u32_ok` tests both and is format-independent)
 *
 * Mode 1 is deliberately absent: it was the shared-x variant of mode 2, and mode 2 beat it
 * on the NVFP4 side by freeing the occupancy cap. There is no reason to port a kernel that
 * lost.
 *
 * Default 2, not 3, for the SAME correctness reason the NVFP4 dispatcher documents: mode 3
 * is chosen per call from the weight POINTER, and expert weights come from a recycling
 * buffer pool, so the same expert lands at a different address run to run — mode 3 on one
 * run, mode 2 on the next, two summation orders, two answers. That produced a real decode
 * divergence on M2.7 and there mode 2 was also faster (read width stops paying past
 * 32 B/warp). Do not use mode 3 to produce reference output. */
static void mxfp4_gemv_dispatch(float *y,const float *x,const uint8_t *w,const uint8_t *bs,
        float g,int K,int N,cudaStream_t s){
    static int s_mode=-1;
    if(s_mode<0){const char*e=getenv("COLI_MXFP4_GEMV");s_mode=e?atoi(e):2;}
    const int tpb=256,wpb=tpb>>5;
    unsigned blocks=(unsigned)((N+wpb-1)/wpb);
    size_t shm=(size_t)K*sizeof(float);
    if(s_mode==0)      mxfp4_gemv     <<<blocks,tpb,shm,s>>>(y,x,w,bs,g,K,N);
    else if(s_mode==3&&nvfp4_u32_ok(w,K)) mxfp4_gemv_u32<<<blocks,tpb,0,s>>>(y,x,w,bs,g,K,N);
    else               mxfp4_gemv_wide_g<<<blocks,tpb,0,s>>>(y,x,w,bs,g,K,N);
}

/* Single-row decode GEMV (S==1) for int8 W8A16 — mirror of `fp8a16_gemv` with the
 * weight decode swapped e4m3 -> signed int8 (1 byte/weight, direct K stride). One warp
 * per output column; all 32 lanes sweep the row coalesced, so the grid is O/warps-per-
 * block blocks instead of the tiled kernel's (O/64, 1).
 *
 * This is the decode shape for the DENSE resident matmuls — Nemotron-H's mamba
 * in/out-proj (28.7% of decode), shared expert (20.6%), fc1/fc2 and the attention
 * projections. `i8a16_matmul` wastes 15/16 of its MMA at S==1; falling back to
 * `quant_matmul` (PR #13) already bought 1.53x, and this replaces that generic path
 * with the same one-warp-per-column shape the fp8/nvfp4 expert GEMVs use.
 *
 * Scale is applied once to the reduced accumulator (as `fp8a16_gemv` does) rather than
 * per weight: `sum(x*w)*scale` factors exactly, and it keeps the inner loop a plain fma. */
__global__ static void i8a16_gemv(float *y,const float *x,const uint8_t *w,
                                  const float *scale,int K,int N){
    extern __shared__ float xs[];
    for(int k=threadIdx.x;k<K;k+=blockDim.x) xs[k]=x[k];
    __syncthreads();
    int warp=threadIdx.x>>5,lane=threadIdx.x&31;
    int n=blockIdx.x*(blockDim.x>>5)+warp;
    if(n>=N) return;
    const signed char *wr=(const signed char*)w+(size_t)n*K; float acc=0.f;
    for(int k=lane;k<K;k+=32) acc+=xs[k]*(float)wr[k];
    #pragma unroll
    for(int o=16;o>0;o>>=1) acc+=__shfl_down_sync(0xffffffff,acc,o);
    if(lane==0) y[n]=acc*scale[n];
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

/* Nemotron-H Mamba2 selective-scan, one decode token (S==1). One block per head, one
 * thread per head-dim row p; each thread loops the d_state axis, updating that row's
 * SSM state and reducing y. B/C are shared per group g=h/(nh/ng); dt_h/dA_h are the
 * host-precomputed per-head step/decay (softplus/exp already applied on the host so
 * they match the CPU reference). The recurrence uses __fmul_rn/__fadd_rn (no FMA
 * contraction) in the exact operand order of the CPU `selective_scan`, so the result
 * is bit-identical: ssm = ssm*dA + dt*B*x ; y = sum_n ssm*C + x*D. */
__global__ static void mamba2_scan_kernel(float *state, float *y, const float *hidden,
        const float *b, const float *c, const float *dt_h, const float *da_h,
        const float *d, int nh, int hd, int ds, int ng) {
    int h = blockIdx.x, pp = threadIdx.x;
    if (h >= nh || pp >= hd) return;
    int hpg = nh / ng, grp = h / hpg;
    const float *b_row = b + (size_t)grp * ds;
    const float *c_row = c + (size_t)grp * ds;
    float dth = dt_h[h], dah = da_h[h], dh = d[h];
    float x_hp = hidden[(size_t)h * hd + pp];
    size_t base = ((size_t)h * hd + pp) * ds;
    float acc = 0.f;
    for (int nn = 0; nn < ds; nn++) {
        float ss = state[base + nn];
        // ss = ss*dA + (dt*B)*x  — same left-to-right f32 order as the CPU scan.
        float upd = __fadd_rn(__fmul_rn(ss, dah),
                              __fmul_rn(__fmul_rn(dth, b_row[nn]), x_hp));
        state[base + nn] = upd;
        acc = __fadd_rn(acc, __fmul_rn(upd, c_row[nn]));   // y += ss*C
    }
    y[(size_t)h * hd + pp] = __fadd_rn(acc, __fmul_rn(x_hp, dh));  // + x*D
}

/* Nemotron-H Mamba2 selective-scan over a WHOLE prefill sequence (S>1).
 *
 * ⚠️ CONTRACT DIFFERS FROM THE S==1 KERNEL: this one is **token-identical, not
 * bit-identical**, and that is a deliberate trade. See below.
 *
 * Decomposition. The recurrence `ss = ss*dA + (dt*B)*x` is, for a fixed
 * (head h, head-dim row pp, state index nn), an independent scalar recurrence over t —
 * dA depends only on (t,h), and nothing couples one nn to another. So one thread per
 * (h,pp,nn) keeps `ss` in a REGISTER and walks t, and that part stays bit-exact: each
 * thread issues the CPU's operand sequence for its own element.
 *
 * What does change is `y = Σ_nn ss*C`. The CPU sums the ds products strictly in nn
 * order; here the block tree-reduces them. That reassociation is the only difference,
 * and it is worth ~1 ULP on a 128-term f32 sum.
 *
 * Why accept it. The bit-identical form (one thread per (h,pp), serial over both t and
 * nn) exposes only nh*hd = 8192 threads — 64 per block, ~2 blocks/SM once the head's
 * state is staged — and MEASURED 7.57 s against the CPU scan's ~7.50 s, i.e. no win at
 * all, on a GB10 that wants tens of thousands of threads resident. Neither shared-memory
 * staging nor padding away a 32-way bank conflict moved it, because the limit was
 * parallelism, not memory. Keeping ss per (h,pp,nn) instead exposes nh*hd*ds =
 * 1,048,576 threads and makes every global access coalesced (consecutive nn are
 * adjacent in state, B and C).
 *
 * Determinism is preserved: the tree order is fixed by block shape, so repeated runs
 * agree exactly. Correctness is gated on TOKEN identity, not bit identity.
 *
 * Launch: grid (hd, nh), block ds threads, shared = 2*ds (B/C rows) + 32 (warp partials).
 * Declines if ds exceeds the max block size — the caller then runs the CPU scan. */
/* `exact`: when nonzero, `y`'s sum over d_state is done in STRICT nn-ascending order by
 * thread 0 (bit-identical to the S==1 kernel and the CPU scan) instead of the warp/block
 * tree. This is what the MTP verify forward needs: its S>1 logits must match the S==1
 * decode path to the bit, or a near-tie argmax forks the accepted token from DRAFT=0. The
 * strict sum serializes ds adds on one thread, so it is only chosen at small seq (verify /
 * tiny prefills) where the tree's parallelism buys nothing anyway; large-S prefill keeps
 * the tree. In exact mode the third shared region holds prod[ds] (host sizes it max(ds,32)). */
__global__ static void mamba2_scan_seq_kernel(float *state, float *y, const float *hidden,
        const float *b, const float *c, const float *dt_h, const float *da_h,
        const float *d, int nh, int hd, int ds, int ng, int seq, int exact) {
    extern __shared__ float sh[];
    float *sh_b = sh;             // [ds]
    float *sh_c = sh_b + ds;      // [ds]
    float *sh_red = sh_c + ds;    // tree: [<=32] one slot per warp; exact: prod[ds]
    int pp = blockIdx.x, h = blockIdx.y, nn = threadIdx.x;
    if (h >= nh || pp >= hd || nn >= ds) return;
    int hpg = nh / ng, grp = h / hpg;
    size_t d_inner = (size_t)nh * hd;
    // Coalesced: consecutive nn are adjacent.
    size_t sidx = ((size_t)h * hd + pp) * ds + nn;
    float ss = state[sidx];
    float dh = d[h];
    int lane = nn & 31, warp = nn >> 5, nwarps = (ds + 31) >> 5;
    for (int t = 0; t < seq; t++) {
        sh_b[nn] = b[((size_t)t * ng + grp) * ds + nn];
        sh_c[nn] = c[((size_t)t * ng + grp) * ds + nn];
        __syncthreads();
        float dth = dt_h[(size_t)t * nh + h], dah = da_h[(size_t)t * nh + h];
        float x_hp = hidden[(size_t)t * d_inner + (size_t)h * hd + pp];
        // Per-element state update — identical operand order to the CPU scan.
        ss = __fadd_rn(__fmul_rn(ss, dah), __fmul_rn(__fmul_rn(dth, sh_b[nn]), x_hp));
        float prod = __fmul_rn(ss, sh_c[nn]);
        if (exact) {
            // Strict nn-ascending sum: bit-identical to the S==1 kernel / CPU scan.
            sh_red[nn] = prod;
            __syncthreads();
            if (nn == 0) {
                float acc = 0.f;
                for (int i = 0; i < ds; i++) acc = __fadd_rn(acc, sh_red[i]);
                y[(size_t)t * d_inner + (size_t)h * hd + pp] =
                    __fadd_rn(acc, __fmul_rn(x_hp, dh));
            }
        } else {
            // Tree reduction (the one reassociation) — fast path for large-S prefill.
            for (int off = 16; off > 0; off >>= 1)
                prod = __fadd_rn(prod, __shfl_down_sync(0xffffffffu, prod, off));
            if (lane == 0) sh_red[warp] = prod;
            __syncthreads();
            if (nn == 0) {
                float acc = 0.f;
                for (int i = 0; i < nwarps; i++) acc = __fadd_rn(acc, sh_red[i]);
                y[(size_t)t * d_inner + (size_t)h * hd + pp] =
                    __fadd_rn(acc, __fmul_rn(x_hp, dh));
            }
        }
        __syncthreads();   // nobody may overwrite sh_b/sh_c/sh_red before all readers finish
    }
    state[sidx] = ss;
}

/* Standard grouped-query attention prefill (MiniMax-M3): Q[S,H,D], full K/V[T,Hkv,D]
 * (no MLA absorption). One block per (query s, head h); a query head maps to KV head
 * h/(H/Hkv). Causal over [0, T-S+s]. Shared-mem softmax, mirroring the absorb batch
 * kernel's reductions. sm = qs[D] ++ scores[T] ++ red[ATTN_TPB]. */
__global__ static void gqa_attn_kernel(float *ctx, const float *Q, const float *K,
        const float *V, int S, int H, int Hkv, int D, int T, float scale) {
    int s = blockIdx.y, h = blockIdx.x, tid = threadIdx.x, nt = T - S + s + 1;
    if (s >= S || nt < 1) return;
    int kvh = h / (H / Hkv);
    extern __shared__ float sm[];
    float *qs = sm, *scores = qs + D, *red = scores + T;
    const float *qrow = Q + ((size_t)s * H + h) * D;
    for (int d = tid; d < D; d += blockDim.x) qs[d] = qrow[d];
    __syncthreads();
    for (int t = tid; t < nt; t += blockDim.x) {
        const float *kt = K + ((size_t)t * Hkv + kvh) * D;
        float a = 0; for (int d = 0; d < D; d++) a += qs[d] * kt[d];
        scores[t] = a * scale;
    }
    __syncthreads();
    float local = -3.402823466e+38F;
    for (int t = tid; t < nt; t += blockDim.x) local = fmaxf(local, scores[t]);
    red[tid] = local; __syncthreads();
    for (int n = blockDim.x >> 1; n; n >>= 1) { if (tid < n) red[tid] = fmaxf(red[tid], red[tid + n]); __syncthreads(); }
    float mx = red[0];
    local = 0; for (int t = tid; t < nt; t += blockDim.x) { float e = expf(scores[t] - mx); scores[t] = e; local += e; }
    red[tid] = local; __syncthreads();
    for (int n = blockDim.x >> 1; n; n >>= 1) { if (tid < n) red[tid] += red[tid + n]; __syncthreads(); }
    float inv = 1.f / red[0];
    __syncthreads();
    for (int d = tid; d < D; d += blockDim.x) {
        float a = 0;
        for (int t = 0; t < nt; t++) a += scores[t] * V[((size_t)t * Hkv + kvh) * D + d];
        ctx[((size_t)s * H + h) * D + d] = a * inv;
    }
}

/* ==== Tensor-core (WMMA) flash GQA prefill core (MiniMax-M3) ==================
 * The scalar gqa_attn_kernel serializes the T loop with only D-way parallelism in
 * the V accumulation, and holds all T scores in shared memory (caps T, thrashes).
 * This is a flash kernel: block = (head h, query-tile of 16 tokens), tiles over key
 * blocks of 16 with online softmax, and runs BOTH GEMMs on tensor cores — QK^T over
 * the D contraction and P@V over the 16-key tile — in fp16 with f32 accumulation.
 * Structure mirrors tc_sparse_attn, minus the MLA W_K/W_V absorption and the DSA
 * mask: standard Q·K^T with causal masking only. Q[S,H,D], K/V[T,Hkv,D] indexed by
 * kvh = h/(H/Hkv), ctx[S,H,D]. `scale` is folded into Q so Scores come out scaled.
 * Requires D % 16 == 0. Launch 256 threads (8 warps); shared = GQA_QT*8*D bytes. */
#define GQA_QT 16
__global__ static void tc_gqa_attn(float *ctx, const float *Q, const float *K,
        const float *V, int S, int H, int Hkv, int D, int T, float scale) {
#if __CUDA_ARCH__ >= 700
    using namespace nvcuda;
    int h = blockIdx.x, qt = blockIdx.y, tid = threadIdx.x, warp = tid >> 5, lane = tid & 31;
    int q0 = qt * GQA_QT, kvh = h / (H / Hkv), nwarp = blockDim.x >> 5;
    int base = T - S;                            // causal: query row r attends to keys <= base+q0+r
    extern __shared__ char smem[];
    __half *QA = (__half*)smem;                  // [GQA_QT][D] fp16 query tile (scale folded)
    __half *KB = QA + GQA_QT * D;                // [GQA_QT][D] fp16 key tile, then reused for V
    float *acc = (float*)(KB + GQA_QT * D);      // [GQA_QT][D] f32 running output
    __shared__ __half Pt[GQA_QT * GQA_QT];       // fp16 softmax-prob tile (P)
    __shared__ __half ah[256], ah8[8][256], bh8[8][256];
    __shared__ float scpart[8 * 256], sc[GQA_QT * GQA_QT], mrow[GQA_QT], lrow[GQA_QT], corr[GQA_QT];
    for (int z = tid; z < GQA_QT * D; z += blockDim.x) { int r = z / D, c = z % D; int s = q0 + r;
        QA[z] = (s < S) ? __float2half(Q[((size_t)s * H + h) * D + c] * scale) : __float2half(0.f); }
    for (int r = tid; r < GQA_QT; r += blockDim.x) { mrow[r] = -3.4e38f; lrow[r] = 0.f; }
    for (int z = tid; z < GQA_QT * D; z += blockDim.x) acc[z] = 0.f;
    __syncthreads();
    int ktmax = base + q0 + GQA_QT; if (ktmax > T) ktmax = T;     // last valid row's causal bound
    for (int kt = 0; kt < ktmax; kt += GQA_QT) {
        // Scores[16,16] = QA @ K_tile^T, split-K over D across warps.
        for (int z = tid; z < GQA_QT * D; z += blockDim.x) { int r = z / D, c = z % D; int t = kt + r;
            KB[z] = (t < T) ? __float2half(K[((size_t)t * Hkv + kvh) * D + c]) : __float2half(0.f); }
        __syncthreads();
        { __half *myah = ah8[warp], *mybh = bh8[warp];
          wmma::fragment<wmma::accumulator,16,16,16,float> accS; wmma::fill_fragment(accS, 0.f);
          for (int k0 = warp * 16; k0 < D; k0 += nwarp * 16) {
            for (int z = lane; z < 256; z += 32) { int m = z / 16, k = z % 16; myah[z] = QA[m * D + (k0 + k)]; mybh[z] = KB[m * D + (k0 + k)]; }
            __syncwarp();
            wmma::fragment<wmma::matrix_a,16,16,16,__half,wmma::row_major> af;
            wmma::fragment<wmma::matrix_b,16,16,16,__half,wmma::col_major> bf;
            wmma::load_matrix_sync(af, myah, 16); wmma::load_matrix_sync(bf, mybh, 16);
            wmma::mma_sync(accS, af, bf, accS); __syncwarp(); }
          wmma::store_matrix_sync(&scpart[warp * 256], accS, 16, wmma::mem_row_major); }
        __syncthreads();
        for (int z = tid; z < GQA_QT * GQA_QT; z += blockDim.x) { float a = 0; for (int wr = 0; wr < nwarp; wr++) a += scpart[wr * 256 + z]; sc[z] = a; }
        __syncthreads();
        // Causal mask + online softmax, one warp per query row.
        for (int r = warp; r < GQA_QT; r += nwarp) { int s = q0 + r; int pos = base + s;
            float tmax = -3.4e38f;
            for (int c = lane; c < GQA_QT; c += 32) { int t = kt + c;
                int keep = (s < S && t < T && t <= pos);
                float v = keep ? sc[r * GQA_QT + c] : -3.4e38f; sc[r * GQA_QT + c] = v; tmax = fmaxf(tmax, v); }
            for (int o = 16; o; o >>= 1) tmax = fmaxf(tmax, __shfl_down_sync(0xffffffff, tmax, o));
            tmax = __shfl_sync(0xffffffff, tmax, 0);
            float mold = mrow[r], mnew = fmaxf(mold, tmax), cr = expf(mold - mnew), lsum = 0.f;
            for (int c = lane; c < GQA_QT; c += 32) { float e = (sc[r * GQA_QT + c] > -1e30f) ? expf(sc[r * GQA_QT + c] - mnew) : 0.f; Pt[r * GQA_QT + c] = __float2half(e); lsum += e; }
            for (int o = 16; o; o >>= 1) lsum += __shfl_down_sync(0xffffffff, lsum, o);
            lsum = __shfl_sync(0xffffffff, lsum, 0);
            if (lane == 0) { mrow[r] = mnew; corr[r] = cr; lrow[r] = lrow[r] * cr + lsum; } }
        __syncthreads();
        // acc = acc*corr + P @ V_tile. Reload KB with V (K no longer needed).
        for (int z = tid; z < GQA_QT * D; z += blockDim.x) { int r = z / D; acc[z] *= corr[r]; }
        for (int z = tid; z < 256; z += blockDim.x) ah[z] = Pt[z];
        for (int z = tid; z < GQA_QT * D; z += blockDim.x) { int r = z / D, c = z % D; int t = kt + r;
            KB[z] = (t < T) ? __float2half(V[((size_t)t * Hkv + kvh) * D + c]) : __float2half(0.f); }
        __syncthreads();
        { __half *mybh = bh8[warp];
          for (int dn = warp * 16; dn < D; dn += nwarp * 16) {
            wmma::fragment<wmma::accumulator,16,16,16,float> accP;
            wmma::load_matrix_sync(accP, &acc[dn], D, wmma::mem_row_major);
            for (int z = lane; z < 256; z += 32) { int n = z / 16, key = z % 16; mybh[z] = KB[key * D + (dn + n)]; }
            __syncwarp();
            wmma::fragment<wmma::matrix_a,16,16,16,__half,wmma::row_major> af;
            wmma::fragment<wmma::matrix_b,16,16,16,__half,wmma::col_major> bf;
            wmma::load_matrix_sync(af, ah, 16); wmma::load_matrix_sync(bf, mybh, 16);
            wmma::mma_sync(accP, af, bf, accP);
            wmma::store_matrix_sync(&acc[dn], accP, D, wmma::mem_row_major); __syncwarp(); } }
        __syncthreads();
    }
    // ctx[s,h,:] = acc[r,:] / l[r]
    for (int r = 0; r < GQA_QT; r++) { int s = q0 + r; if (s >= S) continue; float inv = 1.f / lrow[r];
        for (int d = tid; d < D; d += blockDim.x) ctx[((size_t)s * H + h) * D + d] = acc[r * D + d] * inv;
        __syncthreads(); }
#endif
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

/* Reallocation counters. A cudaMallocHost of ~176 MB costs ~100 ms, so if the grouped
 * path re-reserves per chunk that alone is ~20 s over a prefill — the right order for the
 * pack's unexplained time, given a standalone benchmark of the same copy runs at 64 GB/s
 * while the pack manages 6. Counted rather than assumed: three mechanism guesses have
 * already been wrong here. */
long long g_res_pin_hit=0,g_res_pin_alloc=0,g_res_dev_hit=0,g_res_dev_alloc=0;

static int reserve_pinned(float **ptr,size_t *cap,size_t bytes){
    if(*cap>=bytes){g_res_pin_hit++;return 1;}
    g_res_pin_alloc++;
    if(*ptr)cudaFreeHost(*ptr);*ptr=nullptr;*cap=0;
    if(!cuda_ok(cudaMallocHost(ptr,bytes),"pinned staging allocation"))return 0;*cap=bytes;return 1;
}

/* Serialises init/shutdown against each other. NOT the same lock as `g_scratch_mu`: that
 * one protects a context's scratch while it is in use, this one protects the existence of
 * the contexts themselves. */
static std::mutex g_init_mu;

/* Idempotent by contract, and it has to be.
 *
 * The Rust side probes CUDA through a **thread_local** `OnceCell` (`gpu::available`), so
 * every thread that first touches the GPU calls this — while `g_nctx`, `g_ctx` and the
 * per-device stream are process-global. Re-running the body would set `g_nctx = 0`, wipe a
 * live `DeviceContext` with `*ctx = {}` and create a fresh stream, all underneath threads
 * already launching kernels on the old one. That is the `invalid resource handle` the test
 * suite has been hitting: deterministic with `--test-threads=1` (one init), flaky in
 * parallel (one init per thread), with the victim test varying run to run.
 *
 * So: initialise once, and afterwards confirm the request matches rather than rebuild. A
 * DIFFERENT device set is refused instead of honoured — tearing down contexts other threads
 * hold is exactly the bug, and no caller in this tree asks for it (`coli` initialises once
 * with the cluster's list; re-init after `coli_cuda_shutdown` works, since that resets
 * `g_nctx` to 0). */
extern "C" int coli_cuda_init(const int *devices, int count) {
    int available = 0;
    if (!devices || count < 1 || count > COLI_CUDA_MAX_DEVICES) return 0;
    std::lock_guard<std::mutex> _init_lk(g_init_mu);
    if (g_nctx > 0) {
        if (g_nctx == count) {
            int same = 1;
            for (int i = 0; i < count; i++) if (g_ctx[i].device != devices[i]) { same = 0; break; }
            if (same) return 1; // already up, with exactly these devices
        }
        std::fprintf(stderr,
                     "[CUDA] init called with a different device set while %d context(s) are "
                     "live; refusing to tear them down\n", g_nctx);
        return 0;
    }
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

/* Takes `g_init_mu` for the same reason `coli_cuda_init` does: it destroys the streams and
 * frees the scratch that other threads may be about to use, so it must not interleave with
 * an init. (It does NOT take `g_scratch_mu` — a shutdown racing a live kernel launch is a
 * caller error this cannot paper over, and holding both here would invert the lock order.) */
extern "C" void coli_cuda_shutdown(void) {
    std::lock_guard<std::mutex> _init_lk(g_init_mu);
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
        if(ctx->ms_state)cudaFree(ctx->ms_state);if(ctx->ms_x)cudaFree(ctx->ms_x);if(ctx->ms_y)cudaFree(ctx->ms_y);
        if(ctx->ms_b)cudaFree(ctx->ms_b);if(ctx->ms_c)cudaFree(ctx->ms_c);
        if(ctx->ms_dth)cudaFree(ctx->ms_dth);if(ctx->ms_dah)cudaFree(ctx->ms_dah);if(ctx->ms_d)cudaFree(ctx->ms_d);
        if(ctx->ms_pin_x)cudaFreeHost(ctx->ms_pin_x);if(ctx->ms_pin_y)cudaFreeHost(ctx->ms_pin_y);
        if(ctx->ms_pin_state)cudaFreeHost(ctx->ms_pin_state);
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
        ctx->ebsg=ctx->ebsu=ctx->ebsd=nullptr;ctx->ebsg_cap=ctx->ebsu_cap=ctx->ebsd_cap=0;
        ctx->lres=nullptr;ctx->lres_cap=0;ctx->host_lres=nullptr;ctx->host_lres_cap=0;
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

/* Page-lock a host allocation the engine already owns, so the GPU can DMA straight out of
 * it. This is what lets the expert staging path skip its CPU pack: a copy out of PAGEABLE
 * memory is bounced through the driver's own staging buffer (measured at 1-2 GB/s, and the
 * reason COLI_GROUP_DIRECT came out 2.3x slower), while a copy out of registered memory is
 * a straight DMA at the 56 GB/s this box actually does.
 *
 * Registration is per allocation, not per use — the expert buffer pool recycles a bounded
 * set of allocations, so the cost is paid once each and amortised over every later expert
 * that lands in that slot.
 *
 * Returns 0 on failure and CLEARS the error, which matters more than it looks: a sticky
 * CUDA error turns the next unrelated launch into a silent CPU fallback. Failing to pin is
 * allowed — the caller keeps the pack. Failing quietly and poisoning the context is not. */
extern "C" int coli_cuda_host_register(void *p, size_t bytes) {
    if (!p || !bytes) return 0;
    cudaError_t e = cudaHostRegister(p, bytes, cudaHostRegisterDefault);
    if (e != cudaSuccess) {
        /* Report the first failure and the first one after any success. 10266 of 10283
         * registrations failed on GLM after ~0.2 GB had been locked, and a bare pass/fail
         * counter cannot tell a hard driver limit from a wrong flag — one is a reason to
         * delete this path, the other is a one-word fix. */
        static int s_reported = 0;
        if (!s_reported) {
            s_reported = 1;
            fprintf(stderr, "[page-lock] cudaHostRegister(%p, %zu) failed: %s (%d)\n",
                    p, bytes, cudaGetErrorString(e), (int)e);
        }
        cudaGetLastError();
        return 0;
    }
    return 1;
}

extern "C" void coli_cuda_host_unregister(void *p) {
    if (!p) return;
    if (cudaHostUnregister(p) != cudaSuccess) cudaGetLastError();
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

/* Every live DeviceContext scratch allocation, device and pinned-host alike.
 *
 * On GB10 that distinction does not matter for the RAM ledger: "VRAM" and host memory are
 * the SAME 121 GiB LPDDR5X pool, so a cudaMalloc here is exactly as real as a heap
 * allocation in Rust. None of it was visible to the ledger — `Class::Scratch` is charged
 * in exactly one place (gpu.rs), as a PREDICTION, and only on the grouped NVFP4 path, so
 * an MXFP4 model like Kimi-K3 charges ~nothing while allocating all of the below.
 *
 * `tensor_bytes` is deliberately EXCLUDED: that is the device weight cache, a different
 * class, already exposed by coli_cuda_stats.
 *
 * Sums capacities, not requested sizes. `reserve` keeps a buffer when cap >= bytes, so the
 * capacity is what is actually held. */
extern "C" size_t coli_cuda_scratch_bytes(int device) {
    size_t t = 0;
    for (int i = 0; i < g_nctx; i++) if (device < 0 || g_ctx[i].device == device) {
        const DeviceContext *c = &g_ctx[i];
        t += c->x_cap + c->y_cap + c->gate_cap + c->up_cap;
        t += c->qx_cap + c->qscale_cap;
        t += c->host_x_cap + c->host_y_cap;                       /* pinned */
        t += c->sg_tile_e_cap + c->sg_tile_r0_cap + c->sg_off_cap + c->sg_rows_cap;
        t += c->sg_ug_cap + c->sg_dg_cap + c->sg_uw_cap + c->sg_ubs_cap + c->sg_dw_cap + c->sg_dbs_cap;
        t += c->ewg_cap + c->ewu_cap + c->ewd_cap;
        t += c->esg_cap + c->esu_cap + c->esd_cap;
        t += c->ebsg_cap + c->ebsu_cap + c->ebsd_cap;
        t += c->lres_cap + c->host_lres_cap;                      /* host_lres is pinned */
        t += c->aq_cap + c->al_cap + c->ar_cap + c->ac_cap;
        t += c->ms_state_cap + c->ms_x_cap + c->ms_y_cap + c->ms_b_cap + c->ms_c_cap;
        t += c->ms_dth_cap + c->ms_dah_cap + c->ms_d_cap;
        t += c->ms_pin_x_cap + c->ms_pin_y_cap + c->ms_pin_state_cap;   /* pinned */
        t += c->asel_cap + c->acnt_cap;
        t += c->aqa_cap + c->akb_cap + c->amsk_cap;
        t += c->group_desc_cap;
        for (int b = 0; b < 24; b++) t += c->pipe_cap[b];
    }
    return t;
}

extern "C" void coli_cuda_group_stats(uint64_t *calls, uint64_t *experts, uint64_t *rows,
                                        double *h2d_ms, double *kernel_ms, double *d2h_ms) {
    if(calls) *calls=g_group_calls; if(experts) *experts=g_group_experts; if(rows) *rows=g_group_rows;
    if(h2d_ms) *h2d_ms=g_group_h2d_ms; if(kernel_ms) *kernel_ms=g_group_kernel_ms;
    if(d2h_ms) *d2h_ms=g_group_d2h_ms;
}

/* Resident-weight residency mode (see coli_cuda_set_weight_zerocopy). 0 = upload a
 * device copy; 1 = wrap the host buffer and read it in place. */
static int g_weight_zerocopy = 0;

/* Choose how coli_cuda_tensor_upload materializes a resident weight.
 *
 * Uploading gives the kernel device memory (~273 GB/s) at the cost of a SECOND copy of
 * every resident weight — and on GB10 "VRAM" is the same physical RAM, so that copy is
 * charged against the expert cache's budget. Wrapping reads the host buffer in place
 * (~51 GB/s, the path the streamed experts already take) for free.
 *
 * Kimi-K3 forces the choice: ~63 GB resident of a 121 GB box, so uploading leaves nothing
 * for the expert cache and the process is OOM-killed mid-forward. Wrapping is slower per
 * access but keeps the matmuls ON the GPU, and falling off the GPU entirely measured
 * 6.8x slower on nemotron decode. */
extern "C" void coli_cuda_set_weight_zerocopy(int on) { g_weight_zerocopy = on ? 1 : 0; }

extern "C" int coli_cuda_tensor_upload(ColiCudaTensor **tensor,
                                        const void *weights, const float *scales,
                                        int fmt, int I, int O, int device) {
    DeviceContext *ctx = find_ctx(device);
    if (!tensor || !weights || I < 1 || O < 1 || !select_ctx(ctx)) return 0;
    /* Zero-copy: hand back a wrapped tensor pointing at the host buffers. Every kernel
     * below reads `t->weights` the same way either path; only the pointer's provenance
     * differs, and pageable access makes a host pointer legal. */
    if (g_weight_zerocopy && !*tensor)
        return coli_cuda_tensor_wrap(tensor, weights, scales, fmt, I, O, device);
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
    std::lock_guard<std::mutex> _scratch_lk(scratch_mu(ctx));
    size_t rb = row_bytes(fmt, I);
    size_t xb = (size_t)S * I * sizeof(float), yb = (size_t)S * O * sizeof(float);
    if (!reserve(&ctx->x, &ctx->x_cap, xb) || !reserve(&ctx->y, &ctx->y_cap, yb)) return 0;
    if (!cuda_ok(cudaMemcpy(ctx->x, x, xb, cudaMemcpyHostToDevice), "input upload")) return 0;
    // Tiled tensor-core path for the resident matmuls (attention q/kv/o/kv_b proj):
    // reads each weight once per 16-row tile vs quant_matmul's S-fold re-read.
    const char *tile_env = getenv("COLI_TILE_I8");
    int tile = (!tile_env || strcmp(tile_env, "0") != 0) && ctx->compute_major >= 7;
    // ...but ONLY when there are rows to fill the tile. At S==1 (decode) 15/16 of the
    // MMA is padding and the grid collapses to (O/64, 1) — e.g. 290 blocks for
    // Nemotron's [18560,4096] mamba in_proj, far too few to saturate memory.
    // quant_matmul launches one block per output row instead, and its S-fold weight
    // re-read costs nothing when S is 1. This is the same principle the expert FFN
    // path already applies (fp8a16_gemv / nvfp4_gemv, "waste 15/16 of their MMA at
    // S==1"); it was simply never applied to the dense resident matmuls, which is
    // where Nemotron-H spends most of decode (mamba in/out-proj, shared expert,
    // fc1/fc2). MEASURED on Nemotron-H decode: 4.65 -> 7.0 tok/s (1.50x), tokens
    // byte-identical; prefill unaffected since it runs S>>16. `COLI_TILE_I8=force`
    // keeps tiles on at S==1 for A/B.
    if (S == 1 && !(tile_env && strcmp(tile_env, "force") == 0)) tile = 0;
    // Decode (S==1): use the purpose-built one-warp-per-column GEMV rather than falling
    // back to the generic `quant_matmul`. Needs x in shared memory, so it is limited to
    // weights whose K fits the 48 KB default dynamic-shared cap (I<=12288 floats); the
    // wider ones fall through to quant_matmul as before. COLI_I8_GEMV=0 disables for A/B.
    static int s_i8gemv = -1;
    if (s_i8gemv < 0) { const char *e = getenv("COLI_I8_GEMV"); s_i8gemv = !e || strcmp(e, "0") != 0; }
    size_t gemv_shmem = (size_t)I * sizeof(float);
    if (s_i8gemv && S == 1 && (fmt == 1 || fmt == 4) && ctx->compute_major >= 7 &&
        gemv_shmem <= 48u * 1024u) {
        const int tpb = 128, wpb = tpb / 32;
        unsigned blocks = (unsigned)((O + wpb - 1) / wpb);
        if (fmt == 1)
            i8a16_gemv<<<blocks, tpb, gemv_shmem>>>(ctx->y, ctx->x,
                (const uint8_t *)t->weights, t->scales, I, O);
        else
            fp8a16_gemv<<<blocks, tpb, gemv_shmem>>>(ctx->y, ctx->x,
                (const uint8_t *)t->weights, t->scales, I, O);
    } else if (tile && (fmt == 1 || fmt == 4)) {
        dim3 tg((unsigned)((O + 63) / 64), (unsigned)((S + 15) / 16));
        if (fmt == 4)
            fp8a16_matmul<<<tg, 128>>>(ctx->y, ctx->x, (const uint8_t *)t->weights, t->scales, S, I, O);
        else
            i8a16_matmul<<<tg, 128>>>(ctx->y, ctx->x, (const uint8_t *)t->weights, t->scales, S, I, O);
    } else {
        // gridDim.y is capped at 65535 and this arm launches one block per (output, row), so
        // a prefill with more rows than that does not merely run slowly — the launch fails
        // outright with `invalid argument`, coli_cuda_matmul returns 0, and the caller drops
        // to the single-threaded CPU `matmul_qt`. Measured on M2.7 at S=73728 (#56): the
        // attention core stayed on the GPU while every projection fell to one CPU thread.
        // (The tiled fmt 1/4 arm above is already safe — its grid.y is (S+15)/16.)
        //
        // quant_matmul takes its row from blockIdx.y alone and never reads `S`, so a chunk is
        // just a pointer offset. Chunks share the default stream, so they serialise in order
        // and the single completion check below still covers all of them.
        const unsigned YMAX = 65535u;
        for (int s0 = 0; s0 < S; s0 += (int)YMAX) {
            int sc = S - s0;
            if (sc > (int)YMAX) sc = (int)YMAX;
            dim3 grid((unsigned)O, (unsigned)sc);
            quant_matmul<<<grid, 256>>>(ctx->y + (size_t)s0 * O, ctx->x + (size_t)s0 * I,
                                        t->weights, t->scales, fmt, sc, I, O, rb, t->wrapped);
        }
    }
    if (!cuda_ok(cudaGetLastError(), "matmul launch") ||
        !cuda_ok(cudaMemcpy(y, ctx->y, yb, cudaMemcpyDeviceToHost), "output download")) return 0;
    return 1;
}

/* Defined further down (with the other tensor-wrap entry points); forward-declared here
 * because this translation unit is compiled as one pass and the matmul below calls it. */
extern "C" int coli_cuda_tensor_wrap_nvfp4(ColiCudaTensor **tensor,
        const void *weights, const void *bscale, float gscale,
        int I, int O, int device);

/* Resident NVFP4 matmul: y[S,O] = x[S,I] @ W[O,I]^T with W in NVFP4.
 *
 * The DEVICE kernels for this already existed (`nvfp4_gemv` / `nvfp4_matmul`) — they are
 * fully general `y[M,N] = x[M,K] @ W[N,K]^T` and were only ever reached through the
 * expert-FFN wrappers, which fuse gate/up/down. This is the single-weight entry point the
 * resident path needs, mirroring `coli_cuda_matmul`.
 *
 * Why it can't go through `coli_cuda_matmul`: that uploads a dense `O*I` buffer, but NVFP4
 * stores ~half a byte per weight plus block scales, so the dense upload would read far past
 * the end. The weight is wrapped ZERO-COPY instead (host pointers, unified memory) — the
 * same choice the expert path makes, and the one GB10 wants.
 *
 * S==1 takes the one-warp-per-column GEMV; S>1 takes the WMMA tile. Same split, and for the
 * same reason, as the int8/e4m3 path: at S==1 a 16-row MMA tile wastes 15/16 of its work. */
extern "C" int coli_cuda_matmul_nvfp4(ColiCudaTensor **tensor,
                                       float *y, const float *x,
                                       const void *weights, const void *bscale,
                                       float gscale, int S, int I, int O, int device) {
    if (S < 1 || !y || !x ||
        !coli_cuda_tensor_wrap_nvfp4(tensor, weights, bscale, gscale, I, O, device)) return 0;
    ColiCudaTensor *t = *tensor;
    DeviceContext *ctx = find_ctx(t->device);
    if (!select_ctx(ctx)) return 0;
    std::lock_guard<std::mutex> _scratch_lk(scratch_mu(ctx));
    size_t xb = (size_t)S * I * sizeof(float), yb = (size_t)S * O * sizeof(float);
    if (!reserve(&ctx->x, &ctx->x_cap, xb) || !reserve(&ctx->y, &ctx->y_cap, yb)) return 0;
    if (!cuda_ok(cudaMemcpy(ctx->x, x, xb, cudaMemcpyHostToDevice), "nvfp4 matmul input")) return 0;
    const uint8_t *w = (const uint8_t *)t->weights;
    const uint8_t *bs = (const uint8_t *)t->bscale;
    if (S == 1) {
        nvfp4_gemv_dispatch(ctx->y, ctx->x, w, bs, t->gscale, I, O, ctx->stream);
    } else {
        dim3 grid((unsigned)((O + 63) / 64), (unsigned)((S + 15) / 16));
        nvfp4_matmul<<<grid, 128, 0, ctx->stream>>>(ctx->y, ctx->x, w, bs, t->gscale, S, I, O);
    }
    if (!cuda_ok(cudaGetLastError(), "nvfp4 matmul launch") ||
        !cuda_ok(cudaMemcpyAsync(y, ctx->y, yb, cudaMemcpyDeviceToHost, ctx->stream),
                 "nvfp4 matmul output") ||
        !cuda_ok(cudaStreamSynchronize(ctx->stream), "nvfp4 matmul sync")) return 0;
    return 1;
}

extern "C" int coli_cuda_expert_mlp(ColiCudaTensor *gate, ColiCudaTensor *up,
                                      ColiCudaTensor *down, float *y,
                                      const float *x, int S) {
    if (!gate || !up || !down || !x || !y || S < 1 ||
        gate->device != up->device || gate->device != down->device ||
        gate->I != up->I || gate->O != up->O ||
        down->I != gate->O || down->O != gate->I) return 0;
    /* FORMAT GUARD. This path reads scales as a per-ROW f32 array (`quant_matmul` ->
     * `weight_at`), which is only true for fmt 0/1/3/4. NVFP4 (5) and MXFP4 (6) keep
     * BLOCK scales in a separate array this entry point does not even take a parameter
     * for, so launching for them dereferences a pointer that is empty or wrongly strided.
     *
     * It did: DeepSeek-V4's MXFP4 experts reach here and the kernel takes an illegal
     * memory access. CUDA errors are STICKY, so that one launch poisons the context and
     * every later GPU op in the process fails — the model silently finished on the
     * single-threaded CPU matmul at 0.19 tok/s. Declining instead makes the caller fall
     * back cleanly for these experts and leaves the rest of the model on the GPU.
     *
     * Dims and device were checked; the format was not. That asymmetry is the bug — see
     * the closed-set dispatch trap: most per-format enumerations here fail SILENTLY. */
    if (gate->fmt > 4 || up->fmt > 4 || down->fmt > 4) return 0;
    DeviceContext *ctx = find_ctx(gate->device);
    if (!select_ctx(ctx)) return 0;
    std::lock_guard<std::mutex> _scratch_lk(scratch_mu(ctx));
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
    act_mul(ctx->gate,ctx->up,n,0);
    quant_matmul<<<output_grid,256>>>(ctx->y,ctx->gate,down->weights,down->scales,
        down->fmt,S,I,D,row_bytes(down->fmt,I),down->wrapped);
    if (!cuda_ok(cudaGetLastError(),"expert MLP launch") ||
        !cuda_ok(cudaMemcpy(y,ctx->y,yb,cudaMemcpyDeviceToHost),"expert output download")) return 0;
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
    std::lock_guard<std::mutex> _scratch_lk(scratch_mu(ctx));
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
    // kernels on device pointers. The old `S >= 16` gate assumed small S could not amortize
    // the copy; measured on the NVFP4 twin it is the opposite — a routed expert sees
    // S = tokens*top_k/n_experts (~4), and staging won 1.24x there. Same reasoning applies
    // here: the penalty is per dirty weight line, so it scales with the WEIGHT, not with S,
    // and S only decides how much useful work amortizes it. COLI_FFN_DEVCOPY=0 forces the
    // old zero-copy path. (Not re-measured on the fp8 path — see the NVFP4 twin.)
    const uint8_t *gw=(const uint8_t*)gate->weights,*uw=(const uint8_t*)up->weights,*dw=(const uint8_t*)down->weights;
    const float *gsc=gate->scales,*usc=up->scales,*dsc=down->scales;
    // Same per-PATH gate as the NVFP4 twins: the S==1 gemv reads each weight once, so
    // staging it is a pure loss. See the relu2 path for the 11.04 -> 9.8 tok/s this cost
    // on Nemotron decode when the replacement for `S >= 16` was left out.
    static int s_gemv_pre=-1;
    if(s_gemv_pre<0){const char*e=getenv("COLI_FFN_GEMV");s_gemv_pre=e&&atoi(e);}
    const bool row_path=(s_gemv_pre&&S==1);
    static int s_dc=-1;
    if(s_dc<0){const char*e=getenv("COLI_FFN_DEVCOPY");s_dc=(!e||atoi(e));}
    if(s_dc&&!row_path){
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
    // Single-row decode GEMV path (COLI_FFN_GEMV=1). The tiled WMMA kernels waste
    // 15/16 of their MMA at S==1; the GEMV streams the weight coalesced with the whole
    // block. Restricted to S==1 (decode) — for S>1 the tiled path amortizes the weight
    // read across rows and wins.
    static int s_gemv=-1;
    if(s_gemv<0){const char*e=getenv("COLI_FFN_GEMV");s_gemv=e&&atoi(e);}
    if(s_gemv&&S==1){
        int tpb=256,wpb=tpb>>5;
        fp8a16_gemv<<<(unsigned)((I+wpb-1)/wpb),tpb,(size_t)D*sizeof(float),ctx->stream>>>(ctx->gate,ctx->x,gw,gsc,D,I);
        fp8a16_gemv<<<(unsigned)((I+wpb-1)/wpb),tpb,(size_t)D*sizeof(float),ctx->stream>>>(ctx->up,ctx->x,uw,usc,D,I);
        act_mul(ctx->gate,ctx->up,(size_t)I,ctx->stream);
        fp8a16_gemv<<<(unsigned)((D+wpb-1)/wpb),tpb,(size_t)I*sizeof(float),ctx->stream>>>(ctx->y,ctx->gate,dw,dsc,I,D);
    }else{
        fp8a16_gate_up<<<hidden,256,0,ctx->stream>>>(ctx->gate,ctx->up,ctx->x,gw,uw,gsc,usc,S,D,I);
        act_mul(ctx->gate,ctx->up,(size_t)S*I,ctx->stream);
        fp8a16_matmul<<<output,128,0,ctx->stream>>>(ctx->y,ctx->gate,dw,dsc,S,I,D);
    }
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

/* NVFP4 fused expert FFN — GEMV at S==1 (decode; ~half the weight bytes of fp8), tiled
 * WMMA at S>1 (prefill). Requires fmt==5 on all three projections and compute>=7. Zero-copy
 * only (no device-copy staging path); the block-scale + global travel on the ColiCudaTensor.
 * `COLI_NVFP4_TILED=1` forces the tiled path even at S==1 (A/B against the GEMV). */
/* Kimi-K3 MXFP4 expert FFN: y = down( situ(gate.x, up.x) ).
 *
 * Same shape as coli_cuda_expert_mlp_nvfp4 with the MXFP4 decode (E8M0 per 32) and the
 * situ activation instead of SwiGLU. It exists because K3 could not use the GPU at all:
 * `ffn` deliberately declines the fused SwiGLU path when situ is set (those kernels apply
 * oai-or-SiLU and would return success having computed a DIFFERENT activation), so K3's
 * experts ran the scalar CPU loop — 85% of a measured 5-minute forward pass.
 *
 * `beta`/`linear_beta` come from the model config; `linear_beta <= 0` means unset. */
extern "C" int coli_cuda_expert_mlp_mxfp4_situ(ColiCudaTensor *gate,ColiCudaTensor *up,
        ColiCudaTensor *down,float *y,const float *x,int S,
        float beta,float linear_beta){
    if(!gate||!up||!down||!x||!y||S<1||gate->fmt!=6||up->fmt!=6||down->fmt!=6||
       gate->device!=up->device||gate->device!=down->device||gate->I!=up->I||
       gate->O!=up->O||down->I!=gate->O||down->O!=gate->I)return 0;
    DeviceContext *ctx=find_ctx(gate->device);if(!select_ctx(ctx)||ctx->compute_major<7)return 0;
    std::lock_guard<std::mutex> _scratch_lk(scratch_mu(ctx));
    int D=gate->I,I=gate->O;size_t xb=(size_t)S*D*sizeof(float),ib=(size_t)S*I*sizeof(float);
    if(!reserve(&ctx->x,&ctx->x_cap,xb)||!reserve(&ctx->gate,&ctx->gate_cap,ib)||
       !reserve(&ctx->up,&ctx->up_cap,ib)||!reserve(&ctx->y,&ctx->y_cap,xb)||
       !reserve_pinned(&ctx->host_x,&ctx->host_x_cap,xb)||
       !reserve_pinned(&ctx->host_y,&ctx->host_y_cap,xb))return 0;
    std::memcpy(ctx->host_x,x,xb);
    if(!cuda_ok(cudaMemcpyAsync(ctx->x,ctx->host_x,xb,cudaMemcpyHostToDevice,ctx->stream),
                               "expert mxfp4 input upload"))return 0;
    const uint8_t *gw=(const uint8_t*)gate->weights,*uw=(const uint8_t*)up->weights,*dw=(const uint8_t*)down->weights;
    const uint8_t *gbs=(const uint8_t*)gate->bscale,*ubs=(const uint8_t*)up->bscale,*dbs=(const uint8_t*)down->bscale;
    float gg=gate->gscale,ug=up->gscale,dg=down->gscale;
    unsigned sblocks=(unsigned)(((size_t)S*I+255)/256);
    if(S==1){
        // Same dispatcher as the plain MXFP4 expert — the situ variant differs only in the
        // activation between the two GEMMs, never in how the weight is read.
        mxfp4_gemv_dispatch(ctx->gate,ctx->x,gw,gbs,gg,D,I,ctx->stream);
        mxfp4_gemv_dispatch(ctx->up,  ctx->x,uw,ubs,ug,D,I,ctx->stream);
        situ_mul<<<sblocks,256,0,ctx->stream>>>(ctx->gate,ctx->up,(size_t)I,beta,linear_beta);
        mxfp4_gemv_dispatch(ctx->y,ctx->gate,dw,dbs,dg,I,D,ctx->stream);
    }else{
        bool did_ws=false;
        if(mxfp4_wsmm_launch(ctx->gate,ctx->x,gw,gbs,gg,S,D,I,ctx->stream)){
            mxfp4_wsmm_launch(ctx->up,ctx->x,uw,ubs,ug,S,D,I,ctx->stream);
            situ_mul<<<sblocks,256,0,ctx->stream>>>(ctx->gate,ctx->up,(size_t)S*I,beta,linear_beta);
            did_ws=mxfp4_wsmm_launch(ctx->y,ctx->gate,dw,dbs,dg,S,I,D,ctx->stream);
        }
        if(!did_ws){
            dim3 hidden((unsigned)((I+63)/64),(unsigned)((S+15)/16));
            dim3 output((unsigned)((D+63)/64),(unsigned)((S+15)/16));
            mxfp4_gate_up<<<hidden,256,0,ctx->stream>>>(ctx->gate,ctx->up,ctx->x,gw,uw,gbs,ubs,gg,ug,S,D,I);
            situ_mul<<<sblocks,256,0,ctx->stream>>>(ctx->gate,ctx->up,(size_t)S*I,beta,linear_beta);
            mxfp4_matmul<<<output,128,0,ctx->stream>>>(ctx->y,ctx->gate,dw,dbs,dg,S,I,D);
        }
    }
    if(!cuda_ok(cudaGetLastError(),"expert mxfp4 launch")||
       !cuda_ok(cudaMemcpyAsync(ctx->host_y,ctx->y,xb,cudaMemcpyDeviceToHost,ctx->stream),
                               "expert mxfp4 output download")||
       !cuda_ok(cudaStreamSynchronize(ctx->stream),"expert mxfp4 synchronize"))return 0;
    std::memcpy(y,ctx->host_y,xb);
    return 1;
}

/* MXFP4 experts with the engine's configured SwiGLU, the sibling of the `_situ` entry
 * above. DeepSeek-V4's experts are MXFP4 but gated SwiGLU, not K3's situ, so neither the
 * situ path (wrong activation) nor `coli_cuda_expert_mlp` (reads BLOCK scales as per-row
 * f32 — that combination took an illegal memory access) could serve them, and every one
 * of V4's 258 routed experts per token fell to the scalar CPU loop.
 *
 * `act_mul` applies whatever activation the engine has set, so this computes exactly what
 * the CPU path computes today. In particular it does NOT yet apply V4's `swiglu_limit`
 * clamp — that gap is real and tracked, and this change neither closes nor widens it.
 * Making the GPU silently apply a DIFFERENT activation from the CPU would be far worse
 * than sharing a known-incomplete one. */
extern "C" int coli_cuda_expert_mlp_mxfp4(ColiCudaTensor *gate,ColiCudaTensor *up,
        ColiCudaTensor *down,float *y,const float *x,int S){
    if(!gate||!up||!down||!x||!y||S<1||gate->fmt!=6||up->fmt!=6||down->fmt!=6||
       gate->device!=up->device||gate->device!=down->device||gate->I!=up->I||
       gate->O!=up->O||down->I!=gate->O||down->O!=gate->I)return 0;
    DeviceContext *ctx=find_ctx(gate->device);if(!select_ctx(ctx)||ctx->compute_major<7)return 0;
    std::lock_guard<std::mutex> _scratch_lk(scratch_mu(ctx));
    int D=gate->I,I=gate->O;size_t xb=(size_t)S*D*sizeof(float),ib=(size_t)S*I*sizeof(float);
    if(!reserve(&ctx->x,&ctx->x_cap,xb)||!reserve(&ctx->gate,&ctx->gate_cap,ib)||
       !reserve(&ctx->up,&ctx->up_cap,ib)||!reserve(&ctx->y,&ctx->y_cap,xb)||
       !reserve_pinned(&ctx->host_x,&ctx->host_x_cap,xb)||
       !reserve_pinned(&ctx->host_y,&ctx->host_y_cap,xb))return 0;
    std::memcpy(ctx->host_x,x,xb);
    if(!cuda_ok(cudaMemcpyAsync(ctx->x,ctx->host_x,xb,cudaMemcpyHostToDevice,ctx->stream),
                               "expert mxfp4 input upload"))return 0;
    const uint8_t *gw=(const uint8_t*)gate->weights,*uw=(const uint8_t*)up->weights,*dw=(const uint8_t*)down->weights;
    const uint8_t *gbs=(const uint8_t*)gate->bscale,*ubs=(const uint8_t*)up->bscale,*dbs=(const uint8_t*)down->bscale;
    float gg=gate->gscale,ug=up->gscale,dg=down->gscale;
        if(S==1){
        // Every MXFP4 decode GEMV goes through the dispatcher — see its comment for why
        // the narrow kernel these call sites used to launch was leaving half the read
        // width on the table.
        mxfp4_gemv_dispatch(ctx->gate,ctx->x,gw,gbs,gg,D,I,ctx->stream);
        mxfp4_gemv_dispatch(ctx->up,  ctx->x,uw,ubs,ug,D,I,ctx->stream);
        act_mul(ctx->gate,ctx->up,(size_t)I,ctx->stream);
        mxfp4_gemv_dispatch(ctx->y,ctx->gate,dw,dbs,dg,I,D,ctx->stream);
    }else{
        // Weight-stationary small-M path (the #90 fix, ported to fmt 6). A routed expert
        // at prefill sees S = tokens*top_k/n_experts rows — 12 on V4 at 512 tokens — where
        // the 16x16 WMMA tile below re-dequantizes the whole weight per m-tile. Declines
        // above S=32, which is where the MMA amortizes and WMMA is the better kernel; the
        // shared expert (S = all tokens) therefore still takes WMMA, as it should.
        bool did_ws=false;
        if(mxfp4_wsmm_launch(ctx->gate,ctx->x,gw,gbs,gg,S,D,I,ctx->stream)){
            // Same S ⇒ the up/down launches take the same MT bucket and cannot decline.
            mxfp4_wsmm_launch(ctx->up,ctx->x,uw,ubs,ug,S,D,I,ctx->stream);
            act_mul(ctx->gate,ctx->up,(size_t)S*I,ctx->stream);
            did_ws=mxfp4_wsmm_launch(ctx->y,ctx->gate,dw,dbs,dg,S,I,D,ctx->stream);
        }
        if(!did_ws){
            dim3 hidden((unsigned)((I+63)/64),(unsigned)((S+15)/16));
            dim3 output((unsigned)((D+63)/64),(unsigned)((S+15)/16));
            mxfp4_gate_up<<<hidden,256,0,ctx->stream>>>(ctx->gate,ctx->up,ctx->x,gw,uw,gbs,ubs,gg,ug,S,D,I);
            act_mul(ctx->gate,ctx->up,(size_t)S*I,ctx->stream);
            mxfp4_matmul<<<output,128,0,ctx->stream>>>(ctx->y,ctx->gate,dw,dbs,dg,S,I,D);
        }
    }
    if(!cuda_ok(cudaGetLastError(),"expert mxfp4 launch")||
       !cuda_ok(cudaMemcpyAsync(ctx->host_y,ctx->y,xb,cudaMemcpyDeviceToHost,ctx->stream),
                               "expert mxfp4 output download")||
       !cuda_ok(cudaStreamSynchronize(ctx->stream),"expert mxfp4 synchronize"))return 0;
    std::memcpy(y,ctx->host_y,xb);
    return 1;
}

extern "C" int coli_cuda_expert_mlp_nvfp4(ColiCudaTensor *gate,ColiCudaTensor *up,
        ColiCudaTensor *down,float *y,const float *x,int S){
    if(!gate||!up||!down||!x||!y||S<1||gate->fmt!=5||up->fmt!=5||down->fmt!=5||
       gate->device!=up->device||gate->device!=down->device||gate->I!=up->I||
       gate->O!=up->O||down->I!=gate->O||down->O!=gate->I)return 0;
    DeviceContext *ctx=find_ctx(gate->device);if(!select_ctx(ctx)||ctx->compute_major<7)return 0;
    std::lock_guard<std::mutex> _scratch_lk(scratch_mu(ctx));
    int D=gate->I,I=gate->O;size_t xb=(size_t)S*D*sizeof(float),ib=(size_t)S*I*sizeof(float);
    if(!reserve(&ctx->x,&ctx->x_cap,xb)||!reserve(&ctx->gate,&ctx->gate_cap,ib)||
       !reserve(&ctx->up,&ctx->up_cap,ib)||!reserve(&ctx->y,&ctx->y_cap,xb)||
       !reserve_pinned(&ctx->host_x,&ctx->host_x_cap,xb)||
       !reserve_pinned(&ctx->host_y,&ctx->host_y_cap,xb))return 0;
    // Per-call GPU-timeline split (COLI_NVFP4_EVT=1), mirroring COLI_RELU2_EVT. This was
    // the ONLY expert entry point without one, and it is the one the SwiGLU models take —
    // so the question "where do M2.7's ~12 ms per expert call go?" had no instrument.
    //
    // It answered the first half decisively. Over 6 runs of identical work (M2.7, 128-token
    // prefill, 79.63 GB, 10000 calls) the kernel is 2.47-2.51 s — under 2% spread — while
    // staging is 36.9-78.7 s. **Staging is always >=15x the kernel**, so the GEMM is not
    // where this phase goes, and it stages at 1.0-2.2 GB/s for a host->device copy on a
    // box whose "device" memory is the same LPDDR5X. That rate is the open question.
    //
    // It did NOT answer the second half. Staging cost plateaus mid-run (53.53 s at 8000
    // calls -> 53.89 s at 10000), which reads like first-touch on freshly-pread pages —
    // but a 128-token prefill touches ~15872 DISTINCT experts, so a per-expert cost would
    // grow linearly rather than plateau, and running with a WARM page cache made staging
    // *slower* (78.70 s), which first-touch and I/O-contention both predict the opposite
    // of. The mechanism is unknown; do not optimize against the first-touch story.
    // Bucketed by input width D so the shared expert and the routed experts stay separable.
    static int s_evt=-1; static cudaEvent_t s_v0=0,s_v1=0,s_v2=0;
    struct NvEvt { int d; double stage_ms,kern_ms; long calls,rows,wbytes; };
    static NvEvt s_nv[4]={}; static int s_nnv=0;
    if(s_evt<0){ const char*e=getenv("COLI_NVFP4_EVT"); s_evt=e&&atoi(e);
        if(s_evt){cudaEventCreate(&s_v0);cudaEventCreate(&s_v1);cudaEventCreate(&s_v2);} }
    NvEvt *nv=nullptr;
    if(s_evt){
        for(int i=0;i<s_nnv;i++) if(s_nv[i].d==D){ nv=&s_nv[i]; break; }
        if(!nv&&s_nnv<4){ s_nv[s_nnv].d=D; nv=&s_nv[s_nnv++]; }
    }
    std::memcpy(ctx->host_x,x,xb);
    if(!cuda_ok(cudaMemcpyAsync(ctx->x,ctx->host_x,xb,cudaMemcpyHostToDevice,ctx->stream),
                               "expert nvfp4 input upload"))return 0;
    if(s_evt) cudaEventRecord(s_v0,ctx->stream);
    const uint8_t *gw=(const uint8_t*)gate->weights,*uw=(const uint8_t*)up->weights,*dw=(const uint8_t*)down->weights;
    const uint8_t *gbs=(const uint8_t*)gate->bscale,*ubs=(const uint8_t*)up->bscale,*dbs=(const uint8_t*)down->bscale;
    float gg=gate->gscale,ug=up->gscale,dg=down->gscale;
    // Stage nibbles + ue4m3 block scales to device so the tiled kernel reads clean memory
    // instead of freshly-pread (dirty, coherence-heavy) host pages. Decode (S==1 gemv)
    // never reaches this branch; zero-copy stays the decode path.
    //
    // This used to require S >= 16, and that threshold was wrong for every routed-expert
    // model. An expert in prefill does not see the prompt length — it sees
    // S = tokens*top_k/n_experts, which for MiniMax-M2.7 (top_k 8, 256 experts) is ~4 at a
    // 128-token prompt and ~16 at 512. So the models this exists for sat at or below the
    // gate and read dirty pages on all ~15872 expert calls of a prefill. Measured on M2.7,
    // ABBA-interleaved, 4 runs per arm, tokens identical in all 8: gpu-ffn median
    // 227468 ms at the old gate vs 184007 ms always-staged — **1.24x**, wall 1.20x.
    //
    // Gated on the path actually taken, NOT on a row-count threshold. The single-row gemv
    // below reads the weights ONCE, so staging them costs a full H2D copy to save nothing —
    // and decode is exactly that path. Removing the old `S >= 16` gate without this
    // condition made every decode step stage its experts (~15.8 MB each on Nemotron) and
    // cost **9.8 vs 11.04 tok/s**. `S >= 16` had been keeping decode out by accident; the
    // real predicate is "am I about to run the tiled/WSMM kernel", which re-reads the
    // weight per m-tile and therefore does benefit.
    static int s_tiled=-1;
    if(s_tiled<0){const char*e=getenv("COLI_NVFP4_TILED");s_tiled=e&&atoi(e);}
    const bool row_path=(S==1&&!s_tiled);
    static int s_dc=-1;
    if(s_dc<0){const char*e=getenv("COLI_FFN_DEVCOPY");s_dc=(!e||atoi(e));}
    if(s_dc&&!row_path){
        size_t gnb=(size_t)I*((D+1)/2), dnb=(size_t)D*((I+1)/2);       // nibble bytes (gate/up, down)
        size_t gsb=(size_t)I*((D+15)/16), dsb=(size_t)D*((I+15)/16);   // block-scale bytes
        if(reserve_bytes((void**)&ctx->ewg,&ctx->ewg_cap,gnb)&&reserve_bytes((void**)&ctx->ewu,&ctx->ewu_cap,gnb)&&
           reserve_bytes((void**)&ctx->ewd,&ctx->ewd_cap,dnb)&&reserve_bytes((void**)&ctx->ebsg,&ctx->ebsg_cap,gsb)&&
           reserve_bytes((void**)&ctx->ebsu,&ctx->ebsu_cap,gsb)&&reserve_bytes((void**)&ctx->ebsd,&ctx->ebsd_cap,dsb)){
            cudaMemcpyAsync(ctx->ewg,gw,gnb,cudaMemcpyHostToDevice,ctx->stream);
            cudaMemcpyAsync(ctx->ewu,uw,gnb,cudaMemcpyHostToDevice,ctx->stream);
            cudaMemcpyAsync(ctx->ewd,dw,dnb,cudaMemcpyHostToDevice,ctx->stream);
            cudaMemcpyAsync(ctx->ebsg,gbs,gsb,cudaMemcpyHostToDevice,ctx->stream);
            cudaMemcpyAsync(ctx->ebsu,ubs,gsb,cudaMemcpyHostToDevice,ctx->stream);
            cudaMemcpyAsync(ctx->ebsd,dbs,dsb,cudaMemcpyHostToDevice,ctx->stream);
            gw=ctx->ewg;uw=ctx->ewu;dw=ctx->ewd;gbs=ctx->ebsg;ubs=ctx->ebsu;dbs=ctx->ebsd;
            if(nv) nv->wbytes+=(long)(gnb+gnb+dnb+gsb+gsb+dsb);
        }
    }
    if(s_evt) cudaEventRecord(s_v1,ctx->stream);
    if(row_path){
        nvfp4_gemv_dispatch(ctx->gate,ctx->x,gw,gbs,gg,D,I,ctx->stream);
        nvfp4_gemv_dispatch(ctx->up,  ctx->x,uw,ubs,ug,D,I,ctx->stream);
        act_mul(ctx->gate,ctx->up,(size_t)I,ctx->stream);
        nvfp4_gemv_dispatch(ctx->y,ctx->gate,dw,dbs,dg,I,D,ctx->stream);
    }else{
        // Weight-stationary small-M path, the same #90 fix the relu2 expert already had: at
        // prefill this expert sees only S = tokens*top_k/n_experts rows (~4-16 on M2.7,
        // ~26 on Nemotron), so the WMMA tile re-dequants the whole weight per 16-row m-tile
        // and runs at ~0.26% of tensor peak. Reading each weight once and accumulating over
        // all S rows in registers is the win. Wiring it only into relu2 left the three
        // SwiGLU models (M2.7/M3/GLM) on WMMA. Measured on M2.7, ABBA, tokens identical:
        // gpu-ffn 258623 -> 222521 ms, **1.16x**.
        //
        // Not a knob. `nvfp4_wsmm_launch` selects the smallest MT bucket >= S and declines
        // above 32, where the MMA amortizes and WMMA is the better kernel — and S is
        // already the only thing that decides this. It carries the model (top_k /
        // n_experts) and the cluster shape: under expert parallelism a node owns fewer
        // experts, so each expert it does own sees proportionally MORE rows and slides
        // toward the WMMA end on its own. An env override could only disagree with the
        // shape actually being computed.
        bool did_ws=false;
        if(nvfp4_wsmm_launch(ctx->gate,ctx->x,gw,gbs,gg,S,D,I,ctx->stream)){
            // Same S ⇒ the up/down launches take the same MT bucket and cannot decline.
            nvfp4_wsmm_launch(ctx->up,ctx->x,uw,ubs,ug,S,D,I,ctx->stream);
            act_mul(ctx->gate,ctx->up,(size_t)S*I,ctx->stream);
            did_ws=nvfp4_wsmm_launch(ctx->y,ctx->gate,dw,dbs,dg,S,I,D,ctx->stream);
        }
        if(!did_ws){
            dim3 hidden((unsigned)((I+63)/64),(unsigned)((S+15)/16));
            dim3 output((unsigned)((D+63)/64),(unsigned)((S+15)/16));
            nvfp4_gate_up<<<hidden,256,0,ctx->stream>>>(ctx->gate,ctx->up,ctx->x,gw,uw,gbs,ubs,gg,ug,S,D,I);
            act_mul(ctx->gate,ctx->up,(size_t)S*I,ctx->stream);
            nvfp4_matmul<<<output,128,0,ctx->stream>>>(ctx->y,ctx->gate,dw,dbs,dg,S,I,D);
        }
    }
    if(s_evt) cudaEventRecord(s_v2,ctx->stream);
    if(!cuda_ok(cudaGetLastError(),"expert nvfp4 launch")||
       !cuda_ok(cudaMemcpyAsync(ctx->host_y,ctx->y,xb,cudaMemcpyDeviceToHost,ctx->stream),
                               "expert nvfp4 output download")||
       !cuda_ok(cudaStreamSynchronize(ctx->stream),"expert nvfp4 synchronize"))return 0;
    if(nv){ float sm=0,km=0; cudaEventElapsedTime(&sm,s_v0,s_v1); cudaEventElapsedTime(&km,s_v1,s_v2);
        nv->stage_ms+=sm; nv->kern_ms+=km; nv->calls++; nv->rows+=S;
        if(nv->calls%2000==0) for(int i=0;i<s_nnv;i++){ NvEvt *e=&s_nv[i]; fprintf(stderr,
            "[nvfp4-evt] D=%d calls=%ld rows=%ld stage=%.2fs (%.2f GB, %.2f GB/s) kernel=%.2fs "
            "avg_rows=%.1f per_call=%.3fms\n",
            e->d,e->calls,e->rows,e->stage_ms/1e3,e->wbytes/1e9,
            e->stage_ms>0?(e->wbytes/1e9)/(e->stage_ms/1e3):0.0,e->kern_ms/1e3,
            e->calls?(double)e->rows/e->calls:0.0,
            e->calls?(e->stage_ms+e->kern_ms)/e->calls:0.0); } }
    std::memcpy(y,ctx->host_y,xb);
    return 1;
}

/* Gateless ReLU² NVFP4 expert FFN (Nemotron-H): y = down( relu(up·x)² ). The two-tensor
 * expert has no gate projection, so this reuses the SAME NVFP4 weight decode + GEMV /
 * tiled-matmul device code as coli_cuda_expert_mlp_nvfp4 (nvfp4_gemv / nvfp4_matmul); the
 * only difference is the middle activation — relu²-in-place over the single up projection
 * instead of the SwiGLU gate*up combine. Requires up/down at fmt==5, with down the
 * transpose of up (down->I==up->O, down->O==up->I). x is [S, up->I] (latent). */
extern "C" int coli_cuda_expert_mlp_nvfp4_relu2(ColiCudaTensor *up,
        ColiCudaTensor *down,float *y,const float *x,int S,int exact){
    if(!up||!down||!x||!y||S<1||up->fmt!=5||down->fmt!=5||
       up->device!=down->device||down->I!=up->O||down->O!=up->I)return 0;
    DeviceContext *ctx=find_ctx(up->device);if(!select_ctx(ctx)||ctx->compute_major<7)return 0;
    std::lock_guard<std::mutex> _scratch_lk(scratch_mu(ctx));
    int D=up->I,I=up->O;size_t xb=(size_t)S*D*sizeof(float),ib=(size_t)S*I*sizeof(float);
    if(!reserve(&ctx->x,&ctx->x_cap,xb)||!reserve(&ctx->up,&ctx->up_cap,ib)||
       !reserve(&ctx->y,&ctx->y_cap,xb)||
       !reserve_pinned(&ctx->host_x,&ctx->host_x_cap,xb)||
       !reserve_pinned(&ctx->host_y,&ctx->host_y_cap,xb))return 0;
    // Optional per-call GPU-timeline split (COLI_RELU2_EVT=1), mirroring COLI_FFN_EVT on
    // the fp8 path but with one more event so the weight STAGING is separated from the
    // kernels. Task #98: the shared expert's cost grows ~7x with the expert-cache budget
    // while gpu-ffn — which reads host memory the same way — does not, and the two
    // candidate locations for that time are the staging H2D and the kernels' own reads.
    // Diagnostic only; the events add a stream marker each, no serialization.
    //
    // Bucketed by input width D, because the shared expert and the routed experts both
    // arrive here and the asymmetry between them IS the question: on Nemotron-3-Super the
    // shared expert is D=4096 (hidden) and the routed experts are D=1024 (MoE latent).
    static int s_evt=-1; static cudaEvent_t s_e0=0,s_e1=0,s_e2=0;
    struct Relu2Evt { int d; double stage_ms,kern_ms; long calls,rows,wbytes; };
    static Relu2Evt s_ev[4]={}; static int s_nev=0;
    if(s_evt<0){ const char*e=getenv("COLI_RELU2_EVT"); s_evt=e&&atoi(e);
        if(s_evt){cudaEventCreate(&s_e0);cudaEventCreate(&s_e1);cudaEventCreate(&s_e2);} }
    Relu2Evt *ev=nullptr;
    if(s_evt){
        for(int i=0;i<s_nev;i++) if(s_ev[i].d==D){ ev=&s_ev[i]; break; }
        if(!ev&&s_nev<4){ s_ev[s_nev].d=D; ev=&s_ev[s_nev++]; }
    }
    std::memcpy(ctx->host_x,x,xb);
    if(!cuda_ok(cudaMemcpyAsync(ctx->x,ctx->host_x,xb,cudaMemcpyHostToDevice,ctx->stream),
                               "expert nvfp4 relu2 input upload"))return 0;
    if(s_evt) cudaEventRecord(s_e0,ctx->stream);
    const uint8_t *uw=(const uint8_t*)up->weights,*dw=(const uint8_t*)down->weights;
    const uint8_t *ubs=(const uint8_t*)up->bscale,*dbs=(const uint8_t*)down->bscale;
    float ug=up->gscale,dg=down->gscale;
    // Stage up/down nibbles + ue4m3 block scales to device so the tiled kernel reads clean
    // memory instead of freshly-pread host pages.
    //
    // The old `S >= 16` gate is gone for the reason measured on the SwiGLU twin: a routed
    // expert never sees the prompt length, only S = tokens*top_k/n_experts, so the gate
    // excluded exactly the models it was written for. But it must be replaced by the
    // per-PATH condition below, not by nothing — **"decode stays zero-copy" is load-bearing
    // and `S >= 16` was enforcing it by accident.** Staging on the single-row path copies
    // the whole weight to save one read of it: Nemotron decode went 11.04 -> 9.8 tok/s
    // before this was gated. `exact` (MTP verify) also takes the row path.
    static int s_tiled=-1;
    if(s_tiled<0){const char*e=getenv("COLI_NVFP4_TILED");s_tiled=e&&atoi(e);}
    const bool row_path=((S==1&&!s_tiled)||exact);
    static int s_dc=-1;
    if(s_dc<0){const char*e=getenv("COLI_FFN_DEVCOPY");s_dc=(!e||atoi(e));}
    if(s_dc&&!row_path){
        size_t unb=(size_t)I*((D+1)/2), dnb=(size_t)D*((I+1)/2);       // nibble bytes (up, down)
        size_t usb=(size_t)I*((D+15)/16), dsb=(size_t)D*((I+15)/16);   // block-scale bytes
        if(reserve_bytes((void**)&ctx->ewu,&ctx->ewu_cap,unb)&&reserve_bytes((void**)&ctx->ewd,&ctx->ewd_cap,dnb)&&
           reserve_bytes((void**)&ctx->ebsu,&ctx->ebsu_cap,usb)&&reserve_bytes((void**)&ctx->ebsd,&ctx->ebsd_cap,dsb)){
            cudaMemcpyAsync(ctx->ewu,uw,unb,cudaMemcpyHostToDevice,ctx->stream);
            cudaMemcpyAsync(ctx->ewd,dw,dnb,cudaMemcpyHostToDevice,ctx->stream);
            cudaMemcpyAsync(ctx->ebsu,ubs,usb,cudaMemcpyHostToDevice,ctx->stream);
            cudaMemcpyAsync(ctx->ebsd,dbs,dsb,cudaMemcpyHostToDevice,ctx->stream);
            uw=ctx->ewu;dw=ctx->ewd;ubs=ctx->ebsu;dbs=ctx->ebsd;
            if(ev) ev->wbytes+=(long)(unb+dnb+usb+dsb);
        }
    }
    if(s_evt) cudaEventRecord(s_e1,ctx->stream);
    // `exact`: force the per-row gemv path for EVERY row, even at S>1. This makes a
    // multi-row (collided-expert) call bit-identical to S separate S==1 decode calls —
    // needed by the MTP verify forward, whose S>1 logits must match sequential decode to
    // the bit or a near-tie argmax forks the accepted token from DRAFT=0. The WSMM/WMMA
    // paths below reduce over K in a different order, so they cannot be used for verify.
    // S is tiny at verify, so the lost row-parallelism costs nothing there.
    if(row_path){
        int tpb=256,wpb=tpb>>5;
        for(int r=0;r<S;r++){
            const float *xr=ctx->x+(size_t)r*D; float *tr=ctx->up+(size_t)r*I; float *yr=ctx->y+(size_t)r*D;
            // t = up·x  → tr [I].  The u4 kernel reads 512 B/warp against the narrow
            // kernel's 16 B and is the decode hot path; `exact` (MTP verify) must keep the
            // narrow one, whose reduction order sequential decode is gated against.
            if(!exact && nvfp4_u4_ok(uw,D))
                nvfp4_gemv_u4<<<(unsigned)((I+wpb-1)/wpb),tpb,(size_t)D*sizeof(float),ctx->stream>>>(tr,xr,uw,ubs,ug,D,I);
            else
                nvfp4_gemv<<<(unsigned)((I+wpb-1)/wpb),tpb,(size_t)D*sizeof(float),ctx->stream>>>(tr,xr,uw,ubs,ug,D,I);
            // t = relu(t)²
            relu2_inplace<<<(unsigned)(((size_t)I+255)/256),256,0,ctx->stream>>>(tr,(size_t)I);
            // y = down·t → yr [D]
            if(!exact && nvfp4_u4_ok(dw,I))
                nvfp4_gemv_u4<<<(unsigned)((D+wpb-1)/wpb),tpb,(size_t)I*sizeof(float),ctx->stream>>>(yr,tr,dw,dbs,dg,I,D);
            else
                nvfp4_gemv<<<(unsigned)((D+wpb-1)/wpb),tpb,(size_t)I*sizeof(float),ctx->stream>>>(yr,tr,dw,dbs,dg,I,D);
        }
    }else{
        // Weight-stationary small-M path: reads each weight once and amortizes the dequant
        // across all S rows, vs the WMMA path's per-16-row-m-tile re-dequant at ~1/8 MMA
        // utilization (#90). `nvfp4_wsmm_launch` declines above S=32, where the MMA
        // amortizes, and the call falls through to WMMA — see the SwiGLU twin for why S is
        // the whole decision and this is not a knob.
        bool did_ws=false;
        {
            if(nvfp4_wsmm_launch(ctx->up,ctx->x,uw,ubs,ug,S,D,I,ctx->stream)){
                relu2_inplace<<<(unsigned)(((size_t)S*I+255)/256),256,0,ctx->stream>>>(ctx->up,(size_t)S*I);
                did_ws=nvfp4_wsmm_launch(ctx->y,ctx->up,dw,dbs,dg,S,I,D,ctx->stream);
            }
        }
        if(!did_ws){
            // Single up projection (no gate) via the tiled WMMA matmul, then relu², then down.
            dim3 hidden((unsigned)((I+63)/64),(unsigned)((S+15)/16));
            dim3 output((unsigned)((D+63)/64),(unsigned)((S+15)/16));
            nvfp4_matmul<<<hidden,128,0,ctx->stream>>>(ctx->up,ctx->x,uw,ubs,ug,S,D,I);
            relu2_inplace<<<(unsigned)(((size_t)S*I+255)/256),256,0,ctx->stream>>>(ctx->up,(size_t)S*I);
            nvfp4_matmul<<<output,128,0,ctx->stream>>>(ctx->y,ctx->up,dw,dbs,dg,S,I,D);
        }
    }
    if(s_evt) cudaEventRecord(s_e2,ctx->stream);
    if(!cuda_ok(cudaGetLastError(),"expert nvfp4 relu2 launch")||
       !cuda_ok(cudaMemcpyAsync(ctx->host_y,ctx->y,xb,cudaMemcpyDeviceToHost,ctx->stream),
                               "expert nvfp4 relu2 output download")||
       !cuda_ok(cudaStreamSynchronize(ctx->stream),"expert nvfp4 relu2 synchronize"))return 0;
    if(ev){ float sm=0,km=0; cudaEventElapsedTime(&sm,s_e0,s_e1); cudaEventElapsedTime(&km,s_e1,s_e2);
        ev->stage_ms+=sm; ev->kern_ms+=km; ev->calls++; ev->rows+=S;
        if(ev->calls%2000==0) for(int i=0;i<s_nev;i++){ Relu2Evt *e=&s_ev[i]; fprintf(stderr,
            "[relu2-evt] D=%d calls=%ld rows=%ld stage=%.2fs (%.2f GB, %.2f GB/s) kernel=%.2fs avg_rows=%.1f\n",
            e->d,e->calls,e->rows,e->stage_ms/1e3,e->wbytes/1e9,
            e->stage_ms>0?(e->wbytes/1e9)/(e->stage_ms/1e3):0.0,e->kern_ms/1e3,
            e->calls?(double)e->rows/e->calls:0.0); } }
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
    std::lock_guard<std::mutex> _scratch_lk(scratch_mu(ctx));
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
    act_mul(ctx->gate,ctx->up,(size_t)S*I,ctx->stream);
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
    int all_fp8=1;
    for(int c=0;c<count;c++){
        ColiCudaTensor *g=gates[c],*u=ups[c],*d=downs[c];
        if(!g||!u||!d||rows[c]<1||g->device!=device||u->device!=device||d->device!=device||
           g->I!=D||u->I!=D||g->O!=I||u->O!=I||d->I!=I||d->O!=D) return 0;
        host[c]={g->weights,u->weights,d->weights,g->scales,u->scales,d->scales,
                 g->fmt,u->fmt,d->fmt,rows[c],total,g->wrapped};
        all_fp8&=g->fmt==4&&u->fmt==4&&d->fmt==4;
        total+=rows[c]; if(rows[c]>max_rows) max_rows=rows[c];
    }
    DeviceContext *ctx=find_ctx(device); if(!select_ctx(ctx)) return 0;
    std::lock_guard<std::mutex> _scratch_lk(scratch_mu(ctx));
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
    if(all_fp8&&ctx->compute_major>=7){
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
            act_mul(g8,u8,(size_t)r*I,ctx->stream);
            fp8a16_matmul<<<og8,128,0,ctx->stream>>>(y8,g8,
                (const uint8_t*)host[c].d,host[c].ds,r,I,D);
            off8+=r;
        }
    }else{
        dim3 hg((unsigned)I,(unsigned)max_rows,(unsigned)count),og((unsigned)D,(unsigned)max_rows,(unsigned)count);
        grouped_hidden<<<hg,256,0,ctx->stream>>>(ctx->gate,ctx->x,dev,I,D,0);
        grouped_hidden<<<hg,256,0,ctx->stream>>>(ctx->up,ctx->x,dev,I,D,1);
        act_mul(ctx->gate,ctx->up,(size_t)total*I,ctx->stream);
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

/* Grouped gateless ReLU² NVFP4 experts (Nemotron-H): the coli_cuda_expert_group idea
 * applied to the two-tensor relu² expert. The math is byte-for-byte the single-expert
 * coli_cuda_expert_mlp_nvfp4_relu2 — same nvfp4_gemv / nvfp4_matmul, same relu2_inplace,
 * same per-expert accumulation order — only the transfers are pooled: ONE H2D of all
 * rows and ONE D2H of all results per call instead of a synchronous round-trip each.
 * That is the whole point: the measured decode cost is 22 experts x 40 layers = 880
 * calls/token at ~54 us, of which only ~10 us is kernel; ~44 us is round-trip. Kernels
 * are enqueued back-to-back on ctx->stream and only synchronized once, at the download.
 *
 * x/y hold sum(rows) consecutive [D] rows in expert order (D = up->I, the MoE latent for
 * Nemotron-H); rows[c] is expert c's row count. Zero-copy only, like the single-expert
 * path — no devcopy staging, because the NVFP4 weight scratch (ctx->ewu/ebsu) is a single
 * buffer and a group would need one per expert; decode reads each expert's weights once
 * anyway, so staging would be pure added copy. No GroupDesc upload either: each expert
 * gets its own launch trio with host-side pointers, so there is no ≤64 count cap here
 * (the Rust caller still chunks at 64 to share the shape of the fp8 group path). */
/* SEGMENTED gateless-ReLU2 NVFP4 experts (Nemotron-H): the whole layer in THREE launches
 * (up / relu2 / down) instead of three per expert.
 *
 * Same contract as coli_cuda_expert_group_nvfp4_relu2 — x and y hold sum(rows) rows of
 * `ups[i]->I` floats in call order — but instead of looping experts on the stream it
 * builds row-tile descriptors and issues one `nvfp4_matmul_seg` grid covering all of them.
 * At ~453 experts that is ~453x the blocks per launch, which is the point: the per-expert
 * form is occupancy-bound (86 blocks, 6.2 GB/s, 2.3% of memory peak).
 *
 * Weights stay zero-copy host pointers, as in the per-expert path — COLI_FFN_DEVCOPY is
 * deliberately NOT applied here, having been A/B'd as a slight loss (gpu-ffn 8911 -> 8375
 * ms with it OFF).
 *
 * ⚠️ UNMEASURED SUSPICION (2026-08-03): this hands the kernel zero-copy HOST weight
 * pointers through the `sg_uw`/`sg_dw` DEVICE arrays. On the decode path that exact
 * indirection measured **2.8x slower** than passing the same pointers in kernel parameter
 * space — same kernel, same grid, bit-identical output (see `SegP`). So this path is very
 * likely paying the same tax, and the 2.4x regression recorded on
 * `expert_seg_decode_enabled` was blamed on the wrong thing (16-row MMA tile waste; the
 * decode GEMV tiles nothing and reproduced the penalty). Re-measure with a SegP-style
 * parameter block before concluding anything about segmented prefill.
 *
 * Returns 0 (caller falls back) on any shape mismatch, so this is a pure fast path. */
extern "C" int coli_cuda_expert_seg_nvfp4_relu2(ColiCudaTensor *const *ups,
        ColiCudaTensor *const *downs, const int *rows, int count,
        float *y, const float *x) {
    if (!ups || !downs || !rows || count < 1 || !x || !y) return 0;
    ColiCudaTensor *u0 = ups[0];
    if (!u0 || u0->fmt != 5) return 0;
    int device = u0->device, D = u0->I, I = u0->O;
    DeviceContext *ctx = find_ctx(device);
    if (!select_ctx(ctx) || ctx->compute_major < 7) return 0;
    std::lock_guard<std::mutex> _scratch_lk(scratch_mu(ctx));
    // Uniform shape + device is required: one grid, one (K,N) pair.
    long total = 0;
    for (int c = 0; c < count; c++) {
        ColiCudaTensor *u = ups[c], *d = downs[c];
        if (!u || !d || rows[c] < 1 || u->fmt != 5 || d->fmt != 5 ||
            u->device != device || d->device != device ||
            u->I != D || u->O != I || d->I != I || d->O != D) return 0;
        total += rows[c];
    }
    if (total < 1) return 0;

    // Row-tile descriptors: 16 rows per tile, per expert (partial tiles are masked).
    std::vector<int> h_tile_e, h_tile_r0, h_off(count), h_rows(count);
    std::vector<const uint8_t *> h_uw(count), h_ubs(count), h_dw(count), h_dbs(count);
    std::vector<float> h_ug(count), h_dg(count);
    int off = 0;
    for (int c = 0; c < count; c++) {
        h_off[c] = off; h_rows[c] = rows[c];
        h_uw[c] = (const uint8_t *)ups[c]->weights;  h_ubs[c] = (const uint8_t *)ups[c]->bscale;
        h_dw[c] = (const uint8_t *)downs[c]->weights; h_dbs[c] = (const uint8_t *)downs[c]->bscale;
        h_ug[c] = ups[c]->gscale; h_dg[c] = downs[c]->gscale;
        for (int r = 0; r < rows[c]; r += 16) { h_tile_e.push_back(c); h_tile_r0.push_back(r); }
        off += rows[c];
    }
    int ntiles = (int)h_tile_e.size();

    size_t xb = (size_t)total * D * sizeof(float), ib = (size_t)total * I * sizeof(float);
    if (!reserve(&ctx->x, &ctx->x_cap, xb) || !reserve(&ctx->up, &ctx->up_cap, ib) ||
        !reserve(&ctx->y, &ctx->y_cap, xb) ||
        !reserve_pinned(&ctx->host_x, &ctx->host_x_cap, xb) ||
        !reserve_pinned(&ctx->host_y, &ctx->host_y_cap, xb)) return 0;
    // Descriptor + pointer arrays (small: a few KB at 453 experts).
    size_t ib_i = (size_t)ntiles * sizeof(int), cb_i = (size_t)count * sizeof(int);
    size_t cb_p = (size_t)count * sizeof(void *), cb_f = (size_t)count * sizeof(float);
    if (!reserve_bytes((void **)&ctx->sg_tile_e, &ctx->sg_tile_e_cap, ib_i) ||
        !reserve_bytes((void **)&ctx->sg_tile_r0, &ctx->sg_tile_r0_cap, ib_i) ||
        !reserve_bytes((void **)&ctx->sg_off, &ctx->sg_off_cap, cb_i) ||
        !reserve_bytes((void **)&ctx->sg_rows, &ctx->sg_rows_cap, cb_i) ||
        !reserve_bytes((void **)&ctx->sg_uw, &ctx->sg_uw_cap, cb_p) ||
        !reserve_bytes((void **)&ctx->sg_ubs, &ctx->sg_ubs_cap, cb_p) ||
        !reserve_bytes((void **)&ctx->sg_dw, &ctx->sg_dw_cap, cb_p) ||
        !reserve_bytes((void **)&ctx->sg_dbs, &ctx->sg_dbs_cap, cb_p) ||
        !reserve_bytes((void **)&ctx->sg_ug, &ctx->sg_ug_cap, cb_f) ||
        !reserve_bytes((void **)&ctx->sg_dg, &ctx->sg_dg_cap, cb_f)) return 0;

    cudaStream_t st = ctx->stream;
    std::memcpy(ctx->host_x, x, xb);
    bool ok = cuda_ok(cudaMemcpyAsync(ctx->x, ctx->host_x, xb, cudaMemcpyHostToDevice, st), "seg x up") &&
        cuda_ok(cudaMemcpyAsync(ctx->sg_tile_e, h_tile_e.data(), ib_i, cudaMemcpyHostToDevice, st), "seg tile_e") &&
        cuda_ok(cudaMemcpyAsync(ctx->sg_tile_r0, h_tile_r0.data(), ib_i, cudaMemcpyHostToDevice, st), "seg tile_r0") &&
        cuda_ok(cudaMemcpyAsync(ctx->sg_off, h_off.data(), cb_i, cudaMemcpyHostToDevice, st), "seg off") &&
        cuda_ok(cudaMemcpyAsync(ctx->sg_rows, h_rows.data(), cb_i, cudaMemcpyHostToDevice, st), "seg rows") &&
        cuda_ok(cudaMemcpyAsync(ctx->sg_uw, h_uw.data(), cb_p, cudaMemcpyHostToDevice, st), "seg uw") &&
        cuda_ok(cudaMemcpyAsync(ctx->sg_ubs, h_ubs.data(), cb_p, cudaMemcpyHostToDevice, st), "seg ubs") &&
        cuda_ok(cudaMemcpyAsync(ctx->sg_dw, h_dw.data(), cb_p, cudaMemcpyHostToDevice, st), "seg dw") &&
        cuda_ok(cudaMemcpyAsync(ctx->sg_dbs, h_dbs.data(), cb_p, cudaMemcpyHostToDevice, st), "seg dbs") &&
        cuda_ok(cudaMemcpyAsync(ctx->sg_ug, h_ug.data(), cb_f, cudaMemcpyHostToDevice, st), "seg ug") &&
        cuda_ok(cudaMemcpyAsync(ctx->sg_dg, h_dg.data(), cb_f, cudaMemcpyHostToDevice, st), "seg dg");
    if (!ok) return 0;

    dim3 gup((unsigned)((I + 63) / 64), (unsigned)ntiles);
    dim3 gdn((unsigned)((D + 63) / 64), (unsigned)ntiles);
    nvfp4_matmul_seg<<<gup, 128, 0, st>>>(ctx->up, ctx->x,
        (const uint8_t *const *)ctx->sg_uw, (const uint8_t *const *)ctx->sg_ubs, ctx->sg_ug,
        ctx->sg_tile_e, ctx->sg_tile_r0, ctx->sg_off, ctx->sg_rows, D, I);
    relu2_inplace<<<(unsigned)(((size_t)total * I + 255) / 256), 256, 0, st>>>(ctx->up, (size_t)total * I);
    nvfp4_matmul_seg<<<gdn, 128, 0, st>>>(ctx->y, ctx->up,
        (const uint8_t *const *)ctx->sg_dw, (const uint8_t *const *)ctx->sg_dbs, ctx->sg_dg,
        ctx->sg_tile_e, ctx->sg_tile_r0, ctx->sg_off, ctx->sg_rows, I, D);
    if (!cuda_ok(cudaGetLastError(), "seg launch") ||
        !cuda_ok(cudaMemcpyAsync(ctx->host_y, ctx->y, xb, cudaMemcpyDeviceToHost, st), "seg y down") ||
        !cuda_ok(cudaStreamSynchronize(st), "seg sync")) return 0;
    std::memcpy(y, ctx->host_y, xb);
    return 1;
}

extern "C" int coli_cuda_expert_group_nvfp4_relu2(ColiCudaTensor *const *ups,
        ColiCudaTensor *const *downs,const int *rows,int count,float *y,const float *x){
    if(!ups||!downs||!rows||!x||!y||count<1)return 0;
    ColiCudaTensor *first=ups[0]; if(!first)return 0;
    int device=first->device,D=first->I,I=first->O,total=0;
    for(int c=0;c<count;c++){
        ColiCudaTensor *u=ups[c],*d=downs[c];
        if(!u||!d||rows[c]<1||u->fmt!=5||d->fmt!=5||u->device!=device||d->device!=device||
           u->I!=D||u->O!=I||d->I!=I||d->O!=D)return 0;
        total+=rows[c];
    }
    DeviceContext *ctx=find_ctx(device);if(!select_ctx(ctx)||ctx->compute_major<7)return 0;
    std::lock_guard<std::mutex> _scratch_lk(scratch_mu(ctx));
    size_t xb=(size_t)total*D*sizeof(float),ib=(size_t)total*I*sizeof(float);
    if(!reserve(&ctx->x,&ctx->x_cap,xb)||!reserve(&ctx->up,&ctx->up_cap,ib)||
       !reserve(&ctx->y,&ctx->y_cap,xb)||
       !reserve_pinned(&ctx->host_x,&ctx->host_x_cap,xb)||
       !reserve_pinned(&ctx->host_y,&ctx->host_y_cap,xb))return 0;
    std::memcpy(ctx->host_x,x,xb);
    if(!cuda_ok(cudaMemcpyAsync(ctx->x,ctx->host_x,xb,cudaMemcpyHostToDevice,ctx->stream),
                               "expert group nvfp4 relu2 input upload"))return 0;
    static int s_tiled=-1;
    if(s_tiled<0){const char*e=getenv("COLI_NVFP4_TILED");s_tiled=e&&atoi(e);}
    int tpb=256,wpb=tpb>>5,off=0;

    /* ---- one launch trio for the WHOLE LAYER instead of one per expert ----------
     * Only for the decode shape (every expert exactly one row) with more than one
     * expert — below that there is nothing to batch. `COLI_NVFP4_SEG_GEMV=0` forces
     * the per-expert loop so the two can be A/B'd in one binary.
     *
     * Bit-identical to the loop below by construction (same kernels' inner loops,
     * same lane->k mapping), so a token-identity gate over this is meaningful. */
    static int s_seg=-1;
    if(s_seg<0){const char*e=getenv("COLI_NVFP4_SEG_GEMV");s_seg=e?atoi(e):1;}
    int all_one=1; for(int c=0;c<count;c++) if(rows[c]!=1){all_one=0;break;}
    if(s_seg&&all_one&&count>1&&!s_tiled){
        /* The wide reader needs EVERY expert 16 B aligned — one ineligible expert
         * makes the whole grid unusable, so this is all-or-nothing per launch. */
        int uw_ok=1,dw_ok=1;
        for(int c=0;c<count;c++){
            if(!nvfp4_u4_ok((const uint8_t*)ups[c]->weights,D)) uw_ok=0;
            if(!nvfp4_u4_ok((const uint8_t*)downs[c]->weights,I)) dw_ok=0;
        }
        /* ONE pinned upload for the whole descriptor set. The first version issued six
         * separate pageable `cudaMemcpyAsync` calls here — 240 per token across 40 MoE
         * layers — and a pageable async copy is not async: the driver stages it through
         * an internal bounce buffer and synchronizes, so each one drained the stream. */
        size_t segb=(size_t)count*sizeof(NvSegDesc);
        if(reserve_pinned(&ctx->host_seg,&ctx->host_seg_cap,segb)&&
           reserve_bytes((void**)&ctx->sg_uw,&ctx->sg_uw_cap,segb)){
            NvSegDesc *hs=(NvSegDesc*)ctx->host_seg;
            for(int c=0;c<count;c++){
                hs[c].uw=(const uint8_t*)ups[c]->weights;   hs[c].ubs=(const uint8_t*)ups[c]->bscale;
                hs[c].dw=(const uint8_t*)downs[c]->weights; hs[c].dbs=(const uint8_t*)downs[c]->bscale;
                hs[c].ug=ups[c]->gscale; hs[c].dg=downs[c]->gscale;
            }
            const NvSegDesc *dsg=(const NvSegDesc*)ctx->sg_uw;
            if(cuda_ok(cudaMemcpyAsync(ctx->sg_uw,hs,segb,cudaMemcpyHostToDevice,ctx->stream),"seg-gemv desc")){
            /* How many experts share one launch. `COLI_NVFP4_SEG_CHUNK` exists because
             * the whole-layer answer (chunk == count) turned out to be the WRONG end of
             * a curve, and a single point could not show that. chunk=1 reproduces the
             * per-expert launch structure through the segmented kernel, so the sweep
             * isolates concurrency from every other difference between the two paths. */
            static int s_chunk=-1;
            if(s_chunk<0){const char*e=getenv("COLI_NVFP4_SEG_CHUNK");s_chunk=e?atoi(e):0;}
            int chunk=(s_chunk>0&&s_chunk<count)?s_chunk:count;
            /* `COLI_NVFP4_SEG_DIRECT=1` keeps this entire host-side block — the pinned
             * descriptor upload, the chunk loop, the single D2H — but issues the ORIGINAL
             * per-expert kernels, which take the weight pointer as a kernel PARAMETER
             * instead of reading it from a descriptor. It is the control that separates
             * "the new kernel is slow" from "the new surrounding code is slow"; without
             * it the two are confounded and the measurement says nothing about which. */
            static int s_direct=-1;
            if(s_direct<0){const char*e=getenv("COLI_NVFP4_SEG_DIRECT");s_direct=e&&atoi(e);}
            /* Parameter-space segmented launch: one grid for the whole layer with the
             * expert pointers delivered as kernel arguments. `COLI_NVFP4_SEG_PARAM=0`
             * falls back to the device-buffer form for A/B. */
            static int s_param=-1;
            if(s_param<0){const char*e=getenv("COLI_NVFP4_SEG_PARAM");s_param=e?atoi(e):1;}
            if(s_param&&!s_direct&&uw_ok&&dw_ok){
                /* Chunked at SEGP_MAX so a model routing more experts than fit in one
                 * parameter block still gets parameter-space pointers. Falling back to
                 * the device-buffer kernel instead would be a silent 2.5x cliff keyed on
                 * top-k, which is exactly the kind of gate that hides for months. */
                for(int c0=0;c0<count;c0+=SEGP_MAX){
                    int nc=(c0+SEGP_MAX<count)?SEGP_MAX:(count-c0);
                    SegP pu,pd;
                    for(int c=0;c<nc;c++){
                        ColiCudaTensor *u=ups[c0+c],*d=downs[c0+c];
                        pu.w[c]=(const uint8_t*)u->weights; pu.bs[c]=(const uint8_t*)u->bscale; pu.g[c]=u->gscale;
                        pd.w[c]=(const uint8_t*)d->weights; pd.bs[c]=(const uint8_t*)d->bscale; pd.g[c]=d->gscale;
                    }
                    dim3 gup((unsigned)((I+wpb-1)/wpb),(unsigned)nc);
                    dim3 gdn((unsigned)((D+wpb-1)/wpb),(unsigned)nc);
                    float *xc=ctx->x+(size_t)c0*D,*tc=ctx->up+(size_t)c0*I,*yc=ctx->y+(size_t)c0*D;
                    nvfp4_gemv_u4_segp<<<gup,tpb,(size_t)D*sizeof(float),ctx->stream>>>(tc,xc,pu,D,I);
                    relu2_inplace<<<(unsigned)(((size_t)nc*I+255)/256),256,0,ctx->stream>>>(tc,(size_t)nc*I);
                    nvfp4_gemv_u4_segp<<<gdn,tpb,(size_t)I*sizeof(float),ctx->stream>>>(yc,tc,pd,I,D);
                }
            } else if(s_direct){
                for(int c=0;c<count;c++){
                    const uint8_t *uw=(const uint8_t*)ups[c]->weights,*dw=(const uint8_t*)downs[c]->weights;
                    const uint8_t *ubs=(const uint8_t*)ups[c]->bscale,*dbs=(const uint8_t*)downs[c]->bscale;
                    float *xc=ctx->x+(size_t)c*D,*tc=ctx->up+(size_t)c*I,*yc=ctx->y+(size_t)c*D;
                    nvfp4_gemv_u4<<<(unsigned)((I+wpb-1)/wpb),tpb,(size_t)D*sizeof(float),ctx->stream>>>(tc,xc,uw,ubs,ups[c]->gscale,D,I);
                    relu2_inplace<<<(unsigned)(((size_t)I+255)/256),256,0,ctx->stream>>>(tc,(size_t)I);
                    nvfp4_gemv_u4<<<(unsigned)((D+wpb-1)/wpb),tpb,(size_t)I*sizeof(float),ctx->stream>>>(yc,tc,dw,dbs,downs[c]->gscale,I,D);
                }
            } else
            for(int c0=0;c0<count;c0+=chunk){
                int nc=(c0+chunk<count)?chunk:(count-c0);
                dim3 gup((unsigned)((I+wpb-1)/wpb),(unsigned)nc);
                dim3 gdn((unsigned)((D+wpb-1)/wpb),(unsigned)nc);
                float *xc=ctx->x+(size_t)c0*D,*tc=ctx->up+(size_t)c0*I,*yc=ctx->y+(size_t)c0*D;
                if(uw_ok)
                    nvfp4_gemv_u4_seg<<<gup,tpb,(size_t)D*sizeof(float),ctx->stream>>>(tc,xc,dsg+c0,0,D,I);
                else
                    nvfp4_gemv_seg<<<gup,tpb,(size_t)D*sizeof(float),ctx->stream>>>(tc,xc,dsg+c0,0,D,I);
                relu2_inplace<<<(unsigned)(((size_t)nc*I+255)/256),256,0,ctx->stream>>>(tc,(size_t)nc*I);
                if(dw_ok)
                    nvfp4_gemv_u4_seg<<<gdn,tpb,(size_t)I*sizeof(float),ctx->stream>>>(yc,tc,dsg+c0,1,I,D);
                else
                    nvfp4_gemv_seg<<<gdn,tpb,(size_t)I*sizeof(float),ctx->stream>>>(yc,tc,dsg+c0,1,I,D);
            }
            if(cuda_ok(cudaGetLastError(),"seg-gemv launch")&&
               cuda_ok(cudaMemcpyAsync(ctx->host_y,ctx->y,xb,cudaMemcpyDeviceToHost,ctx->stream),"seg-gemv output download")&&
               cuda_ok(cudaStreamSynchronize(ctx->stream),"seg-gemv synchronize")){
                std::memcpy(y,ctx->host_y,xb);
                { std::lock_guard<std::mutex> lock(g_group_stats_mu);
                  g_group_calls++; g_group_experts+=(uint64_t)count; g_group_rows+=(uint64_t)total; }
                return 1;
            }
            return 0;  /* the launch/copy already failed — retrying per-expert would hide it */
            }
        }
        /* descriptor staging failed: fall through to the per-expert loop below */
    }

    for(int c=0;c<count;c++){
        int r=rows[c];
        const uint8_t *uw=(const uint8_t*)ups[c]->weights,*dw=(const uint8_t*)downs[c]->weights;
        const uint8_t *ubs=(const uint8_t*)ups[c]->bscale,*dbs=(const uint8_t*)downs[c]->bscale;
        float ug=ups[c]->gscale,dg=downs[c]->gscale;
        // This expert's slice of the pooled buffers: rows [off, off+r).
        float *xc=ctx->x+(size_t)off*D,*tc=ctx->up+(size_t)off*I,*yc=ctx->y+(size_t)off*D;
        if(r==1&&!s_tiled){
            // THE decode hot path for gateless NVFP4 experts (Nemotron): one row per
            // expert, so a gemv per projection. Prefer the 512 B/warp reader — the narrow
            // kernel reads 16 B/warp, and the routed-expert gemv measured ~110 GB/s
            // against a 172 GB/s host ceiling, so it is read-width bound, not memory bound.
            if(nvfp4_u4_ok(uw,D))
                nvfp4_gemv_u4<<<(unsigned)((I+wpb-1)/wpb),tpb,(size_t)D*sizeof(float),ctx->stream>>>(tc,xc,uw,ubs,ug,D,I);
            else
                nvfp4_gemv<<<(unsigned)((I+wpb-1)/wpb),tpb,(size_t)D*sizeof(float),ctx->stream>>>(tc,xc,uw,ubs,ug,D,I);
            relu2_inplace<<<(unsigned)(((size_t)I+255)/256),256,0,ctx->stream>>>(tc,(size_t)I);
            if(nvfp4_u4_ok(dw,I))
                nvfp4_gemv_u4<<<(unsigned)((D+wpb-1)/wpb),tpb,(size_t)I*sizeof(float),ctx->stream>>>(yc,tc,dw,dbs,dg,I,D);
            else
                nvfp4_gemv<<<(unsigned)((D+wpb-1)/wpb),tpb,(size_t)I*sizeof(float),ctx->stream>>>(yc,tc,dw,dbs,dg,I,D);
        }else{
            dim3 hidden((unsigned)((I+63)/64),(unsigned)((r+15)/16));
            dim3 output((unsigned)((D+63)/64),(unsigned)((r+15)/16));
            nvfp4_matmul<<<hidden,128,0,ctx->stream>>>(tc,xc,uw,ubs,ug,r,D,I);
            relu2_inplace<<<(unsigned)(((size_t)r*I+255)/256),256,0,ctx->stream>>>(tc,(size_t)r*I);
            nvfp4_matmul<<<output,128,0,ctx->stream>>>(yc,tc,dw,dbs,dg,r,I,D);
        }
        off+=r;
    }
    if(!cuda_ok(cudaGetLastError(),"expert group nvfp4 relu2 launch")||
       !cuda_ok(cudaMemcpyAsync(ctx->host_y,ctx->y,xb,cudaMemcpyDeviceToHost,ctx->stream),
                               "expert group nvfp4 relu2 output download")||
       !cuda_ok(cudaStreamSynchronize(ctx->stream),"expert group nvfp4 relu2 synchronize"))return 0;
    std::memcpy(y,ctx->host_y,xb);
    { std::lock_guard<std::mutex> lock(g_group_stats_mu);
      g_group_calls++; g_group_experts+=(uint64_t)count; g_group_rows+=(uint64_t)total; }
    return 1;
}

/* Grouped NVFP4 **SwiGLU** experts — the three-tensor twin of the relu2 group above.
 *
 * MiniMax-M2.7, MiniMax-M3 and GLM-5.2 had no grouped path at all. `activation().relu2`
 * is false for them, so the dispatcher offered them the *fp8* group, which declines on
 * fmt=5 and drops every one of them to the per-expert entry point:
 * ~15872 separate `coli_cuda_expert_mlp_nvfp4` calls per prefill, each paying its own
 * input H2D, weight staging, output D2H, stream sync and scratch-mutex acquire. That is
 * the whole 40.4 vs 0.9 tok/s prefill gap against Nemotron, which has had this since #37.
 *
 * The win is not fewer kernels — it is one round trip for the whole group instead of one
 * per expert, and **no weight staging at all**: weights are handed to the kernels as host
 * pointers, exactly as the relu2 group does, so the coherence cost that measured >=15x the
 * kernel simply does not arise. Per-expert slices of the pooled x/gate/up/y buffers keep
 * the arithmetic identical to the ungrouped path.
 *
 * Same guard rails as its twin: every expert must agree on device, format and dims, or the
 * call declines and the caller falls back per-expert. */
static int g_grp_evt=-1; static double g_grp_pack_ms=0, g_grp_h2d_ms=0, g_grp_bytes=0; static long long g_grp_chunks=0;
static double g_grp_thr_sum=0, g_grp_thr_max=0, g_grp_thr_cpu=0; static long long g_grp_nthr=0;
static long long g_grp_pin_experts=0, g_grp_all_experts=0, g_grp_last_count=0;
/* Groups that met the rows-per-expert crossover vs those too small to amortise staging.
 * A run that is mostly `skipped` is decode; mostly `staged` is prefill. */
static long long g_lres_staged=0, g_lres_skipped=0;

/* Every-200-chunks was the only reporting, and it made a comparison wrong: a run that ends
 * at chunk 340 last printed at 200, so "47.4 GB staged" was two thirds of a total being
 * read against the reader's complete 119-130 GB. That is where the "the reader drains 2.7x
 * what the pack stages" reading came from. Print the finished totals at exit so a number
 * taken from this line is the whole run. */
static void grp_evt_report(void){
    if(!g_grp_chunks)return;
    fprintf(stderr,"[group-evt FINAL] chunks=%lld staged=%.1f GB | pack %.2f s = %.2f GB/s "
            "| H2D %.2f s = %.2f GB/s | %lld experts/chunk "
            "| thr %lld/chunk elapsed-max %.2f s elapsed-sum %.2f s cpu-sum %.2f s "
            "| DMA-direct %lld/%lld experts (%.0f%%) | groups staged %lld / skipped %lld\n",
            g_grp_chunks,g_grp_bytes/1e9,
            g_grp_pack_ms/1e3,(g_grp_bytes/1e9)/(g_grp_pack_ms/1e3),
            g_grp_h2d_ms/1e3,(g_grp_bytes/1e9)/(g_grp_h2d_ms/1e3),g_grp_last_count,
            g_grp_nthr/g_grp_chunks,g_grp_thr_max/1e3,g_grp_thr_sum/1e3,g_grp_thr_cpu/1e3,
            g_grp_pin_experts,g_grp_all_experts,
            g_grp_all_experts?100.0*g_grp_pin_experts/g_grp_all_experts:0.0,g_lres_staged,g_lres_skipped);
}

extern "C" int coli_cuda_expert_group_nvfp4(ColiCudaTensor *const *gates,
        ColiCudaTensor *const *ups,ColiCudaTensor *const *downs,
        const int *rows,int count,float *y,const float *x){
    if(!gates||!ups||!downs||!rows||!x||!y||count<1)return 0;
    ColiCudaTensor *first=gates[0]; if(!first)return 0;
    int device=first->device,D=first->I,I=first->O,total=0;
    for(int c=0;c<count;c++){
        ColiCudaTensor *g=gates[c],*u=ups[c],*d=downs[c];
        if(!g||!u||!d||rows[c]<1||g->fmt!=5||u->fmt!=5||d->fmt!=5||
           g->device!=device||u->device!=device||d->device!=device||
           g->I!=D||g->O!=I||u->I!=D||u->O!=I||d->I!=I||d->O!=D)return 0;
        total+=rows[c];
    }
    DeviceContext *ctx=find_ctx(device);if(!select_ctx(ctx)||ctx->compute_major<7)return 0;
    std::lock_guard<std::mutex> _scratch_lk(scratch_mu(ctx));
    size_t xb=(size_t)total*D*sizeof(float),ib=(size_t)total*I*sizeof(float);
    if(!reserve(&ctx->x,&ctx->x_cap,xb)||!reserve(&ctx->gate,&ctx->gate_cap,ib)||
       !reserve(&ctx->up,&ctx->up_cap,ib)||!reserve(&ctx->y,&ctx->y_cap,xb)||
       !reserve_pinned(&ctx->host_x,&ctx->host_x_cap,xb)||
       !reserve_pinned(&ctx->host_y,&ctx->host_y_cap,xb))return 0;
    std::memcpy(ctx->host_x,x,xb);
    if(!cuda_ok(cudaMemcpyAsync(ctx->x,ctx->host_x,xb,cudaMemcpyHostToDevice,ctx->stream),
                               "expert group nvfp4 input upload"))return 0;
    static int s_tiled=-1;
    if(s_tiled<0){const char*e=getenv("COLI_NVFP4_TILED");s_tiled=e&&atoi(e);}

    /* Per-layer bulk residency (COLI_LAYER_RESIDENT, default ON here).
     *
     * The whole group's expert weights are packed into ONE pinned host buffer and sent to
     * ONE device arena in a single transfer; the kernels below then read device memory.
     *
     * This is the option the recorded evidence left open, and it is NOT what
     * `COLI_FFN_DEVCOPY` does. That stages one expert at a time out of PAGEABLE memory, so
     * the driver bounces every transfer through its own staging buffer — measured at
     * 1.0-2.2 GB/s and 93.6% of expert-call GPU time (36.91s of staging against 2.51s of
     * kernel on M2.7). Pinned memory removes the bounce, and one transfer per group
     * removes the per-expert launch/latency floor that made the small copies so poor.
     *
     * Why this rather than grouping or a segmented GEMM: both were measured and neither
     * moved it. Grouping's whole ceiling is the ~3% round-trip; the segmented GEMM is
     * described in its own doc as the disproof of the occupancy theory. Every one of those
     * measurements pointed at the weight path — ~47 GB per prefill at ~6 GB/s against a
     * ~51 GB/s ceiling — which is what this addresses and they did not.
     *
     * Declines silently (leaving host pointers, i.e. the previous behaviour) if the arena
     * or the pinned buffer cannot be reserved, so a large layer degrades instead of failing. */
    static int s_lres=-1;
    if(s_lres<0){const char*e=getenv("COLI_LAYER_RESIDENT");s_lres=(!e||atoi(e));}

    /* Stage only when the rows AMORTISE it. Enabled unconditionally, this was a **2.02x
     * regression on serve**: decode routes one token, so a group stages ~190 MB of expert
     * weights to compute ONE ROW each. That can never pay, and it was invisible because the
     * path was built and measured on prefill.
     *
     * M2.7 serve, bench_serve 12 prompts x 32 tok, ABBA, pass1/pass2:
     *   staged always   2.85 / 3.08 tok/s
     *   staged never    5.20 / 6.31 tok/s   <- and 6.31 beats the 5.26 in the notes
     *
     * The threshold is bandwidth arithmetic, not a tuned number. With R rows per expert and
     * W bytes of weights, reading them zero-copy costs R*W/51 GB/s; staging costs the CPU
     * pack (W/40, measured) plus the H2D (W/56, measured) plus R*W/273 for the device reads.
     * Staging wins when R*(1/51 - 1/273) > 1/40 + 1/56, i.e. R > 2.7. Decode is R=1 and
     * loses; a 128-token prefill at top-8 over 256 experts is R=4 and wins, which is exactly
     * the split the two measurements show.
     *
     * Deliberately expressed per EXPERT, not per token or per phase: `rows` is what the
     * arithmetic is about, it is already known here, and it needs no caller to classify
     * itself — a phase flag would have to be plumbed through every arch and would be the
     * closed-set trap again. */
    const long lres_min_rows = 3;   // ceil of the R > 2.7 crossover above
    const bool stage_pays = total >= lres_min_rows*(long)count;
    if(stage_pays) g_lres_staged++; else g_lres_skipped++;
    bool resident=false;
    size_t wtot=0;
    if(s_lres&&stage_pays){
        for(int c=0;c<count;c++){
            size_t gnb=(size_t)I*((D+1)/2), dnb=(size_t)D*((I+1)/2);
            size_t gsb=(size_t)I*((D+15)/16), dsb=(size_t)D*((I+15)/16);
            wtot+=2*gnb+dnb+2*gsb+dsb;   // gate+up nibbles, down nibbles, and their scales
        }
        if(wtot&&reserve_bytes((void**)&ctx->lres,&ctx->lres_cap,wtot)&&
           reserve_pinned(&ctx->host_lres,&ctx->host_lres_cap,wtot)){
            /* COLI_PACK_PROBE=1: one-shot A/B inside the real pack, at the real moment, in
             * the real process, on one thread. Same byte count, same pinned destination,
             * only the SOURCE differs: the engine's own expert buffer versus a private
             * malloc. A standalone reproduction of this copy runs at 64 GB/s while the pack
             * manages 6, and every property of the copy itself is eliminated (scattered
             * sources, pinned destination, thread count, buffer reallocation), so the source
             * mapping is what is left to test — and it is the one thing a standalone
             * benchmark cannot reproduce.
             *
             * Each source is copied twice. A cold-then-warm pair separates first-touch fault
             * cost from steady-state read cost: slow-then-fast is faults, slow-then-slow with
             * a fast malloc source is the mapping itself. */
            static int s_probe=-1;
            if(s_probe<0){const char*e=getenv("COLI_PACK_PROBE");s_probe=e&&atoi(e);}
            if(s_probe==1&&count>0){
                s_probe=2;                       // one shot; it perturbs the chunk it runs in
                size_t n=(size_t)I*((D+1)/2);
                uint8_t *scratch=(uint8_t*)malloc(n);
                if(scratch){
                    memset(scratch,0x5a,n);      // pre-fault, so this is steady-state reads
                    uint8_t *dst=(uint8_t*)ctx->host_lres;
                    const uint8_t *e0=(const uint8_t*)gates[0]->weights;
                    double t[4]; auto mark=std::chrono::steady_clock::now();
                    #define PROBE(i,src) std::memcpy(dst,(src),n); { auto z=std::chrono::steady_clock::now(); \
                        t[i]=std::chrono::duration<double>(z-mark).count(); mark=z; }
                    PROBE(0,e0)                  // expert source, cold
                    PROBE(1,e0)                  // expert source, warm — same pages
                    PROBE(2,scratch)             // private source, warm
                    PROBE(3,(const uint8_t*)ups[0]->weights)   // a second expert source, cold
                    #undef PROBE
                    fprintf(stderr,"[pack-probe] %.1f MB/copy | expert cold %.2f warm %.2f "
                            "| malloc %.2f | expert2 cold %.2f  (GB/s)\n",
                            n/1e6,(n/1e9)/t[0],(n/1e9)/t[1],(n/1e9)/t[2],(n/1e9)/t[3]);
                    free(scratch);
                }
            }
            auto t_pack=std::chrono::steady_clock::now();
            uint8_t *hp=(uint8_t*)ctx->host_lres;
            size_t gnb=(size_t)I*((D+1)/2), dnb=(size_t)D*((I+1)/2);
            size_t gsb=(size_t)I*((D+15)/16), dsb=(size_t)D*((I+15)/16);
            size_t per=2*gnb+dnb+2*gsb+dsb;   // fixed stride: every expert has the same dims
            /* PARALLEL pack. This memcpy is the whole of rule 3's gap: measured
             * single-threaded at 37.58 s / 1.26 GB/s per prefill while the H2D that follows
             * it runs at 56.49 GB/s — already above the ~51 GB/s zero-copy ceiling. So the
             * transfer was never the problem and neither were the kernels; the bottleneck
             * is a CPU copy whose SOURCES are scattered pool and mmap pages, which is why
             * 1.26 GB/s is slow even for one core: it is fault- and latency-bound, not
             * bandwidth-bound, and that is exactly the shape that parallelises.
             * Each expert writes a disjoint `per`-byte slot, so no synchronisation. */
            /* Per-expert route selection, and the reason this is asked of the POINTER rather
             * than of the model: an expert's bytes reach us either as a recycled pool
             * allocation or as a view into an mmap of the container, and which one it is
             * varies by model, by coverage, and within a single run. Enumerating the models
             * that "use heap buffers" is exactly the closed-set mistake that has silently
             * skipped a model three times in this file. cudaPointerGetAttributes knows.
             *
             * A source the driver has page-locked can be DMA'd from directly, so that expert
             * needs no CPU copy at all. An unregistered source still has to be packed, because
             * a copy out of pageable memory bounces through the driver's staging buffer —
             * that is what made COLI_GROUP_DIRECT 2.3x slower, and it is a property of the
             * memory, not of the idea. Mixed groups are normal and both routes land in the
             * same device arena slot, so they compose.
             *
             * COLI_PIN_DIRECT=0 forces everything back through the pack. */
            static int s_pindirect=-1;
            if(s_pindirect<0){const char*e=getenv("COLI_PIN_DIRECT");s_pindirect=(!e||atoi(e));}
            std::vector<uint8_t> pinned((size_t)count,0);
            int npin=0;
            if(s_pindirect){
                for(int c=0;c<count;c++){
                    const void *sp[6]={gates[c]->weights,ups[c]->weights,downs[c]->weights,
                                       gates[c]->bscale,ups[c]->bscale,downs[c]->bscale};
                    int all=1;
                    for(int k=0;k<6&&all;k++){
                        cudaPointerAttributes a{};
                        if(cudaPointerGetAttributes(&a,sp[k])!=cudaSuccess){cudaGetLastError();all=0;}
                        else if(a.type!=cudaMemoryTypeHost)all=0;
                    }
                    pinned[(size_t)c]=(uint8_t)all; npin+=all;
                }
            }
            g_grp_pin_experts+=npin; g_grp_all_experts+=count;

            /* COLI_PACK_THREADS overrides the fan-out. hardware_concurrency() is the wrong
             * default here and the counters say why: the pack's threads spend 85% of their
             * life off-CPU (10.5 s of CPU against 68.9 s of elapsed), queued behind the
             * expert reader's own pool on the same 20 cores. Threads added to a saturated
             * runqueue do not copy faster, they just take the reader's slots. */
            static int s_pthr=-1;
            if(s_pthr<0){const char*e=getenv("COLI_PACK_THREADS");s_pthr=e?atoi(e):0;}
            unsigned nthr=s_pthr>0?(unsigned)s_pthr:std::thread::hardware_concurrency();
            if(!nthr)nthr=8;
            if(nthr>(unsigned)count)nthr=(unsigned)count;
            /* Per-thread busy time against the pack's wall time. A single-threaded copy from a
             * real expert source measures 26-30 GB/s in this same process, so 13 threads
             * should retire a 237 MB chunk in well under a millisecond; the chunk takes ~75.
             * Only three shapes can produce that, and these two counters separate them: if
             * max-thread ~= wall the threads are genuinely busy and the copies are slow in
             * aggregate (contention or faults); if max-thread << wall the time is in spawn,
             * join, or outside the copies entirely. */
            std::atomic<double> thr_sum{0.0}, thr_max{0.0}, thr_cpu{0.0};
            auto pack_range=[&](int lo,int hi){
                auto t_thr=std::chrono::steady_clock::now();
                /* Thread CPU time beside thread elapsed. The copies retire 30x slower in
                 * aggregate than the same copy does alone in this process, and only two
                 * things do that: the thread is off-CPU (oversubscribed against the reader's
                 * pool) or it is on-CPU and stalled on memory. cpu << elapsed is the first,
                 * cpu ~= elapsed the second — and they want different fixes, so guessing
                 * between them is what this avoids. */
                timespec c0,c1; clock_gettime(CLOCK_THREAD_CPUTIME_ID,&c0);
                for(int c=lo;c<hi;c++){
                    if(pinned[(size_t)c])continue;   // DMA'd straight from its source below
                    uint8_t *dst=hp+(size_t)c*per;
                    const uint8_t *src[6]={(const uint8_t*)gates[c]->weights,(const uint8_t*)ups[c]->weights,
                                           (const uint8_t*)downs[c]->weights,(const uint8_t*)gates[c]->bscale,
                                           (const uint8_t*)ups[c]->bscale,(const uint8_t*)downs[c]->bscale};
                    size_t len[6]={gnb,gnb,dnb,gsb,gsb,dsb};
                    size_t at=0;
                    for(int k=0;k<6;k++){ std::memcpy(dst+at,src[k],len[k]); at+=len[k]; }
                }
                clock_gettime(CLOCK_THREAD_CPUTIME_ID,&c1);
                double cpu=(c1.tv_sec-c0.tv_sec)*1e3+(c1.tv_nsec-c0.tv_nsec)/1e6;
                double q=thr_cpu.load(std::memory_order_relaxed);
                while(!thr_cpu.compare_exchange_weak(q,q+cpu,std::memory_order_relaxed)){}
                double e=std::chrono::duration<double,std::milli>(
                    std::chrono::steady_clock::now()-t_thr).count();
                double s=thr_sum.load(std::memory_order_relaxed);   // fetch_add on a double is C++20
                while(!thr_sum.compare_exchange_weak(s,s+e,std::memory_order_relaxed)){}
                double m=thr_max.load(std::memory_order_relaxed);
                while(e>m&&!thr_max.compare_exchange_weak(m,e,std::memory_order_relaxed)){}
            };
            /* COLI_GROUP_DIRECT=1: skip the pinned intermediate and issue the copies
             * straight from the source host pointers into the arena slots. Trades the CPU
             * pack (5.89 GB/s, fault-bound on scattered pool/mmap pages) for 6 async
             * transfers per expert out of PAGEABLE memory. Per-expert devcopy measured
             * 1-2 GB/s doing that, but it also synchronised per expert; here 150+ copies
             * are queued before anything waits, so the driver can pipeline them. Worth an
             * A/B precisely because the recorded evidence does not settle it. */
            static int s_direct=-1;
            if(s_direct<0){const char*e=getenv("COLI_GROUP_DIRECT");s_direct=e&&atoi(e);}
            if(s_direct){
                for(int c=0;c<count;c++){
                    uint8_t *dst=(uint8_t*)ctx->lres+(size_t)c*per;
                    const uint8_t *src[6]={(const uint8_t*)gates[c]->weights,(const uint8_t*)ups[c]->weights,
                                           (const uint8_t*)downs[c]->weights,(const uint8_t*)gates[c]->bscale,
                                           (const uint8_t*)ups[c]->bscale,(const uint8_t*)downs[c]->bscale};
                    size_t len[6]={gnb,gnb,dnb,gsb,gsb,dsb};
                    size_t at=0;
                    for(int k=0;k<6;k++){
                        cudaMemcpyAsync(dst+at,src[k],len[k],cudaMemcpyHostToDevice,ctx->stream);
                        at+=len[k];
                    }
                }
            }
            else if(npin>=count){ /* nothing to pack — every source is DMA-able */ }
            else if(nthr<=1){ pack_range(0,count); }
            else{
                std::vector<std::thread> th; th.reserve(nthr);
                int span=(count+(int)nthr-1)/(int)nthr;
                for(unsigned t=0;t<nthr;t++){
                    int lo=(int)t*span, hi=lo+span>count?count:lo+span;
                    if(lo>=hi)break;
                    th.emplace_back(pack_range,lo,hi);
                }
                for(auto &j:th) j.join();
            }
            double pack_ms=std::chrono::duration<double,std::milli>(
                std::chrono::steady_clock::now()-t_pack).count();
            auto t_h2d=std::chrono::steady_clock::now();
            if(s_direct){
                resident=cuda_ok(cudaGetLastError(),"expert group nvfp4 direct upload");
            }else if(npin>0){
                /* Mixed group: one transfer per expert, from whichever side that expert's
                 * bytes are already on. Kept off the npin==0 path deliberately — splitting
                 * one `wtot` transfer into `count` of them is a few percent of launch
                 * overhead for no gain when nothing is pinned, and this path is on by
                 * default, so it must cost nothing in the case where it can't help. */
                for(int c=0;c<count;c++){
                    uint8_t *dst=(uint8_t*)ctx->lres+(size_t)c*per;
                    if(!pinned[(size_t)c]){
                        cudaMemcpyAsync(dst,(const uint8_t*)ctx->host_lres+(size_t)c*per,per,
                                        cudaMemcpyHostToDevice,ctx->stream);
                        continue;
                    }
                    const uint8_t *src[6]={(const uint8_t*)gates[c]->weights,(const uint8_t*)ups[c]->weights,
                                           (const uint8_t*)downs[c]->weights,(const uint8_t*)gates[c]->bscale,
                                           (const uint8_t*)ups[c]->bscale,(const uint8_t*)downs[c]->bscale};
                    size_t len[6]={gnb,gnb,dnb,gsb,gsb,dsb};
                    size_t at=0;
                    for(int k=0;k<6;k++){
                        cudaMemcpyAsync(dst+at,src[k],len[k],cudaMemcpyHostToDevice,ctx->stream);
                        at+=len[k];
                    }
                }
                resident=cuda_ok(cudaGetLastError(),"expert group nvfp4 mixed upload");
            }else{
                resident=cuda_ok(cudaMemcpyAsync(ctx->lres,ctx->host_lres,wtot,
                                     cudaMemcpyHostToDevice,ctx->stream),
                                 "expert group nvfp4 layer-resident upload");
            }
            /* COLI_GROUP_EVT=1: split the group's transfer from its kernels. Without this
             * the grouped path reports gpu-ffn=0 — it has no counter at all — so the one
             * number rule 3 needs (are we transfer-bound or kernel-bound at 1.26 GB/s
             * against a ~51 GB/s ceiling?) could not be obtained. Host-side memcpy into the
             * pinned buffer is timed too: it is single-threaded and ~105 GB per prefill, so
             * it is a candidate in its own right. */
            if(g_grp_evt<0){const char*e=getenv("COLI_GROUP_EVT");g_grp_evt=e&&atoi(e);}
            if(g_grp_evt){
                static bool s_atexit=false;
                if(!s_atexit){s_atexit=true;atexit(grp_evt_report);}
                g_grp_last_count=count;
                cudaStreamSynchronize(ctx->stream);
                double h2d_ms=std::chrono::duration<double,std::milli>(
                    std::chrono::steady_clock::now()-t_h2d).count();
                g_grp_pack_ms+=pack_ms; g_grp_h2d_ms+=h2d_ms;
                g_grp_bytes+=(double)wtot; g_grp_chunks++;
                g_grp_thr_sum+=thr_sum.load(std::memory_order_relaxed);
                g_grp_thr_max+=thr_max.load(std::memory_order_relaxed);
                g_grp_thr_cpu+=thr_cpu.load(std::memory_order_relaxed);
                g_grp_nthr+=nthr;
                if((g_grp_chunks%200)==0)
                    fprintf(stderr,"[group-evt] chunks=%lld staged=%.1f GB | pack %.2f s = %.2f GB/s "
                            "| H2D %.2f s = %.2f GB/s | %d experts/chunk | pinned %lld hit/%lld alloc "
                            "| thr %lld/chunk elapsed-max %.2f s elapsed-sum %.2f s cpu-sum %.2f s "
                            "| DMA-direct %lld/%lld experts (%.0f%%) | groups staged %lld / skipped %lld\n",
                            g_grp_chunks,g_grp_bytes/1e9,
                            g_grp_pack_ms/1e3,(g_grp_bytes/1e9)/(g_grp_pack_ms/1e3),
                            g_grp_h2d_ms/1e3,(g_grp_bytes/1e9)/(g_grp_h2d_ms/1e3),count,g_res_pin_hit,g_res_pin_alloc,
                            g_grp_nthr/g_grp_chunks,g_grp_thr_max/1e3,g_grp_thr_sum/1e3,g_grp_thr_cpu/1e3,
                            g_grp_pin_experts,g_grp_all_experts,
                            g_grp_all_experts?100.0*g_grp_pin_experts/g_grp_all_experts:0.0,g_lres_staged,g_lres_skipped);
            }
        }
    }

    int off=0;
    size_t rat=0;   // running offset into the device arena, in the same order it was packed
    for(int c=0;c<count;c++){
        int r=rows[c];
        const uint8_t *gw=(const uint8_t*)gates[c]->weights,*uw=(const uint8_t*)ups[c]->weights,
                      *dw=(const uint8_t*)downs[c]->weights;
        const uint8_t *gbs=(const uint8_t*)gates[c]->bscale,*ubs=(const uint8_t*)ups[c]->bscale,
                      *dbs=(const uint8_t*)downs[c]->bscale;
        if(resident){
            size_t gnb=(size_t)I*((D+1)/2), dnb=(size_t)D*((I+1)/2);
            size_t gsb=(size_t)I*((D+15)/16), dsb=(size_t)D*((I+15)/16);
            const uint8_t *base=(const uint8_t*)ctx->lres+rat;
            gw=base;            base+=gnb;
            uw=base;            base+=gnb;
            dw=base;            base+=dnb;
            gbs=base;           base+=gsb;
            ubs=base;           base+=gsb;
            dbs=base;
            rat+=2*gnb+dnb+2*gsb+dsb;
        }
        float gg=gates[c]->gscale,ug=ups[c]->gscale,dg=downs[c]->gscale;
        // This expert's slice of the pooled buffers: rows [off, off+r).
        float *xc=ctx->x+(size_t)off*D,*gc=ctx->gate+(size_t)off*I,
              *uc=ctx->up+(size_t)off*I,*yc=ctx->y+(size_t)off*D;
        if(r==1&&!s_tiled){
            nvfp4_gemv_dispatch(gc,xc,gw,gbs,gg,D,I,ctx->stream);
            nvfp4_gemv_dispatch(uc,xc,uw,ubs,ug,D,I,ctx->stream);
            act_mul(gc,uc,(size_t)I,ctx->stream);
            nvfp4_gemv_dispatch(yc,gc,dw,dbs,dg,I,D,ctx->stream);
        }else{
            // Weight-stationary for the small-M rows a routed expert actually sees;
            // `nvfp4_wsmm_launch` declines above S=32 and the WMMA tile takes over.
            bool did_ws=false;
            if(nvfp4_wsmm_launch(gc,xc,gw,gbs,gg,r,D,I,ctx->stream)){
                nvfp4_wsmm_launch(uc,xc,uw,ubs,ug,r,D,I,ctx->stream);
                act_mul(gc,uc,(size_t)r*I,ctx->stream);
                did_ws=nvfp4_wsmm_launch(yc,gc,dw,dbs,dg,r,I,D,ctx->stream);
            }
            if(!did_ws){
                dim3 hidden((unsigned)((I+63)/64),(unsigned)((r+15)/16));
                dim3 output((unsigned)((D+63)/64),(unsigned)((r+15)/16));
                nvfp4_gate_up<<<hidden,256,0,ctx->stream>>>(gc,uc,xc,gw,uw,gbs,ubs,gg,ug,r,D,I);
                act_mul(gc,uc,(size_t)r*I,ctx->stream);
                nvfp4_matmul<<<output,128,0,ctx->stream>>>(yc,gc,dw,dbs,dg,r,I,D);
            }
        }
        off+=r;
    }
    if(!cuda_ok(cudaGetLastError(),"expert group nvfp4 launch")||
       !cuda_ok(cudaMemcpyAsync(ctx->host_y,ctx->y,xb,cudaMemcpyDeviceToHost,ctx->stream),
                               "expert group nvfp4 output download")||
       !cuda_ok(cudaStreamSynchronize(ctx->stream),"expert group nvfp4 synchronize"))return 0;
    std::memcpy(y,ctx->host_y,xb);
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
    std::lock_guard<std::mutex> _scratch_lk(scratch_mu(dc));
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
    // Static helper: reached only from the public absorb_batch / project_batch wrappers,
    // neither of which locks — so this is the single lock point for both.
    std::lock_guard<std::mutex> _scratch_lk(scratch_mu(dc));
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

/* Standard GQA prefill on the GPU (MiniMax-M3): q[S,H,D], full k/v[T,Hkv,D], ctx[S,H,D]
 * out. Reuses the attention scratch (aq=q, al=k, ar=v, ac=ctx). Caller's layouts match
 * directly (q is [S,H,D]; a KV cache row is [Hkv*D]; ctx is [S,H,D]). */
extern "C" int coli_cuda_gqa_attn(int device, float *ctx, const float *q, const float *k,
        const float *v, int S, int H, int Hkv, int D, int T, float scale, int mode) {
    if (!ctx || !q || !k || !v || S < 1 || H < 1 || Hkv < 1 || D < 1 || D > 1024 ||
        H % Hkv || T < S)
        return 0;
    // mode 1 = WMMA flash (tc_gqa_attn); requires D a multiple of 16. Anything else,
    // or D not tile-aligned, falls back to the scalar gqa_attn_kernel (mode 0).
    int flash = (mode == 1 && D % 16 == 0);
    // The T ceiling is PER-PATH, because only the scalar kernel pays for T in shared memory:
    //   scalar: shared = (D + T + ATTN_TPB)*4 — 33 KB at T=8192, and past the 48 KB limit by
    //           T=16384. For that kernel 8192 is a real hardware bound.
    //   flash:  shared = GQA_QT*8*D — INDEPENDENT of T. It tiles over keys with an online
    //           softmax (mrow/lrow/corr); handling long context is the entire point of it.
    // These shared a single blanket `T > 8192`, so the guard written for the scalar kernel
    // silently refused the one path built for long context — `return 0` is indistinguishable
    // from "no GPU", and the caller drops to the single-threaded CPU core. Measured on M2.7
    // (#54): 8192 tokens ran the GPU core, 16384 fell off a cliff, and 113851 never finished.
    // Flash's real bound is the grid.y hardware limit — one block per GQA_QT queries.
    // Past it, and on scratch-allocation failure below, the CPU fallback still stands.
    if (flash) {
        if ((S + GQA_QT - 1) / GQA_QT > 65535) return 0;
    } else if (T > 8192) {
        return 0;
    }
    DeviceContext *dc = find_ctx(device); if (!select_ctx(dc)) return 0;
    std::lock_guard<std::mutex> _scratch_lk(scratch_mu(dc));
    size_t qb = (size_t)S * H * D * sizeof(float), kb = (size_t)T * Hkv * D * sizeof(float);
    if (!reserve(&dc->aq, &dc->aq_cap, qb) || !reserve(&dc->al, &dc->al_cap, kb) ||
        !reserve(&dc->ar, &dc->ar_cap, kb) || !reserve(&dc->ac, &dc->ac_cap, qb))
        return 0;
    if (!cuda_ok(cudaMemcpyAsync(dc->aq, q, qb, cudaMemcpyHostToDevice, dc->stream), "gqa q upload") ||
        !cuda_ok(cudaMemcpyAsync(dc->al, k, kb, cudaMemcpyHostToDevice, dc->stream), "gqa k upload") ||
        !cuda_ok(cudaMemcpyAsync(dc->ar, v, kb, cudaMemcpyHostToDevice, dc->stream), "gqa v upload"))
        return 0;
    if (flash) {
        size_t shW = (size_t)GQA_QT * 8 * D;   // QA+KB (fp16, 4D) + acc (f32, 4D) per 16 rows
        if (!cuda_ok(cudaFuncSetAttribute(tc_gqa_attn, cudaFuncAttributeMaxDynamicSharedMemorySize, (int)shW), "gqa flash shared attr")) return 0;
        tc_gqa_attn<<<dim3(H, (S + GQA_QT - 1) / GQA_QT), 256, shW, dc->stream>>>(dc->ac, dc->aq, dc->al, dc->ar, S, H, Hkv, D, T, scale);
    } else {
        size_t shared = (size_t)(D + T + ATTN_TPB) * sizeof(float);
        gqa_attn_kernel<<<dim3(H, S), ATTN_TPB, shared, dc->stream>>>(dc->ac, dc->aq, dc->al, dc->ar, S, H, Hkv, D, T, scale);
    }
    if (!cuda_ok(cudaGetLastError(), "gqa launch")) return 0;
    if (!cuda_ok(cudaMemcpyAsync(ctx, dc->ac, qb, cudaMemcpyDeviceToHost, dc->stream), "gqa ctx download") ||
        !cuda_ok(cudaStreamSynchronize(dc->stream), "gqa sync"))
        return 0;
    return 1;
}

/* Nemotron-H Mamba2 selective-scan for one decode token (S==1). Uploads the persisted
 * ssm state + this token's post-conv h/B/C and the host-precomputed per-head dt_h/dA_h/D,
 * runs mamba2_scan_kernel (block per head, thread per head-dim row), then downloads the
 * updated state and the scan output y. Bit-identical twin of the CPU `selective_scan`. */
extern "C" int coli_cuda_mamba2_scan(int device, float *state, float *y, const float *hidden,
        const float *b, const float *c, const float *dt_h, const float *da_h,
        const float *d, int nh, int hd, int ds, int ng) {
    if (!state || !y || !hidden || !b || !c || !dt_h || !da_h || !d) return 0;
    if (nh < 1 || hd < 1 || hd > 1024 || ds < 1 || ng < 1 || nh % ng) return 0;
    DeviceContext *dc = find_ctx(device); if (!select_ctx(dc)) return 0;
    std::lock_guard<std::mutex> _scratch_lk(scratch_mu(dc));
    cudaStream_t st = dc->stream;
    size_t stt = (size_t)nh * hd * ds * sizeof(float);
    size_t xb  = (size_t)nh * hd * sizeof(float);
    size_t bcb = (size_t)ng * ds * sizeof(float);
    size_t hb  = (size_t)nh * sizeof(float);
    if (!reserve(&dc->ms_state, &dc->ms_state_cap, stt) ||
        !reserve(&dc->ms_x, &dc->ms_x_cap, xb) ||
        !reserve(&dc->ms_y, &dc->ms_y_cap, xb) ||
        !reserve(&dc->ms_b, &dc->ms_b_cap, bcb) ||
        !reserve(&dc->ms_c, &dc->ms_c_cap, bcb) ||
        !reserve(&dc->ms_dth, &dc->ms_dth_cap, hb) ||
        !reserve(&dc->ms_dah, &dc->ms_dah_cap, hb) ||
        !reserve(&dc->ms_d, &dc->ms_d_cap, hb))
        return 0;
    if (!cuda_ok(cudaMemcpyAsync(dc->ms_state, state, stt, cudaMemcpyHostToDevice, st), "mamba state up") ||
        !cuda_ok(cudaMemcpyAsync(dc->ms_x, hidden, xb, cudaMemcpyHostToDevice, st), "mamba x up") ||
        !cuda_ok(cudaMemcpyAsync(dc->ms_b, b, bcb, cudaMemcpyHostToDevice, st), "mamba b up") ||
        !cuda_ok(cudaMemcpyAsync(dc->ms_c, c, bcb, cudaMemcpyHostToDevice, st), "mamba c up") ||
        !cuda_ok(cudaMemcpyAsync(dc->ms_dth, dt_h, hb, cudaMemcpyHostToDevice, st), "mamba dth up") ||
        !cuda_ok(cudaMemcpyAsync(dc->ms_dah, da_h, hb, cudaMemcpyHostToDevice, st), "mamba dah up") ||
        !cuda_ok(cudaMemcpyAsync(dc->ms_d, d, hb, cudaMemcpyHostToDevice, st), "mamba d up"))
        return 0;
    mamba2_scan_kernel<<<dim3(nh), hd, 0, st>>>(dc->ms_state, dc->ms_y, dc->ms_x,
        dc->ms_b, dc->ms_c, dc->ms_dth, dc->ms_dah, dc->ms_d, nh, hd, ds, ng);
    if (!cuda_ok(cudaGetLastError(), "mamba launch")) return 0;
    if (!cuda_ok(cudaMemcpyAsync(state, dc->ms_state, stt, cudaMemcpyDeviceToHost, st), "mamba state down") ||
        !cuda_ok(cudaMemcpyAsync(y, dc->ms_y, xb, cudaMemcpyDeviceToHost, st), "mamba y download") ||
        !cuda_ok(cudaStreamSynchronize(st), "mamba sync"))
        return 0;
    return 1;
}

/* Whole-sequence (prefill, S>1) twin of `coli_cuda_mamba2_scan`. Same uploads with the
 * per-token axis added — hidden/y are [seq,nh*hd], B/C [seq,ng,ds], dt_h/dA_h [seq,nh]
 * (all precomputed host-side so the softplus/exp stay bit-identical). One block per
 * head, hd threads, the head's state resident in shared memory for the whole scan.
 * Declines (returns 0, caller falls back to the CPU scan) if the state does not fit in
 * shared memory rather than silently launching something slower or wrong. */
extern "C" int coli_cuda_mamba2_scan_seq(int device, float *state, float *y, const float *hidden,
        const float *b, const float *c, const float *dt_h, const float *da_h,
        const float *d, int nh, int hd, int ds, int ng, int seq, int exact) {
    if (!state || !y || !hidden || !b || !c || !dt_h || !da_h || !d) return 0;
    if (nh < 1 || hd < 1 || ds < 1 || ds > 1024 || ng < 1 || nh % ng || seq < 1) return 0;
    DeviceContext *dc = find_ctx(device); if (!select_ctx(dc)) return 0;
    std::lock_guard<std::mutex> _scratch_lk(scratch_mu(dc));
    cudaStream_t st = dc->stream;
    // Third shared region: tree needs <=32 warp partials; exact needs prod[ds].
    size_t red_slots = exact ? (size_t)ds : 32;
    size_t shmem = (2 * (size_t)ds + red_slots) * sizeof(float);
    int shmax = 0;
    if (!cuda_ok(cudaDeviceGetAttribute(&shmax, cudaDevAttrMaxSharedMemoryPerBlock, device),
                 "mamba seq shmem query"))
        return 0;
    if (shmem > (size_t)shmax) return 0;   // decline; caller uses the CPU scan
    // Staged, with PINNED host buffers. CUDA-event timing located the cost precisely:
    // the kernel is 255 ms for all 40 layers, while download+sync was 5761 ms — ~840 MB
    // of D2H at ~146 MB/s, because an async copy to a *pageable* destination degenerates
    // into a synchronous trickle through a small internal bounce buffer. Compute was
    // never the bottleneck; three earlier attempts (shared-memory staging, bank-conflict
    // padding, 128x more threads) each optimized the 255 ms and moved nothing.
    //
    // ⚠️ Zero-copy (kernel straight onto host pointers, as the expert path does) was
    // MEASURED WORSE: scan 7445 -> 13670 ms. Experts stream large buffers read once;
    // this scan re-reads B/C/hidden across every one of `seq` timesteps, so it pays the
    // coherent-link latency over and over. Do not retry it here.
    size_t sq  = (size_t)seq;
    size_t stt = (size_t)nh * hd * ds * sizeof(float);
    size_t xb  = sq * (size_t)nh * hd * sizeof(float);
    size_t bcb = sq * (size_t)ng * ds * sizeof(float);
    size_t hsb = sq * (size_t)nh * sizeof(float);
    size_t db  = (size_t)nh * sizeof(float);
    if (!reserve(&dc->ms_state, &dc->ms_state_cap, stt) ||
        !reserve(&dc->ms_x, &dc->ms_x_cap, xb) ||
        !reserve(&dc->ms_y, &dc->ms_y_cap, xb) ||
        !reserve(&dc->ms_b, &dc->ms_b_cap, bcb) ||
        !reserve(&dc->ms_c, &dc->ms_c_cap, bcb) ||
        !reserve(&dc->ms_dth, &dc->ms_dth_cap, hsb) ||
        !reserve(&dc->ms_dah, &dc->ms_dah_cap, hsb) ||
        !reserve(&dc->ms_d, &dc->ms_d_cap, db))
        return 0;
    if (!reserve_pinned(&dc->ms_pin_x, &dc->ms_pin_x_cap, xb) ||
        !reserve_pinned(&dc->ms_pin_y, &dc->ms_pin_y_cap, xb) ||
        !reserve_pinned(&dc->ms_pin_state, &dc->ms_pin_state_cap, stt))
        return 0;
    memcpy(dc->ms_pin_x, hidden, xb);
    memcpy(dc->ms_pin_state, state, stt);
    if (!cuda_ok(cudaMemcpyAsync(dc->ms_state, dc->ms_pin_state, stt, cudaMemcpyHostToDevice, st), "mamba seq state up") ||
        !cuda_ok(cudaMemcpyAsync(dc->ms_x, dc->ms_pin_x, xb, cudaMemcpyHostToDevice, st), "mamba seq x up") ||
        !cuda_ok(cudaMemcpyAsync(dc->ms_b, b, bcb, cudaMemcpyHostToDevice, st), "mamba seq b up") ||
        !cuda_ok(cudaMemcpyAsync(dc->ms_c, c, bcb, cudaMemcpyHostToDevice, st), "mamba seq c up") ||
        !cuda_ok(cudaMemcpyAsync(dc->ms_dth, dt_h, hsb, cudaMemcpyHostToDevice, st), "mamba seq dth up") ||
        !cuda_ok(cudaMemcpyAsync(dc->ms_dah, da_h, hsb, cudaMemcpyHostToDevice, st), "mamba seq dah up") ||
        !cuda_ok(cudaMemcpyAsync(dc->ms_d, d, db, cudaMemcpyHostToDevice, st), "mamba seq d up"))
        return 0;
    mamba2_scan_seq_kernel<<<dim3(hd, nh), ds, shmem, st>>>(dc->ms_state, dc->ms_y, dc->ms_x,
        dc->ms_b, dc->ms_c, dc->ms_dth, dc->ms_dah, dc->ms_d, nh, hd, ds, ng, seq, exact);
    if (!cuda_ok(cudaGetLastError(), "mamba seq launch")) return 0;
    if (!cuda_ok(cudaMemcpyAsync(dc->ms_pin_state, dc->ms_state, stt, cudaMemcpyDeviceToHost, st), "mamba seq state down") ||
        !cuda_ok(cudaMemcpyAsync(dc->ms_pin_y, dc->ms_y, xb, cudaMemcpyDeviceToHost, st), "mamba seq y download") ||
        !cuda_ok(cudaStreamSynchronize(st), "mamba seq sync"))
        return 0;
    memcpy(state, dc->ms_pin_state, stt);
    memcpy(y, dc->ms_pin_y, xb);
    return 1;
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
    std::lock_guard<std::mutex> _scratch_lk(scratch_mu(dc));
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
    DeviceContext *dc=find_ctx(w->device);if(!select_ctx(dc))return 0;
    // Lock before the tc branch: tc_sparse_attn_run uses the same per-device scratch and
    // does not lock itself (it runs nested under this guard).
    std::lock_guard<std::mutex> _scratch_lk(scratch_mu(dc));
    // Tensor-core WMMA path (opt-in). Only the non-fused case (no o_proj); ~3x the scalar core.
    { static int tc=-1; if(tc<0){const char*e=getenv("COLI_TC_ATTN");tc=(e&&atoi(e))?1:0;}
      if(tc && !proj) return tc_sparse_attn_run(w,out,q,latent,rope,sel_idx,sel_cnt,maxsel,H0,HC,S,H,Q,R,V,K,T,scale); }
    if(proj&&(proj->device!=w->device||proj->I!=H*V))return 0;
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
// `weights` stays in its on-disk layout (int8 is already signed). No cudaMalloc,
// no memcpy, no conversion, no device memory.
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

/* Zero-copy wrap of an MXFP4 expert weight (fmt=6): nibbles + E8M0 per-32 block scales.
 * Same shape as the NVFP4 wrap; the format code is what selects the per-32 stride and the
 * power-of-two scale decode in the kernels. `gscale` is 1.0 for a native MXFP4 tensor. */
extern "C" int coli_cuda_tensor_wrap_mxfp4(ColiCudaTensor **tensor,
        const void *weights, const void *bscale, float gscale,
        int I, int O, int device) {
    if (!tensor || !weights || !bscale || I < 1 || O < 1) return 0;
    if (*tensor) {
        ColiCudaTensor *t = *tensor;
        return t->fmt == 6 && t->I == I && t->O == O && t->device == device;
    }
    ColiCudaTensor *t = static_cast<ColiCudaTensor *>(std::calloc(1, sizeof(*t)));
    if (!t) return 0;
    t->fmt = 6; t->I = I; t->O = O; t->device = device;
    t->weight_bytes = row_bytes(6, I) * (size_t)O;
    t->weights = const_cast<void *>(weights);
    t->bscale = bscale;
    t->gscale = gscale;
    t->wrapped = 1;
    *tensor = t;
    return 1;
}

/* Zero-copy wrap of an NVFP4 expert weight (fmt=5): nibbles + ue4m3 block scales +
 * per-tensor global, all read from host RAM in place. See coli_cuda_expert_mlp_nvfp4. */
extern "C" int coli_cuda_tensor_wrap_nvfp4(ColiCudaTensor **tensor,
        const void *weights, const void *bscale, float gscale,
        int I, int O, int device) {
    if (!tensor || !weights || !bscale || I < 1 || O < 1) return 0;
    if (*tensor) {
        ColiCudaTensor *t = *tensor;
        return t->fmt == 5 && t->I == I && t->O == O && t->device == device;
    }
    ColiCudaTensor *t = static_cast<ColiCudaTensor *>(std::calloc(1, sizeof(*t)));
    if (!t) return 0;
    t->fmt = 5; t->I = I; t->O = O; t->device = device;
    t->weight_bytes = row_bytes(5, I) * (size_t)O;
    t->weights = const_cast<void *>(weights);
    t->bscale = bscale;
    t->gscale = gscale;
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
    act_mul(gate_dev,up_dev,n,0);
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

// ---------------------------------------------------------------------------
// DeepSeek-V4 sparse attention core.
//
// Measured at 48% of V4 decode (144 of 300 ms/token) as a scalar Rust loop, with
// `coli gen` reporting `0 attention cores` — this path had never touched the GPU.
//
// Contract matches `dsv4::attention_dsv4_sparse` exactly: `idxs` is [S,K] into `kv`,
// `-1` means masked, duplicate indices are MEANINGFUL (V4 attends to a compressed block
// that overlaps the raw window both ways), and the sink contributes to the DENOMINATOR
// only. One shared latent serves as both K and V, so every head reads the same rows.
//
// One block per (query, head). Scores are warp-per-key so the K-loop needs no
// __syncthreads; the reduction over `hd` is a shuffle within the warp.
#define DSV4_WARP 32
__global__ void dsv4_sparse_attn_kernel(
        const float *__restrict__ q, const float *__restrict__ kv,
        const float *__restrict__ sink, const int *__restrict__ idxs,
        int H, int D, int K, int rows, float scale, float *__restrict__ out) {
    extern __shared__ float sm[];
    float *qs = sm;          // [D]
    float *sc = sm + D;      // [K]
    float *red = sm + D + K; // [blockDim.x / WARP]

    const int i = blockIdx.x / H, hh = blockIdx.x % H;
    const int tid = threadIdx.x, nth = blockDim.x;
    const int lane = tid % DSV4_WARP, warp = tid / DSV4_WARP, nwarp = nth / DSV4_WARP;

    for (int d = tid; d < D; d += nth) qs[d] = q[((size_t)i * H + hh) * D + d];
    __syncthreads();

    const int *sel = idxs + (size_t)i * K;
    for (int t = warp; t < K; t += nwarp) {
        const int j = sel[t];
        if (j < 0 || j >= rows) { if (lane == 0) sc[t] = -INFINITY; continue; }
        const float *kr = kv + (size_t)j * D;
        float acc = 0.f;
        for (int d = lane; d < D; d += DSV4_WARP) acc += qs[d] * kr[d];
        #pragma unroll
        for (int o = DSV4_WARP / 2; o; o >>= 1) acc += __shfl_down_sync(0xffffffff, acc, o);
        if (lane == 0) sc[t] = acc * scale;
    }
    __syncthreads();

    // max, then exp/sum — a masked slot stays exactly 0 and adds nothing.
    float m = -INFINITY;
    for (int t = tid; t < K; t += nth) m = fmaxf(m, sc[t]);
    #pragma unroll
    for (int o = DSV4_WARP / 2; o; o >>= 1) m = fmaxf(m, __shfl_down_sync(0xffffffff, m, o));
    if (lane == 0) red[warp] = m;
    __syncthreads();
    if (tid == 0) { float g = -INFINITY; for (int w = 0; w < nwarp; ++w) g = fmaxf(g, red[w]); red[0] = g; }
    __syncthreads();
    m = red[0];

    float ssum = 0.f;
    for (int t = tid; t < K; t += nth) {
        float e = isfinite(sc[t]) ? __expf(sc[t] - m) : 0.f;
        sc[t] = e;
        ssum += e;
    }
    #pragma unroll
    for (int o = DSV4_WARP / 2; o; o >>= 1) ssum += __shfl_down_sync(0xffffffff, ssum, o);
    if (lane == 0) red[warp] = ssum;
    __syncthreads();
    if (tid == 0) {
        float g = 0.f;
        for (int w = 0; w < nwarp; ++w) g += red[w];
        // Sink: denominator only, stabilised against the same max the scores used.
        red[0] = 1.f / (g + __expf(sink[hh] - m));
    }
    __syncthreads();
    const float inv = red[0];

    // Gather. Each thread owns a set of dims, so reads are coalesced across the warp for
    // each key; duplicates in `sel` naturally accumulate twice, which is intended.
    for (int d = tid; d < D; d += nth) {
        float acc = 0.f;
        for (int t = 0; t < K; ++t) {
            const int j = sel[t];
            if (j >= 0 && j < rows && sc[t] != 0.f) acc += sc[t] * kv[(size_t)j * D + d];
        }
        out[((size_t)i * H + hh) * D + d] = acc * inv;
    }
}

extern "C" int coli_cuda_dsv4_sparse_attn(const float *q, const float *kv, const float *sink,
        const int *idxs, int S, int H, int D, int K, int rows, float scale, float *out) {
    if (!q || !kv || !sink || !idxs || !out || S < 1 || H < 1 || D < 1 || K < 1 || rows < 1) return 0;
    if ((long long)S * H > 2147483647LL) return 0;
    DeviceContext *dc = find_ctx(0);
    if (!select_ctx(dc)) return 0;
    int nth = 256;
    size_t shmem = ((size_t)D + K + nth / DSV4_WARP) * sizeof(float);
    // Bail rather than silently truncate — the caller's CPU path is correct, just slow.
    if (shmem > 48 * 1024) return 0;
    dsv4_sparse_attn_kernel<<<S * H, nth, shmem, dc->stream>>>(q, kv, sink, idxs, H, D, K, rows, scale, out);
    if (!cuda_ok(cudaGetLastError(), "dsv4 sparse attn launch")) return 0;
    if (!cuda_ok(cudaStreamSynchronize(dc->stream), "dsv4 sparse attn sync")) return 0;
    return 1;
}

// ---------------------------------------------------------------------------
// Grouped MXFP4 SwiGLU experts (DeepSeek-V4).
//
// V4 had no fmt-6 arm in the grouped dispatch chain, so every expert took
// `coli_cuda_expert_mlp_mxfp4` one at a time. Measured: **301 dispatches per decode
// token** (43 layers x 6 routed + 43 shared), 190 us each. The cost is not the four
// kernel launches — it is that EVERY call takes the scratch mutex, memcpys x into a
// pinned buffer, uploads, downloads, and does a full `cudaStreamSynchronize`, because
// `ctx->x/gate/up/y` is shared scratch that cannot overlap between calls.
//
// This hoists all of that out of the loop: one lock, one upload, one download, ONE sync
// for the whole group. The per-expert math kernels are unchanged — deliberately, so any
// output difference is a bug in this plumbing and not in the arithmetic.
//
// `act_mul` reads the activation globals, so V4's clamped SwiGLU (`swiglu_limit`) applies
// here exactly as it does on the per-expert path.
extern "C" int coli_cuda_expert_group_mxfp4(ColiCudaTensor *const *gates,
        ColiCudaTensor *const *ups, ColiCudaTensor *const *downs,
        const int *rows, int count, float *y, const float *x) {
    if (!gates || !ups || !downs || !rows || !x || !y || count < 1) return 0;
    ColiCudaTensor *first = gates[0];
    if (!first) return 0;
    int device = first->device, D = first->I, I = first->O, total = 0;
    for (int c = 0; c < count; c++) {
        ColiCudaTensor *g = gates[c], *u = ups[c], *d = downs[c];
        if (!g || !u || !d || rows[c] < 1 || g->fmt != 6 || u->fmt != 6 || d->fmt != 6 ||
            g->device != device || u->device != device || d->device != device ||
            g->I != D || g->O != I || u->I != D || u->O != I || d->I != I || d->O != D) return 0;
        total += rows[c];
    }
    DeviceContext *ctx = find_ctx(device);
    if (!select_ctx(ctx) || ctx->compute_major < 7) return 0;
    std::lock_guard<std::mutex> _scratch_lk(scratch_mu(ctx));
    size_t xb = (size_t)total * D * sizeof(float), ib = (size_t)total * I * sizeof(float);
    if (!reserve(&ctx->x, &ctx->x_cap, xb) || !reserve(&ctx->gate, &ctx->gate_cap, ib) ||
        !reserve(&ctx->up, &ctx->up_cap, ib) || !reserve(&ctx->y, &ctx->y_cap, xb) ||
        !reserve_pinned(&ctx->host_x, &ctx->host_x_cap, xb) ||
        !reserve_pinned(&ctx->host_y, &ctx->host_y_cap, xb)) return 0;
    std::memcpy(ctx->host_x, x, xb);
    if (!cuda_ok(cudaMemcpyAsync(ctx->x, ctx->host_x, xb, cudaMemcpyHostToDevice, ctx->stream),
                 "expert group mxfp4 upload")) return 0;
    int off = 0;
    for (int c = 0; c < count; c++) {
        const int S = rows[c];
        ColiCudaTensor *g = gates[c], *u = ups[c], *d = downs[c];
        const uint8_t *gw = (const uint8_t *)g->weights, *uw = (const uint8_t *)u->weights,
                      *dw = (const uint8_t *)d->weights;
        const uint8_t *gbs = (const uint8_t *)g->bscale, *ubs = (const uint8_t *)u->bscale,
                      *dbs = (const uint8_t *)d->bscale;
        float *xc = ctx->x + (size_t)off * D, *gc = ctx->gate + (size_t)off * I,
              *uc = ctx->up + (size_t)off * I, *yc = ctx->y + (size_t)off * D;
        if (S == 1) {
            mxfp4_gemv_dispatch(gc, xc, gw, gbs, g->gscale, D, I, ctx->stream);
            mxfp4_gemv_dispatch(uc, xc, uw, ubs, u->gscale, D, I, ctx->stream);
            act_mul(gc, uc, (size_t)I, ctx->stream);
            mxfp4_gemv_dispatch(yc, gc, dw, dbs, d->gscale, I, D, ctx->stream);
        } else if (mxfp4_wsmm_launch(gc, xc, gw, gbs, g->gscale, S, D, I, ctx->stream)) {
            // Weight-stationary below S=32; declines above and the WMMA tile takes over.
            mxfp4_wsmm_launch(uc, xc, uw, ubs, u->gscale, S, D, I, ctx->stream);
            act_mul(gc, uc, (size_t)S * I, ctx->stream);
            mxfp4_wsmm_launch(yc, gc, dw, dbs, d->gscale, S, I, D, ctx->stream);
        } else {
            dim3 hidden((unsigned)((I + 63) / 64), (unsigned)((S + 15) / 16));
            dim3 output((unsigned)((D + 63) / 64), (unsigned)((S + 15) / 16));
            mxfp4_gate_up<<<hidden, 256, 0, ctx->stream>>>(gc, uc, xc, gw, uw, gbs, ubs,
                                                           g->gscale, u->gscale, S, D, I);
            act_mul(gc, uc, (size_t)S * I, ctx->stream);
            mxfp4_matmul<<<output, 128, 0, ctx->stream>>>(yc, gc, dw, dbs, d->gscale, S, I, D);
        }
        off += S;
    }
    if (!cuda_ok(cudaGetLastError(), "expert group mxfp4 launch") ||
        !cuda_ok(cudaMemcpyAsync(ctx->host_y, ctx->y, xb, cudaMemcpyDeviceToHost, ctx->stream),
                 "expert group mxfp4 download") ||
        !cuda_ok(cudaStreamSynchronize(ctx->stream), "expert group mxfp4 synchronize")) return 0;
    std::memcpy(y, ctx->host_y, xb);
    return 1;
}
