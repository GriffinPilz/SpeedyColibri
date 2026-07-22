//! CUDA backend for Blackwell (DGX Spark GB10) — an FFI binding to the reference
//! `c/backend_cuda.cu` (declared in `c/backend_cuda.h`).
//!
//! Approach (per the roadmap): **bind the battle-tested `.cu` first, then port
//! kernels.** `build.rs` compiles `backend_cuda.cu` with `nvcc` when the `cuda`
//! feature is on and links `cudart` + `stdc++`; this module declares its
//! `extern "C"` surface and wraps the essential entry points safely.
//!
//! # Cannot be verified without hardware
//!
//! This binds real CUDA kernels; it can only be compiled with `nvcc` present and
//! only exercised on an NVIDIA GPU. On a host without CUDA, `build.rs` skips the
//! `nvcc` step (with a warning) so `cargo check --features cuda` still
//! type-checks this FFI, but the symbols won't link and nothing runs.
//!
//! Compute capability defaults to the C build's `-arch=native`; set
//! `CUDA_ARCH=sm_121` when cross-building for the DGX Spark GB10 (Blackwell).

use crate::{Backend, Device};
use std::os::raw::{c_int, c_void};

/// Opaque, persistent device copy of one resident quantized tensor
/// (`ColiCudaTensor` in the C ABI).
#[repr(C)]
pub struct ColiCudaTensor {
    _private: [u8; 0],
}

// The essential slice of backend_cuda.h. More entry points (attention absorb,
// resident-pipeline primitives) are bound as the forward pass is moved onto the
// GPU.
extern "C" {
    fn coli_cuda_init(devices: *const c_int, count: c_int) -> c_int;
    fn coli_cuda_shutdown();
    fn coli_cuda_device_count() -> c_int;
    fn coli_cuda_mem_info(device: c_int, free_bytes: *mut usize, total_bytes: *mut usize) -> c_int;

    fn coli_cuda_tensor_upload(
        tensor: *mut *mut ColiCudaTensor,
        weights: *const c_void,
        scales: *const f32,
        fmt: c_int,
        i: c_int,
        o: c_int,
        device: c_int,
    ) -> c_int;
    fn coli_cuda_tensor_wrap(
        tensor: *mut *mut ColiCudaTensor,
        weights: *const c_void,
        scales: *const f32,
        fmt: c_int,
        i: c_int,
        o: c_int,
        device: c_int,
    ) -> c_int;
    // Zero-copy wrap of an NVFP4 expert weight: `weights` = packed e2m1 nibbles
    // [O, ceil(I/2)], `bscale` = ue4m3 per-16 block scales [O, ceil(I/16)], `gscale` =
    // per-tensor global. Sets fmt=5.
    fn coli_cuda_tensor_wrap_nvfp4(
        tensor: *mut *mut ColiCudaTensor,
        weights: *const c_void,
        bscale: *const c_void,
        gscale: f32,
        i: c_int,
        o: c_int,
        device: c_int,
    ) -> c_int;
    fn coli_cuda_pageable_access(device: c_int) -> c_int;
    fn coli_cuda_matmul(
        tensor: *mut *mut ColiCudaTensor,
        y: *mut f32,
        x: *const f32,
        weights: *const c_void,
        scales: *const f32,
        fmt: c_int,
        s: c_int,
        i: c_int,
        o: c_int,
        device: c_int,
    ) -> c_int;
    fn coli_cuda_expert_mlp(
        gate: *mut ColiCudaTensor,
        up: *mut ColiCudaTensor,
        down: *mut ColiCudaTensor,
        y: *mut f32,
        x: *const f32,
        s: c_int,
    ) -> c_int;
    fn coli_cuda_expert_mlp_fp8(
        gate: *mut ColiCudaTensor,
        up: *mut ColiCudaTensor,
        down: *mut ColiCudaTensor,
        y: *mut f32,
        x: *const f32,
        s: c_int,
    ) -> c_int;
    fn coli_cuda_expert_mlp_i8a16(
        gate: *mut ColiCudaTensor,
        up: *mut ColiCudaTensor,
        down: *mut ColiCudaTensor,
        y: *mut f32,
        x: *const f32,
        s: c_int,
    ) -> c_int;
    fn coli_cuda_expert_mlp_nvfp4(
        gate: *mut ColiCudaTensor,
        up: *mut ColiCudaTensor,
        down: *mut ColiCudaTensor,
        y: *mut f32,
        x: *const f32,
        s: c_int,
    ) -> c_int;
    fn coli_cuda_expert_group(
        gates: *const *mut ColiCudaTensor,
        ups: *const *mut ColiCudaTensor,
        downs: *const *mut ColiCudaTensor,
        rows: *const c_int,
        count: c_int,
        y: *mut f32,
        x: *const f32,
    ) -> c_int;
    #[allow(clippy::too_many_arguments)]
    fn coli_cuda_attention_absorb_batch(
        kv_b: *mut ColiCudaTensor,
        ctx: *mut f32,
        q: *const f32,
        latent: *const f32,
        rope: *const f32,
        s: c_int,
        h: c_int,
        q_nope: c_int,
        r: c_int,
        v: c_int,
        k: c_int,
        t: c_int,
        scale: f32,
    ) -> c_int;
    // DSA lightning-indexer scores (the indexer's CPU hot loop, moved to the GPU).
    fn coli_cuda_dsa_indexer_scores(
        scores: *mut f32,
        qi: *const f32,
        hw: *const f32,
        keys: *const f32,
        nsp: c_int,
        s0: c_int,
        nh: c_int,
        hd: c_int,
        t: c_int,
        pos_base: c_int,
        device: c_int,
    ) -> c_int;
    // DSA sparse prefill absorb: each query attends only to its indexer selection.
    fn coli_cuda_attention_absorb_sparse(
        kv_b: *mut ColiCudaTensor,
        ctx: *mut f32,
        q: *const f32,
        latent: *const f32,
        rope: *const f32,
        sel_idx: *const c_int,
        sel_cnt: *const c_int,
        maxsel: c_int,
        h0: c_int,
        hc: c_int,
        s: c_int,
        h: c_int,
        q_nope: c_int,
        r: c_int,
        v: c_int,
        k: c_int,
        t: c_int,
        scale: f32,
    ) -> c_int;
    // Single-token (S=1) absorb with DEVICE latent/rope (the persistent-KV path).
    #[allow(clippy::too_many_arguments)]
    fn coli_cuda_attention_absorb_kvdev(
        kv_b: *mut ColiCudaTensor,
        ctx: *mut f32,
        q: *const f32,
        latent_dev: *const f32,
        rope_dev: *const f32,
        h: c_int,
        q_nope: c_int,
        r: c_int,
        v: c_int,
        k: c_int,
        t: c_int,
        scale: f32,
    ) -> c_int;
    // Device-memory pipeline primitives (for the KV shadow).
    fn coli_cuda_pipe_alloc(device: c_int, bytes: usize) -> *mut c_void;
    fn coli_cuda_pipe_free(device: c_int, p: *mut c_void);
    fn coli_cuda_pipe_upload(device: c_int, dst: *mut c_void, src: *const c_void, bytes: usize) -> c_int;
    fn coli_cuda_tensor_free(tensor: *mut ColiCudaTensor);
    fn coli_cuda_tensor_bytes(tensor: *const ColiCudaTensor) -> usize;
    fn coli_cuda_tensor_device(tensor: *const ColiCudaTensor) -> c_int;
}

// ---- safe wrappers ---------------------------------------------------------

/// Number of usable CUDA devices (0 if none / driver missing).
pub fn device_count() -> i32 {
    unsafe { coli_cuda_device_count() }
}

/// Initialize the given CUDA device ordinals. Returns whether init succeeded.
pub fn init(devices: &[i32]) -> bool {
    unsafe { coli_cuda_init(devices.as_ptr(), devices.len() as c_int) != 0 }
}

/// Release all CUDA resources.
pub fn shutdown() {
    unsafe { coli_cuda_shutdown() }
}

/// Whether `device` can read pageable host memory directly (coherent unified
/// memory like the GB10). When true, the zero-copy [`ResidentTensor::wrap_raw`]
/// path avoids copying weights into device memory entirely.
pub fn pageable_access(device: i32) -> bool {
    unsafe { coli_cuda_pageable_access(device) != 0 }
}

/// `(free, total)` device memory in bytes.
pub fn mem_info(device: i32) -> Option<(usize, usize)> {
    let mut free = 0usize;
    let mut total = 0usize;
    if unsafe { coli_cuda_mem_info(device, &mut free, &mut total) } != 0 {
        Some((free, total))
    } else {
        None
    }
}

/// A resident weight tensor on a CUDA device. Owns the device slot and frees it
/// on drop.
pub struct ResidentTensor {
    ptr: *mut ColiCudaTensor,
}

impl ResidentTensor {
    /// Upload a quantized weight `[O, I]` (fmt: 0=f32, 1=int8, 2=int4, 3=int2)
    /// to `device`, so a later matmul reuses it. `weights` is the raw code bytes.
    pub fn upload(
        weights: &[u8],
        scales: &[f32],
        fmt: i32,
        i: i32,
        o: i32,
        device: i32,
    ) -> Option<ResidentTensor> {
        let mut ptr: *mut ColiCudaTensor = std::ptr::null_mut();
        let ok = unsafe {
            coli_cuda_tensor_upload(
                &mut ptr,
                weights.as_ptr() as *const c_void,
                scales.as_ptr(),
                fmt,
                i,
                o,
                device,
            )
        };
        if ok != 0 && !ptr.is_null() {
            Some(ResidentTensor { ptr })
        } else {
            None
        }
    }

    /// Upload from raw pointers (no byte-slice reshaping). `weights` points at the
    /// CPU code bytes, `scales` at the `O` row scales.
    ///
    /// # Safety
    /// `weights`/`scales` must be valid for the tensor's `[O, I]` shape and `fmt`.
    pub unsafe fn upload_raw(
        weights: *const c_void,
        scales: *const f32,
        fmt: i32,
        i: i32,
        o: i32,
        device: i32,
    ) -> Option<ResidentTensor> {
        let mut ptr: *mut ColiCudaTensor = std::ptr::null_mut();
        if coli_cuda_tensor_upload(&mut ptr, weights, scales, fmt, i, o, device) != 0
            && !ptr.is_null()
        {
            Some(ResidentTensor { ptr })
        } else {
            None
        }
    }

    /// Zero-copy wrap of host buffers `[O, I]` (fmt: 0=f32, 1=int8, 2=int4). The
    /// GPU reads the RAM copy in place — no device allocation, no memcpy. int4 must
    /// be **offset-binary** (the on-disk / CPU form); the kernel handles it.
    ///
    /// # Safety
    /// `weights`/`scales` must stay alive and valid for `[O, I]`/`fmt` for as long
    /// as this tensor is used in a kernel. Only call when [`pageable_access`] is true.
    pub unsafe fn wrap_raw(
        weights: *const c_void,
        scales: *const f32,
        fmt: i32,
        i: i32,
        o: i32,
        device: i32,
    ) -> Option<ResidentTensor> {
        let mut ptr: *mut ColiCudaTensor = std::ptr::null_mut();
        if coli_cuda_tensor_wrap(&mut ptr, weights, scales, fmt, i, o, device) != 0
            && !ptr.is_null()
        {
            Some(ResidentTensor { ptr })
        } else {
            None
        }
    }

    /// Zero-copy wrap of an NVFP4 expert weight `[O, I]`: `weights` = packed e2m1
    /// nibbles, `bscale` = ue4m3 per-16 block scales, `gscale` = per-tensor global.
    /// Sets fmt=5; the GPU reads all three from host RAM in place.
    ///
    /// # Safety
    /// `weights` (`O*ceil(I/2)` bytes) and `bscale` (`O*ceil(I/16)` bytes) must stay
    /// alive and valid while this tensor is used in a kernel. Only when [`pageable_access`].
    pub unsafe fn wrap_raw_nvfp4(
        weights: *const c_void,
        bscale: *const c_void,
        gscale: f32,
        i: i32,
        o: i32,
        device: i32,
    ) -> Option<ResidentTensor> {
        let mut ptr: *mut ColiCudaTensor = std::ptr::null_mut();
        if coli_cuda_tensor_wrap_nvfp4(&mut ptr, weights, bscale, gscale, i, o, device) != 0
            && !ptr.is_null()
        {
            Some(ResidentTensor { ptr })
        } else {
            None
        }
    }

    /// Raw device handle (for the fused expert pipeline). Borrowed; do not free.
    pub fn as_raw(&self) -> *mut ColiCudaTensor {
        self.ptr
    }

    /// Resident byte size on the device.
    pub fn bytes(&self) -> usize {
        unsafe { coli_cuda_tensor_bytes(self.ptr) }
    }

    /// CUDA ordinal this tensor lives on.
    pub fn device(&self) -> i32 {
        unsafe { coli_cuda_tensor_device(self.ptr) }
    }
}

impl Drop for ResidentTensor {
    fn drop(&mut self) {
        if !self.ptr.is_null() {
            unsafe { coli_cuda_tensor_free(self.ptr) };
            self.ptr = std::ptr::null_mut();
        }
    }
}

/// `y[S,O] = x[S,I] @ W[O,I]^T` on the GPU. On the first call the weight is
/// uploaded into `slot` and reused thereafter. `weights`/`scales` are the CPU
/// copy used for the initial upload. Returns whether the GPU path ran.
///
/// # Safety
/// `slot` must be a persistent cell whose lifetime spans all calls reusing this
/// weight; `y` must hold `S*O` floats and `x` `S*I` floats.
#[allow(clippy::too_many_arguments)]
pub unsafe fn matmul(
    slot: &mut *mut ColiCudaTensor,
    y: &mut [f32],
    x: &[f32],
    weights: &[u8],
    scales: &[f32],
    fmt: i32,
    s: i32,
    i: i32,
    o: i32,
    device: i32,
) -> bool {
    coli_cuda_matmul(
        slot as *mut *mut ColiCudaTensor,
        y.as_mut_ptr(),
        x.as_ptr(),
        weights.as_ptr() as *const c_void,
        scales.as_ptr(),
        fmt,
        s,
        i,
        o,
        device,
    ) != 0
}

/// Raw-pointer matmul for the engine's dispatcher: `y[S,O] = x[S,I] @ W[O,I]^T`
/// on the GPU, uploading `weights`/`scales` into `slot` on first use. Lower-level
/// twin of [`matmul`] that avoids byte-slice reshaping at the call site.
///
/// # Safety
/// `slot` persists across calls reusing this weight; `y` has `s*o` floats, `x`
/// has `s*i` floats; `weights`/`scales` point to the CPU copy for the upload.
#[allow(clippy::too_many_arguments)]
pub unsafe fn matmul_raw(
    slot: &mut *mut ColiCudaTensor,
    y: *mut f32,
    x: *const f32,
    weights: *const c_void,
    scales: *const f32,
    fmt: i32,
    s: i32,
    i: i32,
    o: i32,
    device: i32,
) -> bool {
    coli_cuda_matmul(
        slot as *mut *mut ColiCudaTensor,
        y,
        x,
        weights,
        scales,
        fmt,
        s,
        i,
        o,
        device,
    ) != 0
}

/// Fused expert FFN `y = down(silu(gate(x)) * up(x))` for `S` rows, all three
/// weights already resident on one device.
pub fn expert_mlp(
    gate: &ResidentTensor,
    up: &ResidentTensor,
    down: &ResidentTensor,
    y: &mut [f32],
    x: &[f32],
    s: i32,
) -> bool {
    unsafe {
        coli_cuda_expert_mlp(gate.ptr, up.ptr, down.ptr, y.as_mut_ptr(), x.as_ptr(), s) != 0
    }
}

/// Raw-pointer fused expert FFN for the engine's dispatcher.
///
/// # Safety
/// The three handles must be resident on the same device; `y`/`x` hold `s*O`/`s*I`
/// floats.
pub unsafe fn expert_mlp_raw(
    gate: *mut ColiCudaTensor,
    up: *mut ColiCudaTensor,
    down: *mut ColiCudaTensor,
    y: *mut f32,
    x: *const f32,
    s: i32,
) -> bool {
    coli_cuda_expert_mlp(gate, up, down, y, x, s) != 0
}

/// Tiled FP8 (e4m3 weights, fp16 activations) fused expert FFN — the tensor-core
/// replacement for [`expert_mlp_raw`]. Requires all three tensors at fmt==4.
///
/// # Safety
/// Same contract as [`expert_mlp_raw`].
pub unsafe fn expert_mlp_fp8_raw(
    gate: *mut ColiCudaTensor,
    up: *mut ColiCudaTensor,
    down: *mut ColiCudaTensor,
    y: *mut f32,
    x: *const f32,
    s: i32,
) -> bool {
    coli_cuda_expert_mlp_fp8(gate, up, down, y, x, s) != 0
}

/// NVFP4 (e2m1 nibbles + ue4m3 per-16 block scale + f32 global) fused expert FFN —
/// GEMV at S==1 (decode; reads half the bytes of e4m3), tiled tensor-core at S>1
/// (prefill). Requires all three tensors at fmt==5.
///
/// # Safety
/// Same contract as [`expert_mlp_raw`].
pub unsafe fn expert_mlp_nvfp4_raw(
    gate: *mut ColiCudaTensor,
    up: *mut ColiCudaTensor,
    down: *mut ColiCudaTensor,
    y: *mut f32,
    x: *const f32,
    s: i32,
) -> bool {
    coli_cuda_expert_mlp_nvfp4(gate, up, down, y, x, s) != 0
}

/// Tiled int8 (W8A16) fused expert/MLP FFN — tensor-core replacement for the naive
/// `quant_matmul` on resident int8 weights (the shared expert). Requires fmt==1.
///
/// # Safety
/// Same contract as [`expert_mlp_raw`].
pub unsafe fn expert_mlp_i8a16_raw(
    gate: *mut ColiCudaTensor,
    up: *mut ColiCudaTensor,
    down: *mut ColiCudaTensor,
    y: *mut f32,
    x: *const f32,
    s: i32,
) -> bool {
    coli_cuda_expert_mlp_i8a16(gate, up, down, y, x, s) != 0
}

/// Batched fused expert FFN: all `count` (≤64) experts computed with ONE H2D + ONE
/// D2H and async kernels on the stream — pays the upload/download round-trip once for
/// the whole group instead of once per expert. `x`/`y` hold `sum(rows)` consecutive
/// `[I]`/`[O]` rows in expert order; `rows[c]` is expert `c`'s row count.
///
/// # Safety
/// The three slices are `count`-long arrays of resident handles on one device; `x`/`y`
/// hold `sum(rows)*I` / `sum(rows)*O` floats. Handles must outlive the call.
pub unsafe fn expert_group_raw(
    gates: &[*mut ColiCudaTensor],
    ups: &[*mut ColiCudaTensor],
    downs: &[*mut ColiCudaTensor],
    rows: &[i32],
    y: *mut f32,
    x: *const f32,
) -> bool {
    let count = gates.len();
    if count == 0 || ups.len() != count || downs.len() != count || rows.len() != count {
        return false;
    }
    coli_cuda_expert_group(
        gates.as_ptr(),
        ups.as_ptr(),
        downs.as_ptr(),
        rows.as_ptr(),
        count as c_int,
        y,
        x,
    ) != 0
}

/// Causal MLA weight-absorption attention on the GPU: computes `ctx[S, H*V]` from
/// query `q[S, H*(Q+R)]`, the compressed KV cache (`latent[T, K]` + `rope[T, R]`),
/// and the resident `kv_b` `[H*(Q+V), K]`. Twin of the CPU `absorb_core`.
///
/// # Safety
/// `kv_b` must be resident; `ctx`/`q`/`latent`/`rope` sized per the dims.
#[allow(clippy::too_many_arguments)]
pub unsafe fn attention_absorb_batch_raw(
    kv_b: *mut ColiCudaTensor,
    ctx: *mut f32,
    q: *const f32,
    latent: *const f32,
    rope: *const f32,
    s: i32,
    h: i32,
    q_nope: i32,
    r: i32,
    v: i32,
    k: i32,
    t: i32,
    scale: f32,
) -> bool {
    coli_cuda_attention_absorb_batch(kv_b, ctx, q, latent, rope, s, h, q_nope, r, v, k, t, scale)
        != 0
}

/// DSA sparse MLA attention: like [`attention_absorb_batch_raw`] but each query
/// attends only to its indexer selection. `sel_idx` is `[S, maxsel]` (row s = the
/// query's chosen cache rows, relative to the latent's first row), `sel_cnt` is `[S]`
/// (count per query; `<= 0` = the is_dense case → attend causally). Twin of the CPU
/// `reconstruct_core` sparse path.
///
/// # Safety
/// `kv_b` resident; `ctx`/`q`/`latent`/`rope` sized per the dims; `sel_idx` has
/// `s*maxsel` ints and `sel_cnt` has `s` ints.
#[allow(clippy::too_many_arguments)]
/// DSA indexer scores for the selecting queries — `scores[nsp, t]`, row `si` valid
/// for `t < pos_base+s0+si+1`.
///
/// # Safety
/// `qi` has `nsp*nh*hd` floats, `hw` `nsp*nh`, `keys` `t*hd`, `scores` `nsp*t`.
#[allow(clippy::too_many_arguments)]
pub unsafe fn dsa_indexer_scores_raw(
    scores: *mut f32,
    qi: *const f32,
    hw: *const f32,
    keys: *const f32,
    nsp: i32,
    s0: i32,
    nh: i32,
    hd: i32,
    t: i32,
    pos_base: i32,
    device: i32,
) -> bool {
    coli_cuda_dsa_indexer_scores(scores, qi, hw, keys, nsp, s0, nh, hd, t, pos_base, device) != 0
}

pub unsafe fn attention_absorb_sparse_raw(
    kv_b: *mut ColiCudaTensor,
    ctx: *mut f32,
    q: *const f32,
    latent: *const f32,
    rope: *const f32,
    sel_idx: *const i32,
    sel_cnt: *const i32,
    maxsel: i32,
    h0: i32,
    hc: i32,
    s: i32,
    h: i32,
    q_nope: i32,
    r: i32,
    v: i32,
    k: i32,
    t: i32,
    scale: f32,
) -> bool {
    coli_cuda_attention_absorb_sparse(
        kv_b, ctx, q, latent, rope, sel_idx, sel_cnt, maxsel, h0, hc, s, h, q_nope, r, v, k, t,
        scale,
    ) != 0
}

/// Single-token MLA absorb reading the KV cache from **device** memory (the
/// persistent-KV decode path). `latent_dev`/`rope_dev` point into the device KV
/// shadow; `ctx`/`q` are host.
///
/// # Safety
/// `kv_b` resident; device pointers valid for `[T, K]`/`[T, R]`; `ctx`/`q` host.
#[allow(clippy::too_many_arguments)]
pub unsafe fn attention_absorb_kvdev_raw(
    kv_b: *mut ColiCudaTensor,
    ctx: *mut f32,
    q: *const f32,
    latent_dev: *const f32,
    rope_dev: *const f32,
    h: i32,
    q_nope: i32,
    r: i32,
    v: i32,
    k: i32,
    t: i32,
    scale: f32,
) -> bool {
    coli_cuda_attention_absorb_kvdev(kv_b, ctx, q, latent_dev, rope_dev, h, q_nope, r, v, k, t, scale)
        != 0
}

/// Allocate `bytes` of device memory on `device`. `None` on failure.
pub fn pipe_alloc(device: i32, bytes: usize) -> Option<*mut c_void> {
    let p = unsafe { coli_cuda_pipe_alloc(device, bytes) };
    if p.is_null() {
        None
    } else {
        Some(p)
    }
}

/// Free device memory from [`pipe_alloc`].
///
/// # Safety
/// `p` must be a live pointer from `pipe_alloc(device, _)`.
pub unsafe fn pipe_free(device: i32, p: *mut c_void) {
    coli_cuda_pipe_free(device, p);
}

/// Host→device copy of `bytes`.
///
/// # Safety
/// `dst` is device memory ≥ `bytes`; `src` is host memory ≥ `bytes`.
pub unsafe fn pipe_upload(device: i32, dst: *mut c_void, src: *const c_void, bytes: usize) -> bool {
    coli_cuda_pipe_upload(device, dst, src, bytes) != 0
}

/// The CUDA (Blackwell) backend on one device.
pub struct CudaBackend {
    pub device: u32,
}

impl CudaBackend {
    /// Probe for a usable CUDA device and initialize device 0. Honors the
    /// `COLI_CUDA=0` opt-out. `None` when no device / init fails.
    ///
    /// `init` performs the real `cudaGetDeviceCount` + context setup internally
    /// (`device_count()` only reports *configured* contexts, so it is 0 until
    /// after init — don't gate on it here).
    pub fn probe() -> Option<CudaBackend> {
        if std::env::var("COLI_CUDA").map(|v| v == "0").unwrap_or(false) {
            return None;
        }
        if !init(&[0]) {
            return None;
        }
        Some(CudaBackend { device: 0 })
    }
}

impl Backend for CudaBackend {
    fn name(&self) -> &'static str {
        "cuda"
    }
    fn is_available(&self) -> bool {
        true
    }
    fn device(&self) -> Device {
        Device::Cuda(self.device)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    // Runs only when built `--features cuda` on a machine with a GPU. Proves the
    // GPU compute path (upload + gemm + download), not just device init.
    #[test]
    fn cuda_matmul_matches_hand_computed() {
        if !init(&[0]) {
            eprintln!("skipping: no usable CUDA device");
            return;
        }
        // W = [[1,2,3],[4,5,6]] (O=2, I=3), x = [1,1,1] -> y = [6, 15]
        let w: Vec<f32> = vec![1., 2., 3., 4., 5., 6.];
        let wb: Vec<u8> = w.iter().flat_map(|v| v.to_le_bytes()).collect();
        let x = vec![1.0f32, 1.0, 1.0];
        let scales = vec![1.0f32; 2];
        let mut y = vec![0f32; 2];
        let mut slot: *mut ColiCudaTensor = std::ptr::null_mut();
        let ok = unsafe { matmul(&mut slot, &mut y, &x, &wb, &scales, 0, 1, 3, 2, 0) };
        assert!(ok, "coli_cuda_matmul returned 0");
        assert!(
            (y[0] - 6.0).abs() < 1e-3 && (y[1] - 15.0).abs() < 1e-3,
            "GPU matmul y = {y:?}, expected [6, 15]"
        );
        unsafe { coli_cuda_tensor_free(slot) };
        shutdown();
    }
}
