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
