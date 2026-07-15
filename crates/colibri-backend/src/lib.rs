//! Compute-backend abstraction for the expert/attention matmuls.
//!
//! Deployment target is **NVIDIA DGX Spark** (GB10 Grace Blackwell): the primary
//! backend is **CUDA** on the Blackwell GPU; the CPU path (Grace, aarch64 NEON)
//! is the fallback. Apple-Silicon **Metal is off the critical path** — kept only
//! as an optional, deprioritized feature stub.
//!
//! The C engine selects a backend at runtime via `c/backend_loader.c` (which
//! `dlopen`s `c/backend_cuda.cu`), falling back to the CPU integer-dot kernels.
//! This crate mirrors that: a [`Backend`] trait, an always-available
//! [`CpuBackend`], and a feature-gated CUDA backend (Metal behind a separate,
//! optional feature).
//!
//! # Status
//!
//! [`CpuBackend`] is real (it will delegate to `colibri-kernels`). The `cuda`
//! feature is the placeholder for the Blackwell backend — it will first bind
//! `c/backend_cuda.cu` over FFI, then be ported. Off by default so the image
//! builds without the CUDA toolkit present; the DGX Spark image turns it on.

/// Which backend a resident tensor should run on.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Device {
    Cpu,
    /// CUDA device by ordinal (Blackwell on DGX Spark).
    Cuda(u32),
    /// Apple Metal — deprioritized, not a deployment target.
    Metal,
}

/// A compute backend for the hot matmuls. Kept deliberately small for now; it
/// will grow the expert-tier and attention entry points as the engine lands.
pub trait Backend {
    /// Human-readable name for logs (`[BACKEND] ...`).
    fn name(&self) -> &'static str;

    /// Whether this backend is actually usable in the current process.
    fn is_available(&self) -> bool;

    /// The device this backend represents.
    fn device(&self) -> Device;
}

/// The always-available CPU backend. On DGX Spark this is the Grace ARM cores;
/// delegates to `colibri-kernels` (aarch64 NEON is the CPU SIMD target).
#[derive(Debug, Clone, Copy, Default)]
pub struct CpuBackend;

impl Backend for CpuBackend {
    fn name(&self) -> &'static str {
        "cpu"
    }
    fn is_available(&self) -> bool {
        true
    }
    fn device(&self) -> Device {
        Device::Cpu
    }
}

/// Select the best available backend, honoring the `COLI_CUDA` env toggle the C
/// engine reads. Prefers CUDA (Blackwell) when compiled in and available, else
/// falls back to the CPU backend.
pub fn autoselect() -> Box<dyn Backend> {
    #[cfg(feature = "cuda")]
    {
        if std::env::var("COLI_CUDA").map(|v| v != "0").unwrap_or(true) {
            if let Some(b) = cuda::CudaBackend::probe() {
                return Box::new(b);
            }
        }
    }
    // TODO(port c/backend_loader.c): full runtime probe order (CUDA -> CPU).
    Box::new(CpuBackend)
}

#[cfg(feature = "cuda")]
pub mod cuda;
#[cfg(feature = "cuda")]
pub use cuda::CudaBackend;

#[cfg(feature = "metal")]
pub mod metal {
    //! Metal backend — port of `c/backend_metal.mm`. DEPRIORITIZED: not a
    //! deployment target (DGX Spark is CUDA). Kept as an optional stub only.
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn cpu_backend_is_available_and_default_without_cuda() {
        let b = autoselect();
        // Without the `cuda` feature (default), autoselect returns CPU.
        assert!(b.is_available());
        #[cfg(not(feature = "cuda"))]
        {
            assert_eq!(b.name(), "cpu");
            assert_eq!(b.device(), Device::Cpu);
        }
    }
}
