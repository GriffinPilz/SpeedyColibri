//! Compute-backend abstraction for the expert/attention matmuls.
//!
//! The C engine selects a backend at runtime via `c/backend_loader.c`, which
//! `dlopen`s an optional CUDA shared library (`c/backend_cuda.cu`) or uses the
//! Metal backend (`c/backend_metal.mm`); everything falls back to the CPU
//! integer-dot kernels. This crate mirrors that: a [`Backend`] trait with a
//! always-available [`CpuBackend`], and feature-gated CUDA/Metal implementations.
//!
//! # Status
//!
//! [`CpuBackend`] is the real target for the CPU forward pass (it will delegate
//! to `colibri-kernels`). The `cuda`/`metal` features are placeholders that will
//! first bind the existing C backends over FFI, then be ported to Rust. Both are
//! off by default so the CPU engine builds on any platform.

/// Which backend a resident tensor should run on.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Device {
    Cpu,
    /// CUDA device by ordinal.
    Cuda(u32),
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

/// The always-available CPU backend. Delegates to `colibri-kernels`.
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
/// engine reads. For now this always returns the CPU backend.
pub fn autoselect() -> Box<dyn Backend> {
    // TODO(port c/backend_loader.c): probe CUDA (COLI_CUDA), then Metal, then CPU.
    Box::new(CpuBackend)
}

#[cfg(feature = "cuda")]
pub mod cuda {
    //! CUDA backend — port of `c/backend_cuda.cu` / `c/backend_cuda.h`.
    //! TODO: bind the existing `.cu` via FFI, then port kernels.
}

#[cfg(feature = "metal")]
pub mod metal {
    //! Metal backend — port of `c/backend_metal.mm` / `c/backend_metal.h`.
    //! TODO: bind the existing `.mm` via FFI, then port shaders.
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn cpu_backend_is_default_and_available() {
        let b = autoselect();
        assert_eq!(b.name(), "cpu");
        assert!(b.is_available());
        assert_eq!(b.device(), Device::Cpu);
    }
}
