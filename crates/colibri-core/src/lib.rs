//! Core shared types for the colibrì engine.
//!
//! Ports the small, dependency-light pieces the rest of the engine builds on:
//! dtype conversions (`c/st.h`), the quantized-tensor container and `qt_bytes`
//! (`c/glm.c`), the model `Config`/`load_cfg` (`c/glm.c`), and the tier/hot-store
//! eviction policy (`c/tier.h`).

pub mod config;
pub mod dtype;
pub mod quant;
pub mod tier;

pub use config::{Config, ConfigError};
pub use dtype::{bf16_to_f32, f16_to_f32, DType};
pub use quant::{Bytes, QFormat, QTensor};
pub use tier::{decay, lfru_score, pick_lfru, pick_swap, Swap};
