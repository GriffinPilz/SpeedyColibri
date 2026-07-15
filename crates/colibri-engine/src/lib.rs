//! GLM-5.2 (`glm_moe_dsa`) MoE inference engine — port of `c/glm.c`.
//!
//! This is the heart of colibrì: streaming-expert MoE forward pass with MLA
//! attention, compressed KV-cache, DeepSeek-style sigmoid routing, MTP
//! speculative decoding, and the CPU integer-dot kernels.
//!
//! # Status
//!
//! Sampling ([`sampling`]) is ported and tested. The loader, attention, MoE,
//! KV-cache, and generation loop are scaffolded with faithful signatures and
//! are the active porting front — see PORTING.md for the milestone order.

pub mod model;
pub mod sampling;

pub use colibri_core::Config;
pub use model::{KvState, Layer, Model};
pub use sampling::{argmax, sample_top_p, SampleConfig};

use std::path::Path;

/// Errors from loading or running the engine.
#[derive(Debug)]
pub enum EngineError {
    Config(colibri_core::ConfigError),
    Io(std::io::Error),
    /// A subsystem that is scaffolded but not yet ported was invoked.
    NotImplemented(&'static str),
}

impl std::fmt::Display for EngineError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            EngineError::Config(e) => write!(f, "{e}"),
            EngineError::Io(e) => write!(f, "io error: {e}"),
            EngineError::NotImplemented(what) => {
                write!(f, "not yet ported to Rust: {what}")
            }
        }
    }
}

impl std::error::Error for EngineError {}

impl From<colibri_core::ConfigError> for EngineError {
    fn from(e: colibri_core::ConfigError) -> Self {
        EngineError::Config(e)
    }
}
impl From<std::io::Error> for EngineError {
    fn from(e: std::io::Error) -> Self {
        EngineError::Io(e)
    }
}

/// Load a model snapshot directory (`config.json` + `*.safetensors`).
///
/// Today this loads and validates the config and indexes the shards — the parts
/// already ported — then reports the weight-materialization step as pending.
pub fn load_model(snap: impl AsRef<Path>) -> Result<Model, EngineError> {
    let snap = snap.as_ref();
    let cfg = Config::load(snap)?;
    let shards = colibri_safetensors::Shards::open(snap)?;
    // TODO(port c/glm.c load_weights): materialize embed/lm_head/layers from
    // shards (qt_from_disk), detect DSA/MTP, size the expert LRU.
    let _ = (&cfg, &shards);
    Err(EngineError::NotImplemented("weight loading (glm.c load_weights)"))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn load_missing_snapshot_errors_cleanly() {
        // Config load fails first (no config.json) — a clean error, not a panic.
        // (`Model` intentionally doesn't derive Debug, so match instead of unwrap.)
        match load_model("/nonexistent/snapshot/path") {
            Err(EngineError::Config(_)) => {}
            Err(other) => panic!("expected config error, got: {other}"),
            Ok(_) => panic!("expected an error for a missing snapshot"),
        }
    }
}
