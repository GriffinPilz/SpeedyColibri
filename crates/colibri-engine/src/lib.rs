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

pub mod attention;
pub mod cache;
pub mod forward;
pub mod linear;
pub mod loader;
pub mod math;
pub mod model;
pub mod moe;
pub mod quantize;
pub mod sampling;

pub use attention::{attention, attention_with, AttnCore};
pub use cache::{available_ram_bytes, capacity, CacheStats, ExpertCache};
pub use colibri_core::Config;
pub use forward::{forward, generate_greedy, logits};
pub use linear::{embed_row, matmul_f32, matmul_qt};
pub use loader::{ld, qt_load};
pub use math::{layernorm, rmsnorm, rope_interleave, sigmoid, silu, softmax};
pub use model::{KvCache, Layer, Model};
pub use moe::{dense_mlp, moe, route, Expert, ExpertProvider, ShardsExpertProvider};
pub use quantize::qtensor_from_f32;
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

/// Options controlling weight materialization. Defaults match the int4 GLM-5.2
/// container (`dbits = ebits = 4`); the pre-quantized `.qs` tensors are
/// self-describing, so `bits` only affects any full-precision fallback tensors.
#[derive(Debug, Clone, Copy)]
pub struct LoadOptions {
    /// bits/param for the dense part (attention, shared expert, embeddings)
    pub dbits: u32,
    /// bits/param for the routed experts (streamed; recorded on the model)
    pub ebits: u32,
}

impl Default for LoadOptions {
    fn default() -> Self {
        LoadOptions { dbits: 4, ebits: 4 }
    }
}

/// Load a model snapshot directory (`config.json` + `*.safetensors`) with default
/// options.
pub fn load_model(snap: impl AsRef<Path>) -> Result<Model, EngineError> {
    load_model_with(snap, LoadOptions::default())
}

/// Load a model snapshot, materializing the **dense** weights (embeddings,
/// lm_head, final norm, and per-layer attention + dense-MLP / shared-expert +
/// router). Port of the dense path of `model_init` / `load_weights` in `c/glm.c`.
///
/// The routed experts are **not** loaded here — they are streamed from the shards
/// on demand during the forward pass (the whole point of the engine). DSA-indexer
/// and MTP-head weights are detected (`has_dsa`/`has_mtp`) but their extra tensors
/// are loaded lazily by those subsystems (still being ported).
pub fn load_model_with(
    snap: impl AsRef<Path>,
    opts: LoadOptions,
) -> Result<Model, EngineError> {
    let snap = snap.as_ref();
    let cfg = Config::load(snap)?;
    let shards = colibri_safetensors::Shards::open(snap)?;

    let d = cfg.hidden as usize;
    let h = cfg.n_heads as usize;
    let dbits = opts.dbits;
    // embed/lm_head are the I/O boundary — keep them high precision (f32 when
    // dbits >= 8, else dbits), matching the C `io_bits`.
    let io_bits = if dbits >= 8 { 16 } else { dbits };

    let embed = qt_load(&shards, "model.embed_tokens.weight", cfg.vocab as usize, d, io_bits)?;
    let lm_head = qt_load(&shards, "lm_head.weight", cfg.vocab as usize, d, io_bits)?;
    let final_norm = ld(&shards, "model.norm.weight")?;

    let mut layers = Vec::with_capacity(cfg.n_layers as usize);
    for i in 0..cfg.n_layers as usize {
        let p = |s: &str| format!("model.layers.{i}.{s}");
        let mut l = Layer::default();
        l.in_ln = ld(&shards, &p("input_layernorm.weight"))?;
        l.post_ln = ld(&shards, &p("post_attention_layernorm.weight"))?;
        // MLA attention projections
        l.q_a = qt_load(&shards, &p("self_attn.q_a_proj.weight"), cfg.q_lora as usize, d, dbits)?;
        l.q_a_ln = ld(&shards, &p("self_attn.q_a_layernorm.weight"))?;
        l.q_b = qt_load(
            &shards,
            &p("self_attn.q_b_proj.weight"),
            h * cfg.qk_head as usize,
            cfg.q_lora as usize,
            dbits,
        )?;
        l.kv_a = qt_load(
            &shards,
            &p("self_attn.kv_a_proj_with_mqa.weight"),
            (cfg.kv_lora + cfg.qk_rope) as usize,
            d,
            dbits,
        )?;
        l.kv_a_ln = ld(&shards, &p("self_attn.kv_a_layernorm.weight"))?;
        l.kv_b = qt_load(
            &shards,
            &p("self_attn.kv_b_proj.weight"),
            h * (cfg.qk_nope + cfg.v_head) as usize,
            cfg.kv_lora as usize,
            dbits,
        )?;
        l.o = qt_load(
            &shards,
            &p("self_attn.o_proj.weight"),
            d,
            h * cfg.v_head as usize,
            dbits,
        )?;

        l.sparse = i as i32 >= cfg.first_dense;
        if !l.sparse {
            // dense MLP
            let inter = cfg.dense_inter as usize;
            l.gate_proj = qt_load(&shards, &p("mlp.gate_proj.weight"), inter, d, dbits)?;
            l.up_proj = qt_load(&shards, &p("mlp.up_proj.weight"), inter, d, dbits)?;
            l.down_proj = qt_load(&shards, &p("mlp.down_proj.weight"), d, inter, dbits)?;
        } else {
            // MoE: router (f32) + shared expert. Routed experts stream on demand.
            l.router = ld(&shards, &p("mlp.gate.weight"))?;
            l.router_bias = ld(&shards, &p("mlp.gate.e_score_correction_bias"))?;
            let s_i = (cfg.moe_inter * cfg.n_shared) as usize;
            l.sh_gate = qt_load(&shards, &p("mlp.shared_experts.gate_proj.weight"), s_i, d, dbits)?;
            l.sh_up = qt_load(&shards, &p("mlp.shared_experts.up_proj.weight"), s_i, d, dbits)?;
            l.sh_down = qt_load(&shards, &p("mlp.shared_experts.down_proj.weight"), d, s_i, dbits)?;
        }
        layers.push(l);
    }

    // MTP head lives at the extra layer index n_layers; DSA indexer weights are
    // per-layer `self_attn.indexer.*`.
    let has_mtp = shards.has(&format!("model.layers.{}.eh_proj.weight", cfg.n_layers));
    let has_dsa = (0..cfg.n_layers as usize).any(|i| {
        shards.has(&format!("model.layers.{i}.self_attn.indexer.wq_b.weight"))
    });

    Ok(Model {
        cfg,
        shards,
        ebits: opts.ebits as i32,
        dbits: dbits as i32,
        embed,
        lm_head,
        final_norm,
        layers,
        has_dsa,
        has_mtp,
    })
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
