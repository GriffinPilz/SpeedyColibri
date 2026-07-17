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
pub mod convert;
pub mod forward;
#[cfg(feature = "cuda")]
pub mod gpu;
pub mod linear;
pub mod loader;
pub mod math;
pub mod model;
pub mod moe;
pub mod mtp;
pub mod preload;
pub mod quantize;
pub mod sampling;
pub mod usage;

pub use attention::{attention, attention_with, AttnCore};
pub use cache::{available_ram_bytes, capacity, total_ram_bytes, CacheStats, ExpertCache};
pub use convert::{
    convert_snapshot, detect_format, quant_error, ConvertOpts, ConvertStats, Scheme,
    SourceFormat, TensorErr,
};
pub use usage::UsageHistory;
pub use colibri_core::Config;
pub use forward::{
    forward, generate_greedy, generate_stream, generate_stream_drafting, layer_forward, logits,
    DecodeStats,
};
pub use linear::{embed_row, matmul_f32, matmul_qt};
pub use loader::{ld, qt_load};
pub use math::{layernorm, rmsnorm, rope_interleave, sigmoid, silu, softmax};
pub use model::{KvCache, Layer, Model, MtpHead, KV_UNSET};
pub use moe::{
    cluster_ctx, compute_experts_partial, dense_mlp, moe, moe_sharded, route, set_cluster,
    ClusterCtx, Expert, ExpertProvider, ShardsExpertProvider,
};
pub use mtp::{absorb as mtp_absorb, draft as mtp_draft};
pub use preload::{default_num_files, preload_parallel, repack, Manifest, PreloadStore};
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

/// Load one transformer layer's resident weights: MLA attention plus either the
/// dense MLP (`sparse == false`) or the MoE router + shared expert
/// (`sparse == true`). Routed experts are **not** loaded — they stream.
///
/// Shared by the main stack and by the MTP head, which is structurally just a
/// sparse layer at index `n_layers` (C: `mtpL`, always `sparse = 1`).
fn load_layer(
    shards: &colibri_safetensors::Shards,
    cfg: &Config,
    i: usize,
    dbits: u32,
    sparse: bool,
) -> Result<Layer, EngineError> {
    let d = cfg.hidden as usize;
    let h = cfg.n_heads as usize;
    let p = |s: &str| format!("model.layers.{i}.{s}");
    let mut l = Layer::default();
    l.in_ln = ld(shards, &p("input_layernorm.weight"))?;
    l.post_ln = ld(shards, &p("post_attention_layernorm.weight"))?;
    // MLA attention projections
    l.q_a = qt_load(shards, &p("self_attn.q_a_proj.weight"), cfg.q_lora as usize, d, dbits)?;
    l.q_a_ln = ld(shards, &p("self_attn.q_a_layernorm.weight"))?;
    l.q_b = qt_load(
        shards,
        &p("self_attn.q_b_proj.weight"),
        h * cfg.qk_head as usize,
        cfg.q_lora as usize,
        dbits,
    )?;
    l.kv_a = qt_load(
        shards,
        &p("self_attn.kv_a_proj_with_mqa.weight"),
        (cfg.kv_lora + cfg.qk_rope) as usize,
        d,
        dbits,
    )?;
    l.kv_a_ln = ld(shards, &p("self_attn.kv_a_layernorm.weight"))?;
    l.kv_b = qt_load(
        shards,
        &p("self_attn.kv_b_proj.weight"),
        h * (cfg.qk_nope + cfg.v_head) as usize,
        cfg.kv_lora as usize,
        dbits,
    )?;
    l.o = qt_load(shards, &p("self_attn.o_proj.weight"), d, h * cfg.v_head as usize, dbits)?;

    l.sparse = sparse;
    if !sparse {
        // dense MLP
        let inter = cfg.dense_inter as usize;
        l.gate_proj = qt_load(shards, &p("mlp.gate_proj.weight"), inter, d, dbits)?;
        l.up_proj = qt_load(shards, &p("mlp.up_proj.weight"), inter, d, dbits)?;
        l.down_proj = qt_load(shards, &p("mlp.down_proj.weight"), d, inter, dbits)?;
    } else {
        // MoE: router (f32) + shared expert. Routed experts stream on demand.
        l.router = ld(shards, &p("mlp.gate.weight"))?;
        l.router_bias = ld(shards, &p("mlp.gate.e_score_correction_bias"))?;
        let s_i = (cfg.moe_inter * cfg.n_shared) as usize;
        l.sh_gate = qt_load(shards, &p("mlp.shared_experts.gate_proj.weight"), s_i, d, dbits)?;
        l.sh_up = qt_load(shards, &p("mlp.shared_experts.up_proj.weight"), s_i, d, dbits)?;
        l.sh_down = qt_load(shards, &p("mlp.shared_experts.down_proj.weight"), d, s_i, dbits)?;
    }
    Ok(l)
}

/// Load the MTP speculative head at layer index `n_layers`, if the container
/// ships a **complete** one. Port of the MTP block of `model_init`.
///
/// The completeness gate matters: the head's tensors span several shards, so a
/// partial conversion (or a `--mtp` pass that was interrupted) leaves a subset
/// behind. The C refuses to enable MTP unless every required tensor is present —
/// a half-loaded head would draft garbage. `MTP=0` disables it regardless.
fn load_mtp(
    shards: &colibri_safetensors::Shards,
    cfg: &Config,
    dbits: u32,
) -> Result<Option<MtpHead>, EngineError> {
    let i = cfg.n_layers as usize;
    let last_e = (cfg.n_experts - 1).max(0) as usize;
    // Same required set as the C, with the last expert index taken from the
    // config rather than hardcoded at 255. experts.0/experts.{last} are probed
    // because they live on different shards than the rest of the head.
    let required = [
        "eh_proj.weight".to_string(),
        "enorm.weight".to_string(),
        "hnorm.weight".to_string(),
        "shared_head.norm.weight".to_string(),
        "input_layernorm.weight".to_string(),
        "post_attention_layernorm.weight".to_string(),
        "self_attn.q_a_proj.weight".to_string(),
        "self_attn.q_b_proj.weight".to_string(),
        "self_attn.kv_a_proj_with_mqa.weight".to_string(),
        "self_attn.kv_b_proj.weight".to_string(),
        "self_attn.o_proj.weight".to_string(),
        "mlp.gate.weight".to_string(),
        "mlp.shared_experts.gate_proj.weight".to_string(),
        "mlp.shared_experts.down_proj.weight".to_string(),
        "mlp.experts.0.gate_proj.weight".to_string(),
        format!("mlp.experts.{last_e}.down_proj.weight"),
    ];
    if !required.iter().all(|s| shards.has(&format!("model.layers.{i}.{s}"))) {
        return Ok(None);
    }
    if std::env::var("MTP").ok().as_deref() == Some("0") {
        return Ok(None);
    }

    let d = cfg.hidden as usize;
    let p = |s: &str| format!("model.layers.{i}.{s}");
    // The head's block is always sparse (C: `l->sparse = 1`).
    let layer = load_layer(shards, cfg, i, dbits, true)?;
    Ok(Some(MtpHead {
        layer,
        // [D, 2D]: consumes the concatenated [embed_normed ; hidden_normed].
        eh_proj: qt_load(shards, &p("eh_proj.weight"), d, 2 * d, dbits)?,
        enorm: ld(shards, &p("enorm.weight"))?,
        hnorm: ld(shards, &p("hnorm.weight"))?,
        mtp_norm: ld(shards, &p("shared_head.norm.weight"))?,
    }))
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

    // Fail fast with an actionable message on a partial download. An interrupted HF
    // pull leaves config.json plus only some `*.safetensors` shards, so tensors go
    // missing deep in loading (the "missing tensor: model.norm.weight" that a
    // half-downloaded node hits). Probe a few sentinels spanning the file set first.
    {
        let last = cfg.n_layers.saturating_sub(1) as usize;
        let sentinels = [
            "model.embed_tokens.weight".to_string(),
            "lm_head.weight".to_string(),
            "model.norm.weight".to_string(),
            format!("model.layers.{last}.input_layernorm.weight"),
        ];
        let missing: Vec<&str> =
            sentinels.iter().map(String::as_str).filter(|t| !shards.has(t)).collect();
        if !missing.is_empty() {
            return Err(std::io::Error::new(
                std::io::ErrorKind::NotFound,
                format!(
                    "model snapshot at {} looks INCOMPLETE: missing {}/{} core tensors [{}]. \
                     This is almost always a partial download — fetch the remaining .safetensors \
                     shards (re-run the Hugging Face download with network access, plus a token if \
                     the repo is gated) or mount a complete snapshot.",
                    snap.display(),
                    missing.len(),
                    sentinels.len(),
                    missing.join(", ")
                ),
            )
            .into());
        }
    }

    let d = cfg.hidden as usize;
    let dbits = opts.dbits;
    // embed/lm_head are the I/O boundary — keep them high precision (f32 when
    // dbits >= 8, else dbits), matching the C `io_bits`.
    let io_bits = if dbits >= 8 { 16 } else { dbits };

    let embed = qt_load(&shards, "model.embed_tokens.weight", cfg.vocab as usize, d, io_bits)?;
    let lm_head = qt_load(&shards, "lm_head.weight", cfg.vocab as usize, d, io_bits)?;
    let final_norm = ld(&shards, "model.norm.weight")?;

    let mut layers = Vec::with_capacity(cfg.n_layers as usize);
    for i in 0..cfg.n_layers as usize {
        let sparse = i as i32 >= cfg.first_dense;
        layers.push(load_layer(&shards, &cfg, i, dbits, sparse)?);
    }

    // MTP head lives at the extra layer index n_layers; DSA indexer weights are
    // per-layer `self_attn.indexer.*`.
    let mtp = load_mtp(&shards, &cfg, dbits)?;
    let has_dsa = (0..cfg.n_layers as usize).any(|i| {
        shards.has(&format!("model.layers.{i}.self_attn.indexer.wq_b.weight"))
    });

    let mut model = Model {
        cfg,
        shards,
        ebits: opts.ebits as i32,
        dbits: dbits as i32,
        embed,
        lm_head,
        final_norm,
        layers,
        has_dsa,
        has_mtp: mtp.is_some(),
        mtp,
    };
    // Dense weights are resident for the model's lifetime → GPU-cacheable.
    model.embed.gpu_eligible = true;
    model.lm_head.gpu_eligible = true;
    for l in &mut model.layers {
        for t in [
            &mut l.q_a,
            &mut l.q_b,
            &mut l.kv_a,
            &mut l.kv_b,
            &mut l.o,
            &mut l.gate_proj,
            &mut l.up_proj,
            &mut l.down_proj,
            &mut l.sh_gate,
            &mut l.sh_up,
            &mut l.sh_down,
        ] {
            t.gpu_eligible = true;
        }
    }
    Ok(model)
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
