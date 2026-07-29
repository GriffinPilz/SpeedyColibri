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
pub mod chunk;
pub mod convert;
pub mod dsa;
pub mod forward;
#[cfg(feature = "cuda")]
pub mod gpu;
pub mod kda;
pub mod linear;
pub mod loader;
pub mod mamba2;
pub mod math;
pub mod model;
pub mod moe;
pub mod mtp;
pub mod preload;
pub mod quantize;
pub mod sampling;
pub mod usage;

pub use attention::{
    attention, attention_sharded, attention_with, attention_with_heads, compute_attention_partial,
    dsa_selection_for, head_slice, AttnCore,
};
pub use cache::{available_ram_bytes, capacity, total_ram_bytes, CacheStats, ExpertCache};
pub use convert::{
    convert_snapshot, detect_format, quant_error, requant_experts_nvfp4, ConvertOpts,
    ConvertStats, Scheme, SourceFormat, TensorErr,
};
pub use usage::UsageHistory;
pub use colibri_core::{Config, QTensor};
pub use forward::{
    forward, forward_batched, generate_greedy, generate_stream, generate_stream_drafting,
    kimi_forward, layer_forward, layer_forward_kind, logits, mamba2_mixer, DecodeStats,
};
pub use linear::{embed_row, matmul_f32, matmul_qt};
pub use loader::{ld, qt_load};
pub use math::{layernorm, rmsnorm, rope_interleave, sigmoid, silu, softmax};
pub use model::{KvCache, Layer, Model, MtpBlock, MtpHead, KV_UNSET};
pub use moe::{
    cluster_ctx, compute_experts_partial, dense_mlp, kimi_moe, moe, moe_sharded, nemotron_moe,
    route,
    set_activation, set_cluster, ClusterCtx, Expert, ExpertLayout, ExpertProvider,
    ShardsExpertProvider,
};
pub use mtp::{absorb as mtp_absorb, draft as mtp_draft};
pub use preload::{default_num_files, preload_parallel, repack, Manifest, PreloadStore};
pub use quantize::qtensor_from_f32;
pub use sampling::{argmax, sample_top_p, SampleConfig};

use colibri_core::{Arch, LayerKind};
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

/// Options controlling weight materialization. Defaults match the int8-resident
/// GLM-5.2 container (`dbits = ebits = 8`); the pre-quantized `.qs`/nvfp4 tensors are
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
        LoadOptions { dbits: 8, ebits: 8 }
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
    // Nemotron-H layers are heterogeneous (Mamba2 / GQA attention / latent-MoE) and use a
    // single block norm under a different tensor layout (`mixer.*`), so they load via a
    // dedicated per-kind path rather than the shared MLA/GQA prefix below.
    if cfg.arch == Arch::NemotronH {
        return load_layer_nemotron(shards, cfg, i, dbits);
    }
    // Kimi-K3 is hybrid on the mixer axis (KDA / gated MLA) and carries an output gate
    // plus the attention-residual pair on every layer, so it takes its own path rather
    // than the MLA/GQA branch below. `sparse` is derived from `first_dense` inside.
    if cfg.arch == Arch::KimiK3 {
        return load_layer_kimi(shards, cfg, i, dbits);
    }
    let d = cfg.hidden as usize;
    let h = cfg.n_heads as usize;
    let p = |s: &str| format!("model.layers.{i}.{s}");
    let mut l = Layer::default();
    l.in_ln = ld(shards, &p("input_layernorm.weight"))?;
    l.post_ln = ld(shards, &p("post_attention_layernorm.weight"))?;
    // Output projection is shared by both attention flavours (`[hidden, n_heads*head_dim]`).
    l.o = qt_load(shards, &p("self_attn.o_proj.weight"), d, h * cfg.v_head as usize, dbits)?;

    if cfg.arch.is_gqa() {
        // GQA attention (MiniMax M3/M2): q/k/v projections + QK-norm. `qk_head` is the
        // head dim; K/V carry `n_kv_heads` heads. Sparse-indexer weights below load
        // only on M3's sparse layers (idx_type all-false for M2 → dense everywhere).
        let hd = cfg.qk_head as usize;
        let kvh = cfg.n_kv_heads as usize;
        l.q_proj = Some(qt_load(shards, &p("self_attn.q_proj.weight"), h * hd, d, dbits)?);
        l.k_proj = Some(qt_load(shards, &p("self_attn.k_proj.weight"), kvh * hd, d, dbits)?);
        l.v_proj = Some(qt_load(shards, &p("self_attn.v_proj.weight"), kvh * hd, d, dbits)?);
        // Fuse q/k/v (they share the input x) into ONE matmul per layer: at S=1 decode
        // each was a separate synchronized GPU dispatch, ~25% of decode across the
        // projections. Drop the separate three — `attention_gqa` uses the fused tensor.
        l.qkv_proj = Some(crate::loader::concat_rows(&[
            l.q_proj.as_ref().unwrap(),
            l.k_proj.as_ref().unwrap(),
            l.v_proj.as_ref().unwrap(),
        ]));
        l.q_proj = None;
        l.k_proj = None;
        l.v_proj = None;
        l.q_norm = ld(shards, &p("self_attn.q_norm.weight"))?;
        l.k_norm = ld(shards, &p("self_attn.k_norm.weight"))?;
        // Block-sparse Lightning Indexer weights on sparse attention layers.
        if cfg.idx_type.get(i).copied().unwrap_or(false)
            && cfg.index_hd > 0
            && shards.has(&p("self_attn.index_q_proj.weight"))
        {
            let (inh, ihd) = (cfg.index_nh as usize, cfg.index_hd as usize);
            l.idx_q_proj = Some(qt_load(shards, &p("self_attn.index_q_proj.weight"), inh * ihd, d, dbits)?);
            l.idx_k_proj = Some(qt_load(shards, &p("self_attn.index_k_proj.weight"), ihd, d, dbits)?);
            l.idx_q_norm = ld(shards, &p("self_attn.index_q_norm.weight"))?;
            l.idx_k_norm = ld(shards, &p("self_attn.index_k_norm.weight"))?;
        }
    } else {
        // MLA attention projections (GLM)
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

        // DSA lightning indexer — present only when the checkpoint was converted with the
        // indexer weights (`--indexer`). Load per layer that carries them; a model without
        // these tensors leaves the fields `None` and attention runs dense. Names/dims match
        // the C loader (`self_attn.indexer.{wq_b,wk,weights_proj,k_norm}`).
        if cfg.index_hd > 0 && cfg.index_nh > 0 && shards.has(&p("self_attn.indexer.wq_b.weight")) {
            let (nh, hd) = (cfg.index_nh as usize, cfg.index_hd as usize);
            l.ix_wq = Some(qt_load(shards, &p("self_attn.indexer.wq_b.weight"), nh * hd, cfg.q_lora as usize, dbits)?);
            l.ix_wk = Some(qt_load(shards, &p("self_attn.indexer.wk.weight"), hd, d, dbits)?);
            l.ix_wp = Some(qt_load(shards, &p("self_attn.indexer.weights_proj.weight"), nh, d, dbits)?);
            l.ix_knorm_w = ld(shards, &p("self_attn.indexer.k_norm.weight"))?;
            l.ix_knorm_b = ld(shards, &p("self_attn.indexer.k_norm.bias"))?;
        }
    }

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
        // The router selection bias sits under `.gate.` on GLM but directly under the
        // MoE block on MiniMax-M3 (`block_sparse_moe.e_score_correction_bias` →
        // `mlp.e_score_correction_bias`); accept either.
        l.router_bias = ld(shards, &p("mlp.gate.e_score_correction_bias"))
            .or_else(|_| ld(shards, &p("mlp.e_score_correction_bias")))?;
        // Shared expert — GLM/M3 have one; MiniMax-M2 has none (n_shared 0). Only load
        // (and later compute) it when present, else the tensors are absent from the
        // container and the fields stay at their empty default.
        let s_i = (cfg.moe_inter * cfg.n_shared) as usize;
        if s_i > 0 {
            l.sh_gate = qt_load(shards, &p("mlp.shared_experts.gate_proj.weight"), s_i, d, dbits)?;
            l.sh_up = qt_load(shards, &p("mlp.shared_experts.up_proj.weight"), s_i, d, dbits)?;
            l.sh_down = qt_load(shards, &p("mlp.shared_experts.down_proj.weight"), d, s_i, dbits)?;
        }
    }
    Ok(l)
}

/// Load one Nemotron-H layer, keyed on `cfg.layer_kind[i]`. Tensors live under the
/// `mixer.*` prefix and the block's single norm is `norm.weight` (no post-norm).
///
///   * **Mamba2** (`in_proj`/`out_proj` resident matmuls; `conv1d.{weight,bias}`, `A_log`,
///     `D`, `dt_bias`, gated `norm.weight` as f32 vectors). `conv1d.weight` is stored
///     `[conv_dim, 1, k]`; read flat it is exactly the `[conv_dim, k]` the mixer wants.
///   * **Attention** — GQA `q/k/v/o` projections (NoPE, no QK-norm), q/k/v fused into one
///     matmul like the M3/M2 path so `attention_gqa` runs a single projection.
///   * **MoE** — router (`gate.weight` f32 + `gate.e_score_correction_bias`), the two
///     latent projections, and the resident shared expert (`shared_experts.{up,down}_proj`
///     → `up_proj`/`down_proj`, gateless ReLU²). The routed experts stream via the provider.
fn load_layer_nemotron(
    shards: &colibri_safetensors::Shards,
    cfg: &Config,
    i: usize,
    dbits: u32,
) -> Result<Layer, EngineError> {
    load_layer_nemotron_kind(shards, cfg, i, dbits, cfg.layer_kind[i])
}

/// [`load_layer_nemotron`] with the mixer kind supplied instead of looked up.
///
/// The MTP head's sublayers live at `i >= n_layers`, past the end of `cfg.layer_kind`
/// (which is `num_hidden_layers` long and describes the main stack only), so they cannot
/// look their own kind up — see `Config::mtp_layer_kind`.
fn load_layer_nemotron_kind(
    shards: &colibri_safetensors::Shards,
    cfg: &Config,
    i: usize,
    dbits: u32,
    kind: LayerKind,
) -> Result<Layer, EngineError> {
    let d = cfg.hidden as usize;
    let p = |s: &str| format!("model.layers.{i}.{s}");
    let mut l = Layer::default();
    // The block's single input RMSNorm (Nemotron-H has no post-attention norm).
    // Convert emits it under the canonical `input_layernorm.weight` name (the source's
    // `layers.N.norm.weight`), so the generic completeness check finds it.
    l.in_ln = ld(shards, &p("input_layernorm.weight"))?;

    match kind {
        LayerKind::Mamba => {
            let nh = cfg.mamba_n_heads as usize;
            let d_inner = cfg.mamba_inter as usize;
            let conv_dim = d_inner + 2 * cfg.mamba_n_groups as usize * cfg.mamba_d_state as usize;
            let proj_out = d_inner + conv_dim + nh;
            l.mamba_in_proj = Some(qt_load(shards, &p("mixer.in_proj.weight"), proj_out, d, dbits)?);
            l.mamba_out_proj =
                Some(qt_load(shards, &p("mixer.out_proj.weight"), d, d_inner, dbits)?);
            // `[conv_dim, 1, k]` read flat == `[conv_dim, k]`; bias present iff use_conv_bias
            // (empty is tolerated by `causal_conv1d_silu`).
            l.mamba_conv_w = ld(shards, &p("mixer.conv1d.weight"))?;
            l.mamba_conv_b = ld(shards, &p("mixer.conv1d.bias")).unwrap_or_default();
            l.mamba_a_log = ld(shards, &p("mixer.A_log"))?;
            l.mamba_d = ld(shards, &p("mixer.D"))?;
            l.mamba_dt_bias = ld(shards, &p("mixer.dt_bias"))?;
            l.mamba_norm = ld(shards, &p("mixer.norm.weight"))?;
        }
        LayerKind::Attn => {
            let hd = cfg.qk_head as usize;
            let h = cfg.n_heads as usize;
            let kvh = cfg.n_kv_heads as usize;
            // Fuse q/k/v (shared input) into ONE matmul, as on the M3/M2 GQA path; NoPE and
            // no QK-norm, so `q_norm`/`k_norm` stay empty (see `attention_gqa`).
            let q = qt_load(shards, &p("mixer.q_proj.weight"), h * hd, d, dbits)?;
            let k = qt_load(shards, &p("mixer.k_proj.weight"), kvh * hd, d, dbits)?;
            let v = qt_load(shards, &p("mixer.v_proj.weight"), kvh * hd, d, dbits)?;
            l.qkv_proj = Some(crate::loader::concat_rows(&[&q, &k, &v]));
            l.o = qt_load(shards, &p("mixer.o_proj.weight"), d, h * hd, dbits)?;
        }
        LayerKind::Moe => {
            l.sparse = true;
            // Router (f32) + its additive selection bias (both reused from the M3/M2 path).
            l.router = ld(shards, &p("mixer.gate.weight"))?;
            l.router_bias = ld(shards, &p("mixer.gate.e_score_correction_bias"))?;
            let dl = cfg.moe_latent as usize;
            l.fc1_latent = Some(qt_load(shards, &p("mixer.fc1_latent_proj.weight"), dl, d, dbits)?);
            l.fc2_latent = Some(qt_load(shards, &p("mixer.fc2_latent_proj.weight"), d, dl, dbits)?);
            // Shared expert (gateless ReLU²) reuses `up_proj`/`down_proj`.
            let si = cfg.shared_inter as usize;
            l.up_proj = qt_load(shards, &p("mixer.shared_experts.up_proj.weight"), si, d, dbits)?;
            l.down_proj =
                qt_load(shards, &p("mixer.shared_experts.down_proj.weight"), d, si, dbits)?;
            // Routed experts (latent-space, gateless ReLU²) stream via the ExpertProvider.
        }
        // Kimi-K3's KDA layers carry a different tensor set entirely (q/k/v + separate
        // `*_conv1d`, `f_a`/`f_b` low-rank gate, `A_log`, `dt_bias`, `b_proj`) and live
        // under a different prefix. Loading them through the Nemotron-H mixer names would
        // silently produce a half-populated layer, so refuse instead.
        LayerKind::Kda => {
            return Err(EngineError::NotImplemented("Kimi-K3 KDA layer loading"));
        }
    }
    Ok(l)
}

/// Load one Kimi-K3 layer. Both mixers share the same weight set; which one runs is
/// `cfg.layer_kind[i]`, never the tensor names.
///
/// NOTE the layer is NOT the ordinary two-sublayer `x += attn(ln(x))` form. K3 has no
/// residual stream at all: a `prefix_sum` accumulates sublayer outputs and the running
/// state is recomputed by a softmax attention over saved states (see the
/// `attn_res_*`/`mlp_res_*` note on [`Layer`]). That only affects the driver — the
/// weights loaded here are the same either way.
///
/// * **KDA** ([`LayerKind::Kda`], 69 layers): q/k/v projections (reusing the GQA
///   fields), a per-head decay projection `b_proj`, a factored forget gate
///   `f_a`/`f_b`, a short causal depthwise conv on each of q/k/v, and the `A_log`
///   / `dt_bias` / `o_norm` vectors. Structurally a delta rule, so the vectors
///   parallel Mamba2's — but the state is a `[head_dim, head_dim]` matrix per head,
///   not a selective scan (see `KvCache::enable_kda`).
/// * **gated MLA** ([`LayerKind::Attn`], 24 layers): the same q_a/q_b/kv_a/kv_b
///   latent projections GLM uses, so those fields are reused verbatim.
///
/// Both carry `g_proj`, the output gate, on top of the shared `o` projection — the
/// header sweep confirmed it on all 93 layers, not just one mixer's.
///
/// Every layer also carries the attention-residual score vectors (`*_res_norm` and the
/// `[1, hidden]` `*_res_proj`), which no other arch has — they are multiplied into one
/// `[hidden]` vector at use, not applied as a norm then a projection.
fn load_layer_kimi(
    shards: &colibri_safetensors::Shards,
    cfg: &Config,
    i: usize,
    dbits: u32,
) -> Result<Layer, EngineError> {
    let d = cfg.hidden as usize;
    let p = |s: &str| format!("model.layers.{i}.{s}");
    let mut l = Layer::default();

    l.in_ln = ld(shards, &p("input_layernorm.weight"))?;
    l.post_ln = ld(shards, &p("post_attention_layernorm.weight"))?;
    // Attention-residual score vectors, one pair per sublayer. NOT a residual — see the
    // note on `Layer` and `forward::apply_attn_res`.
    l.attn_res_norm = ld(shards, &p("self_attention_res_norm.weight"))?;
    l.attn_res_proj = ld(shards, &p("self_attention_res_proj.weight"))?;
    l.mlp_res_norm = ld(shards, &p("mlp_res_norm.weight"))?;
    l.mlp_res_proj = ld(shards, &p("mlp_res_proj.weight"))?;

    // Mixer output width, feeding both `o` and the gate. The two mixers reach it by
    // different routes — KDA via its own head geometry, MLA via `n_heads * v_head` —
    // and they happen to coincide at 12288 on K3. Deriving it per kind rather than
    // reusing one for both means a checkpoint where they diverge fails in `qt_load`
    // with a shape error instead of silently reading the wrong number of rows.
    let kind = cfg.layer_kind.get(i).copied();
    let c = match kind {
        Some(LayerKind::Kda) => cfg.kda_n_heads as usize * cfg.kda_head_dim as usize,
        _ => cfg.n_heads as usize * cfg.v_head as usize,
    };
    l.o = qt_load(shards, &p("self_attn.o_proj.weight"), d, c, dbits)?;
    l.attn_gate = Some(qt_load(shards, &p("self_attn.g_proj.weight"), c, d, dbits)?);

    match kind {
        Some(LayerKind::Kda) => {
            let (nh, hd) = (cfg.kda_n_heads as usize, cfg.kda_head_dim as usize);
            l.q_proj = Some(qt_load(shards, &p("self_attn.q_proj.weight"), c, d, dbits)?);
            l.k_proj = Some(qt_load(shards, &p("self_attn.k_proj.weight"), c, d, dbits)?);
            l.v_proj = Some(qt_load(shards, &p("self_attn.v_proj.weight"), c, d, dbits)?);
            l.kda_b_proj = Some(qt_load(shards, &p("self_attn.b_proj.weight"), nh, d, dbits)?);
            // Factored forget gate: hidden -> r -> n_heads*head_dim. `r` is derived from
            // the checkpoint rather than assumed — it is not any other config field.
            //
            // NOT from `shape[0]`: a CONTAINER stores every weight FLAT, so `f_a`'s shape
            // is `[917504]` (the element count), not the source checkpoint's
            // `[128, 7168]`. Reading `shape[0]` yielded r = 917504, which made `qt_load`
            // infer a bogus format and blew up inside `matmul_qt` on the first KDA layer.
            // The loader only ever sees containers, so that read was never going to work.
            //
            // The `.qs` sidecar carries exactly one f32 scale per ROW, so its length is
            // the row count — unambiguous regardless of how the weight itself is packed.
            // Fall back to `elements / hidden` (right for any 1-byte-per-weight format),
            // then to `head_dim`.
            let f_a = p("self_attn.f_a_proj.weight");
            let r = shards
                .find(&format!("{f_a}.qs"))
                .and_then(|t| t.shape.first().copied())
                .or_else(|| {
                    shards.find(&f_a).and_then(|t| t.shape.first().copied()).map(|n| n / d as i64)
                })
                .unwrap_or(hd as i64) as usize;
            l.kda_f_a = Some(qt_load(shards, &p("self_attn.f_a_proj.weight"), r, d, dbits)?);
            l.kda_f_b = Some(qt_load(shards, &p("self_attn.f_b_proj.weight"), c, r, dbits)?);
            // `[C, 1, k]` on disk; read flat it is exactly the `[C, k]` the mixer wants
            // (same trick as Nemotron's `conv1d.weight`).
            l.kda_conv_q = ld(shards, &p("self_attn.q_conv1d.weight"))?;
            l.kda_conv_k = ld(shards, &p("self_attn.k_conv1d.weight"))?;
            l.kda_conv_v = ld(shards, &p("self_attn.v_conv1d.weight"))?;
            l.kda_a_log = ld(shards, &p("self_attn.A_log"))?;
            l.kda_dt_bias = ld(shards, &p("self_attn.dt_bias"))?;
            l.kda_o_norm = ld(shards, &p("self_attn.o_norm.weight"))?;
        }
        _ => {
            // Gated MLA — the GLM latent projections verbatim, plus `attn_gate` above.
            let (nh, ql, kl) = (cfg.n_heads as usize, cfg.q_lora as usize, cfg.kv_lora as usize);
            l.q_a = qt_load(shards, &p("self_attn.q_a_proj.weight"), ql, d, dbits)?;
            l.q_b = qt_load(
                shards,
                &p("self_attn.q_b_proj.weight"),
                nh * (cfg.qk_nope + cfg.qk_rope) as usize,
                ql,
                dbits,
            )?;
            l.kv_a = qt_load(
                shards,
                &p("self_attn.kv_a_proj_with_mqa.weight"),
                kl + cfg.qk_rope as usize,
                d,
                dbits,
            )?;
            l.kv_b = qt_load(
                shards,
                &p("self_attn.kv_b_proj.weight"),
                nh * (cfg.qk_nope + cfg.v_head) as usize,
                kl,
                dbits,
            )?;
            l.q_a_ln = ld(shards, &p("self_attn.q_a_layernorm.weight"))?;
            l.kv_a_ln = ld(shards, &p("self_attn.kv_a_layernorm.weight"))?;
        }
    }

    // FFN: the `first_dense` prefix is a plain MLP, everything after it is latent MoE.
    // Read off `first_dense`, NOT `layer_kind` — MoE-ness is not on the mixer axis
    // for K3 (see `Config::moe_layers`).
    if (i as i32) < cfg.first_dense {
        let di = cfg.dense_inter as usize;
        l.gate_proj = qt_load(shards, &p("mlp.gate_proj.weight"), di, d, dbits)?;
        l.up_proj = qt_load(shards, &p("mlp.up_proj.weight"), di, d, dbits)?;
        l.down_proj = qt_load(shards, &p("mlp.down_proj.weight"), d, di, dbits)?;
    } else {
        l.sparse = true;
        let (dl, si) = (cfg.moe_latent as usize, cfg.shared_inter as usize);
        l.router = ld(shards, &p("mlp.gate.weight"))?;
        l.router_bias = ld(shards, &p("mlp.gate.e_score_correction_bias"))?;
        l.fc1_latent = Some(qt_load(shards, &p("mlp.fc1_latent_proj.weight"), dl, d, dbits)?);
        l.fc2_latent = Some(qt_load(shards, &p("mlp.fc2_latent_proj.weight"), d, dl, dbits)?);
        l.routed_expert_norm = ld(shards, &p("mlp.routed_expert_norm.weight"))?;
        // Shared experts: ONE fused pair, `shared_inter = n_shared * moe_inter` wide.
        l.sh_gate = qt_load(shards, &p("mlp.shared_experts.gate_proj.weight"), si, d, dbits)?;
        l.sh_up = qt_load(shards, &p("mlp.shared_experts.up_proj.weight"), si, d, dbits)?;
        l.sh_down = qt_load(shards, &p("mlp.shared_experts.down_proj.weight"), d, si, dbits)?;
        // Routed experts stream through the ExpertProvider.
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
    // `MTP=0` disables any head, on every architecture. Checked before the per-arch
    // probes so the env override cannot be defeated by tensor layout.
    if std::env::var("MTP").ok().as_deref() == Some("0") {
        return Ok(None);
    }
    if cfg.arch == Arch::NemotronH {
        return load_mtp_nemotron(shards, cfg, dbits);
    }
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

    let d = cfg.hidden as usize;
    let p = |s: &str| format!("model.layers.{i}.{s}");
    // The head's block is always sparse (C: `l->sparse = 1`).
    let layer = load_layer(shards, cfg, i, dbits, true)?;
    Ok(Some(MtpHead {
        // GLM/M3: exactly one block, and no `kind` — those arches never consult one.
        blocks: vec![MtpBlock { layer, kind: None }],
        // [D, 2D]: consumes the concatenated [embed_normed ; hidden_normed].
        eh_proj: qt_load(shards, &p("eh_proj.weight"), d, 2 * d, dbits)?,
        enorm: ld(shards, &p("enorm.weight"))?,
        hnorm: ld(shards, &p("hnorm.weight"))?,
        mtp_norm: ld(shards, &p("shared_head.norm.weight"))?,
    }))
}

/// Load the Nemotron-H MTP head, which is **two** sublayers rather than GLM's one:
/// `mtp_hybrid_override_pattern == "*E"` — a NoPE-GQA attention block at layer index
/// `n_layers`, then a gateless latent-MoE block at `n_layers + 1`. Both load through the
/// ordinary [`load_layer_nemotron`] (same `mixer.*` names, same canonical
/// `input_layernorm.weight`), which is the whole point of the container mapping in
/// `convert::nemotron_container_name`: the head is just two more layers.
///
/// The fusion tensors (`eh_proj`/`enorm`/`hnorm`) and the head's final norm
/// (`shared_head.norm.weight`, from the source's `mtp.layers.1.final_layernorm`) all sit
/// at the head's BASE index, sharing GLM's names — so the two loaders differ only in the
/// sublayer list, not in the head's own plumbing.
///
/// Same completeness contract as GLM: probe every required tensor first and return
/// `None` on any gap. A half-loaded head drafts garbage, and the head's tensors span
/// several shards, so a truncated conversion is the realistic failure. The routed
/// experts of sublayer 1 stream through the provider at layer index `n_layers + 1`, so
/// only the first and last are probed (they land on different shards than the rest).
fn load_mtp_nemotron(
    shards: &colibri_safetensors::Shards,
    cfg: &Config,
    dbits: u32,
) -> Result<Option<MtpHead>, EngineError> {
    // The head's sublayer kinds come from the config, not from the tensor layout: a
    // checkpoint with no `mtp_hybrid_override_pattern` has no head to load.
    if cfg.mtp_layer_kind.is_empty() {
        return Ok(None);
    }
    let base = cfg.n_layers as usize;
    let last_e = (cfg.n_experts - 1).max(0) as usize;

    // Per-sublayer required tensors, derived from the sublayer's kind so a future head
    // shape (e.g. a Mamba sublayer) is a matter of extending this match, not the caller.
    let mut required: Vec<String> = vec![
        format!("model.layers.{base}.eh_proj.weight"),
        format!("model.layers.{base}.enorm.weight"),
        format!("model.layers.{base}.hnorm.weight"),
        format!("model.layers.{base}.shared_head.norm.weight"),
    ];
    for (j, kind) in cfg.mtp_layer_kind.iter().enumerate() {
        let li = base + j;
        required.push(format!("model.layers.{li}.input_layernorm.weight"));
        let m = |s: &str| format!("model.layers.{li}.mixer.{s}");
        match kind {
            LayerKind::Attn => required.extend([
                m("q_proj.weight"),
                m("k_proj.weight"),
                m("v_proj.weight"),
                m("o_proj.weight"),
            ]),
            LayerKind::Moe => required.extend([
                m("gate.weight"),
                m("gate.e_score_correction_bias"),
                m("fc1_latent_proj.weight"),
                m("fc2_latent_proj.weight"),
                m("shared_experts.up_proj.weight"),
                m("shared_experts.down_proj.weight"),
                m("experts.0.up_proj.weight"),
                m(&format!("experts.{last_e}.down_proj.weight")),
            ]),
            LayerKind::Mamba => required.extend([
                m("in_proj.weight"),
                m("out_proj.weight"),
                m("conv1d.weight"),
                m("A_log"),
                m("D"),
                m("dt_bias"),
                m("norm.weight"),
            ]),
            // Only Nemotron-H populates `mtp_layer_kind`, and it never emits KDA. If one
            // ever appears, this is not a Nemotron head — report "no head" via the
            // function's existing completeness contract rather than probing wrong names.
            LayerKind::Kda => return Ok(None),
        }
    }
    if !required.iter().all(|s| shards.has(s)) {
        return Ok(None);
    }

    let d = cfg.hidden as usize;
    let p = |s: &str| format!("model.layers.{base}.{s}");
    let mut blocks = Vec::with_capacity(cfg.mtp_layer_kind.len());
    for (j, &kind) in cfg.mtp_layer_kind.iter().enumerate() {
        // `load_layer_nemotron` normally reads `cfg.layer_kind[li]`, which does not extend
        // to the head — hence the explicit-kind entry point. `sparse` is unused on this
        // path (the Nemotron loader keys off the kind), so pass the kind's own answer.
        let mut layer = load_layer_nemotron_kind(shards, cfg, base + j, dbits, kind)?;
        // The `gpu_eligible` trap: a resident weight left off this list silently takes the
        // single-threaded CPU matmul (it cost 84% of an M3 prefill and 94% of Nemotron's
        // mamba before those were caught). The main stack is marked in `load_model_with`,
        // which iterates `model.layers` only — the head is not in that vector, so mark it
        // here or every draft step runs its projections on one core.
        mark_gpu_eligible(&mut layer);
        blocks.push(MtpBlock { layer, kind: Some(kind) });
    }
    Ok(Some(MtpHead {
        blocks,
        eh_proj: qt_load(shards, &p("eh_proj.weight"), d, 2 * d, dbits)?,
        enorm: ld(shards, &p("enorm.weight"))?,
        hnorm: ld(shards, &p("hnorm.weight"))?,
        mtp_norm: ld(shards, &p("shared_head.norm.weight"))?,
    }))
}

/// Mark a layer's resident weights as GPU-cacheable.
///
/// **The `gpu_eligible` trap**: a resident weight missing from these lists silently takes
/// `matmul_qt`'s single-threaded CPU path. That has cost 84% of an M3 prefill (q/k/v),
/// 94% of Nemotron's mamba total (in/out_proj) and ~25 s of a 40 s Nemotron MoE phase
/// (fc1/fc2). The tell is a phase total far exceeding the sum of its GPU sub-timers.
/// Marking is token-identical — it only changes which kernel runs. Audit this function
/// when adding any architecture.
///
/// Lives as a free function because the MTP head's sublayers are NOT in `model.layers`
/// and so are never reached by the loop in [`load_model_with`].
fn mark_gpu_eligible(l: &mut Layer) {
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
    // DSA indexer projections: batched in `indexer_forward`, so they want the GPU.
    // `matmul_qt`'s CPU path is single-threaded — a batched call on one core is
    // *slower* than the old per-query GEMVs spread across the indexer's worker
    // threads (measured: 46s -> 81s). On the GPU the batched form is the fast one.
    for t in [&mut l.ix_wk, &mut l.ix_wq, &mut l.ix_wp] {
        if let Some(t) = t {
            t.gpu_eligible = true;
        }
    }
    // MiniMax-M3 GQA attention projections (Option; absent on the GLM path). These
    // are the resident q/k/v projections and the block-sparse indexer projections —
    // dense int8 weights that route through `matmul_qt`. WITHOUT this they fall to
    // the single-threaded CPU path: the COLI_PROFILE breakdown measured the q/k/v
    // projections at 197 s of a 236 s / 512-tok prefill (84%!) — dwarfing both the
    // attention core (5.6 s) and expert I/O (31 s). `l.o` (o_proj) is already marked
    // above via the GLM list, which is why it was fast; these were simply omitted.
    for t in [&mut l.qkv_proj, &mut l.q_proj, &mut l.k_proj, &mut l.v_proj, &mut l.idx_q_proj, &mut l.idx_k_proj] {
        if let Some(t) = t {
            t.gpu_eligible = true;
        }
    }
    // Nemotron-H Mamba2 in_proj/out_proj — resident dense int8 weights routed through
    // `matmul_qt`. WITHOUT this they fall to the single-threaded CPU path: the mamba
    // COLI_PROFILE breakdown measured them at 9322 ms of a 9945 ms mamba total (94%),
    // dwarfing the selective scan (451 ms). Same omission as the M3 q/k/v projections
    // above; marking them eligible routes them to the GPU int8 matmul (token-identical).
    for t in [&mut l.mamba_in_proj, &mut l.mamba_out_proj] {
        if let Some(t) = t {
            t.gpu_eligible = true;
        }
    }
    // Nemotron-H latent-MoE fc1/fc2 — the resident dense projections that lift x into
    // the shared moe_latent space (hidden->1024) and back (1024->hidden) around the
    // routed experts. `nemotron_moe` runs them through `matmul_qt`; WITHOUT this they
    // take the single-threaded CPU path. The prefill profile measured moe=40 s of which
    // only 14.4 s was GPU (expert-load 7.7 + gpu-ffn 6.7) — the ~25 s remainder was
    // these two projections over 512 tok × 40 MoE layers on one core. Same omission /
    // same fix as the mamba proj above (the shared expert reuses up_proj/down_proj,
    // already eligible). Token-identical.
    for t in [&mut l.fc1_latent, &mut l.fc2_latent] {
        if let Some(t) = t {
            t.gpu_eligible = true;
        }
    }
    // Kimi-K3's own resident projections. K3 inherits most of this list already — its
    // KDA q/k/v reuse the GQA fields above, `o` comes from the GLM list, and the latent
    // MoE reuses Nemotron's fc1/fc2 — but these four have no analogue anywhere else and
    // would otherwise be the ONLY dense weights left on the single-threaded CPU matmul:
    //
    //   attn_gate (g_proj)  12288x7168  = 88.1M/layer x 93 = 8.19B params
    //   kda_f_b             12288x128   =  1.6M/layer x 69 = 0.11B
    //   kda_f_a               128x7168  =  0.9M/layer x 69 = 0.06B
    //   kda_b_proj             96x7168  =  0.7M/layer x 69 = 0.05B
    //
    // `attn_gate` is the one that matters: it runs on EVERY layer (both mixers carry the
    // output gate) and is the same order as `q_b`/`kv_b`. Leaving it off is the same
    // omission that cost 84% of an M3 prefill and 94% of Nemotron's mamba time — the
    // tell is a phase total far exceeding the sum of its GPU sub-timers.
    for t in [&mut l.attn_gate, &mut l.kda_b_proj, &mut l.kda_f_a, &mut l.kda_f_b] {
        if let Some(t) = t {
            t.gpu_eligible = true;
        }
    }
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
    // Record the SwiGLU variant for the FFN choke point (SiLU for GLM, clamped
    // OpenAI-SwiGLU for MiniMax-M3) before any forward pass runs.
    crate::moe::set_activation(&cfg);
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

    // Kimi-K3's model-level attention residual, applied after the last layer. Loaded
    // unconditionally for K3 rather than probed: a K3 container missing these cannot be
    // run (the stack's final mix would silently become "just the accumulator"), so a
    // missing-tensor error here is the right failure.
    let (output_attn_res_norm, output_attn_res_proj) = if cfg.arch == Arch::KimiK3 {
        (ld(&shards, "model.output_attn_res_norm.weight")?, ld(&shards, "model.output_attn_res_proj.weight")?)
    } else {
        (Vec::new(), Vec::new())
    };

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
        output_attn_res_norm,
        output_attn_res_proj,
        has_dsa,
        has_mtp: mtp.is_some(),
        mtp,
    };
    // Dense weights are resident for the model's lifetime → GPU-cacheable.
    //
    // EXCEPT on Kimi-K3. `try_matmul_qt` caches each eligible weight in a DEVICE buffer
    // keyed by its host pointer — and on GB10 "VRAM" is the same physical RAM as the
    // host, so that cache is a second copy of every resident weight, not a move. K3's
    // resident set is ~53 GB (vs GLM's ~19), so host + device is ~118 GB of 121 and
    // earlyoom SIGTERMs the process partway through the first forward pass. Measured:
    // RSS plateaued at 65 GB while MemAvailable kept falling to 4 GB — the giveaway that
    // the growth was allocations outside RSS.
    //
    // Every byte spent duplicating a resident weight is a byte the expert cache cannot
    // use, and on K3 the routed experts are 1347 GB against a cache that only ever holds
    // tens of GB — so cache coverage is the scarce resource, not matmul latency.
    // `COLI_DEVICE_WEIGHTS=0` disables it for every arch — the A/B handle for "does this
    // buy speed, or just cost the expert cache RAM it could have used?"
    let device_cache_weights = model.cfg.arch != Arch::KimiK3
        && std::env::var("COLI_DEVICE_WEIGHTS").ok().as_deref() != Some("0");
    if device_cache_weights {
        model.embed.gpu_eligible = true;
        model.lm_head.gpu_eligible = true;
        for l in &mut model.layers {
            mark_gpu_eligible(l);
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

#[cfg(test)]
mod gpu_eligible_tests {
    use super::*;

    /// `mark_gpu_eligible` must cover EVERY `Option<QTensor>` on a layer.
    ///
    /// The list is hand-maintained and enumerated per field, so a new arch's new fields
    /// are eligible only if someone remembers them. That omission has cost 84% of an M3
    /// prefill (q/k/v), 94% of Nemotron's mamba (in/out_proj), 62% of its MoE phase
    /// (fc1/fc2_latent) and would have cost K3 8.41B params (`attn_gate` and the KDA gate
    /// projections). Nothing type-checks it and nothing fails at runtime — the weight
    /// just silently takes the single-threaded CPU matmul.
    ///
    /// K3 does not currently call this (its resident set is too large to duplicate in
    /// GB10's unified memory), but the list must stay complete for the arches that do and
    /// for whenever a device-cache budget makes K3 eligible again.
    #[test]
    fn mark_gpu_eligible_covers_every_optional_projection() {
        let q = || Some(colibri_core::QTensor { o: 2, i: 2, ..Default::default() });
        let mut l = Layer {
            q_proj: q(), k_proj: q(), v_proj: q(), qkv_proj: q(),
            idx_q_proj: q(), idx_k_proj: q(),
            mamba_in_proj: q(), mamba_out_proj: q(),
            fc1_latent: q(), fc2_latent: q(),
            attn_gate: q(), kda_b_proj: q(), kda_f_a: q(), kda_f_b: q(),
            ix_wk: q(), ix_wq: q(), ix_wp: q(),
            ..Default::default()
        };
        mark_gpu_eligible(&mut l);
        let opts: [(&str, &Option<colibri_core::QTensor>); 17] = [
            ("q_proj", &l.q_proj), ("k_proj", &l.k_proj), ("v_proj", &l.v_proj),
            ("qkv_proj", &l.qkv_proj), ("idx_q_proj", &l.idx_q_proj),
            ("idx_k_proj", &l.idx_k_proj), ("mamba_in_proj", &l.mamba_in_proj),
            ("mamba_out_proj", &l.mamba_out_proj), ("fc1_latent", &l.fc1_latent),
            ("fc2_latent", &l.fc2_latent), ("attn_gate", &l.attn_gate),
            ("kda_b_proj", &l.kda_b_proj), ("kda_f_a", &l.kda_f_a),
            ("kda_f_b", &l.kda_f_b), ("ix_wk", &l.ix_wk), ("ix_wq", &l.ix_wq),
            ("ix_wp", &l.ix_wp),
        ];
        for (name, t) in opts {
            assert!(
                t.as_ref().expect("fixture sets every field").gpu_eligible,
                "{name} is missing from mark_gpu_eligible"
            );
        }
    }
}

#[cfg(test)]
mod kimi_container_tests {
    use super::*;

    /// Load real Kimi-K3 layers out of a converted container.
    ///
    /// This is the check that `load_layer_kimi`'s tensor names and dimensions match what
    /// `coli convert` actually writes — the two are written from the same understanding of
    /// the checkpoint, so a shared misreading would pass every unit test and only surface
    /// here. Ignored by default because it needs a container on disk; on a box with one:
    ///
    /// ```text
    /// COLI_K3_CONTAINER=$HOME/models/Kimi-K3-slice-container \
    ///   cargo test -p colibri-engine --lib kimi_loads_real -- --ignored --nocapture
    /// ```
    ///
    /// The 4-shard slice container holds layers 0 (KDA + dense MLP), 1 (KDA + latent MoE)
    /// and 3 (gated MLA + latent MoE), so one run covers both mixers and both FFN kinds.
    #[test]
    #[ignore]
    fn kimi_loads_real_container_layers() {
        let Ok(dir) = std::env::var("COLI_K3_CONTAINER") else {
            eprintln!("COLI_K3_CONTAINER unset — skipping");
            return;
        };
        let cfg = Config::load(&dir).expect("load config.json");
        assert_eq!(cfg.arch, Arch::KimiK3, "container is not Kimi-K3");
        let shards = colibri_safetensors::Shards::open(&dir).expect("open shards");

        let d = cfg.hidden as usize;
        let c = cfg.kda_n_heads as usize * cfg.kda_head_dim as usize;
        for li in [0usize, 1, 3] {
            let l = load_layer_kimi(&shards, &cfg, li, 8)
                .unwrap_or_else(|e| panic!("layer {li} failed to load: {e}"));

            // Shared by both mixers.
            assert_eq!((l.o.o as usize, l.o.i as usize), (d, c), "L{li} o_proj");
            let g = l.attn_gate.as_ref().unwrap_or_else(|| panic!("L{li} has no g_proj"));
            assert_eq!((g.o as usize, g.i as usize), (c, d), "L{li} g_proj");
            assert_eq!(l.in_ln.len(), d, "L{li} in_ln");
            assert_eq!(l.post_ln.len(), d, "L{li} post_ln");
            // The attention-residual pair: a norm plus a single-row projection.
            assert_eq!(l.attn_res_norm.len(), d, "L{li} attn_res_norm");
            assert_eq!(l.attn_res_proj.len(), d, "L{li} attn_res_proj is [1, hidden]");
            assert_eq!(l.mlp_res_norm.len(), d, "L{li} mlp_res_norm");
            assert_eq!(l.mlp_res_proj.len(), d, "L{li} mlp_res_proj is [1, hidden]");

            match cfg.layer_kind[li] {
                LayerKind::Kda => {
                    let (nh, hd) = (cfg.kda_n_heads as usize, cfg.kda_head_dim as usize);
                    for (nm, t) in [("q", &l.q_proj), ("k", &l.k_proj), ("v", &l.v_proj)] {
                        let t = t.as_ref().unwrap_or_else(|| panic!("L{li} no {nm}_proj"));
                        assert_eq!((t.o as usize, t.i as usize), (c, d), "L{li} {nm}_proj");
                    }
                    let b = l.kda_b_proj.as_ref().expect("b_proj");
                    assert_eq!((b.o as usize, b.i as usize), (nh, d), "L{li} b_proj");
                    // The forget gate is factored through a rank derived from the
                    // checkpoint. NOTE `fa.o == fb.i` is NOT sufficient on its own: when
                    // the rank came from the container's flat `shape[0]` both were
                    // 917504, so they agreed with each other and this test passed while
                    // the model was unloadable. Check the PAYLOAD against the dims —
                    // that is the invariant a wrong rank actually violates.
                    let (fa, fb) = (l.kda_f_a.as_ref().unwrap(), l.kda_f_b.as_ref().unwrap());
                    assert_eq!(fa.i as usize, d, "L{li} f_a input");
                    assert_eq!(fb.o as usize, c, "L{li} f_b output");
                    assert_eq!(fa.o, fb.i, "L{li} f_a/f_b rank must agree");
                    for (nm, t) in [("f_a", fa), ("f_b", fb), ("b", b)] {
                        if t.fmt_code == 1 {
                            assert_eq!(
                                t.q8.len(),
                                (t.o as usize) * (t.i as usize),
                                "L{li} {nm} int8 payload does not match {}x{}",
                                t.o,
                                t.i
                            );
                            assert_eq!(t.s.len(), t.o as usize, "L{li} {nm} one scale per row");
                        }
                    }
                    // `[C, 1, k]` read flat is `[C, k]`.
                    let k = cfg.kda_d_conv as usize;
                    for (nm, v) in [("q", &l.kda_conv_q), ("k", &l.kda_conv_k), ("v", &l.kda_conv_v)]
                    {
                        assert_eq!(v.len(), c * k, "L{li} {nm}_conv1d");
                    }
                    assert_eq!(l.kda_a_log.len(), hd, "L{li} A_log");
                    assert_eq!(l.kda_dt_bias.len(), c, "L{li} dt_bias");
                    assert_eq!(l.kda_o_norm.len(), hd, "L{li} o_norm");
                    assert!(l.q_a.qf.is_empty() && l.q_a.q8.is_empty(), "L{li} KDA has no MLA q_a");
                }
                _ => {
                    let (nh, ql, kl) =
                        (cfg.n_heads as usize, cfg.q_lora as usize, cfg.kv_lora as usize);
                    assert_eq!((l.q_a.o as usize, l.q_a.i as usize), (ql, d), "L{li} q_a");
                    assert_eq!(
                        (l.q_b.o as usize, l.q_b.i as usize),
                        (nh * (cfg.qk_nope + cfg.qk_rope) as usize, ql),
                        "L{li} q_b"
                    );
                    assert_eq!(
                        (l.kv_a.o as usize, l.kv_a.i as usize),
                        (kl + cfg.qk_rope as usize, d),
                        "L{li} kv_a"
                    );
                    assert_eq!(
                        (l.kv_b.o as usize, l.kv_b.i as usize),
                        (nh * (cfg.qk_nope + cfg.v_head) as usize, kl),
                        "L{li} kv_b"
                    );
                    assert_eq!(l.q_a_ln.len(), ql, "L{li} q_a_layernorm");
                    assert_eq!(l.kv_a_ln.len(), kl, "L{li} kv_a_layernorm");
                    assert!(l.kda_a_log.is_empty(), "L{li} MLA must not load KDA vectors");
                }
            }

            // FFN: the `first_dense` prefix is dense, the rest is latent MoE.
            if (li as i32) < cfg.first_dense {
                assert!(!l.sparse, "L{li} should be dense");
                let di = cfg.dense_inter as usize;
                assert_eq!((l.gate_proj.o as usize, l.gate_proj.i as usize), (di, d));
                assert_eq!((l.down_proj.o as usize, l.down_proj.i as usize), (d, di));
            } else {
                assert!(l.sparse, "L{li} should be MoE");
                let (dl, si) = (cfg.moe_latent as usize, cfg.shared_inter as usize);
                assert_eq!(l.router.len(), cfg.n_experts as usize * d, "L{li} router");
                assert_eq!(l.router_bias.len(), cfg.n_experts as usize, "L{li} router bias");
                let f1 = l.fc1_latent.as_ref().expect("fc1_latent");
                let f2 = l.fc2_latent.as_ref().expect("fc2_latent");
                assert_eq!((f1.o as usize, f1.i as usize), (dl, d), "L{li} fc1_latent");
                assert_eq!((f2.o as usize, f2.i as usize), (d, dl), "L{li} fc2_latent");
                assert_eq!(l.routed_expert_norm.len(), dl, "L{li} routed_expert_norm");
                // Shared experts ship as ONE fused pair, `n_shared * moe_inter` wide.
                assert_eq!(si, cfg.n_shared as usize * cfg.moe_inter as usize);
                assert_eq!((l.sh_gate.o as usize, l.sh_gate.i as usize), (si, d), "L{li} sh_gate");
                assert_eq!((l.sh_down.o as usize, l.sh_down.i as usize), (d, si), "L{li} sh_down");
            }
            eprintln!("layer {li:>2} ({:?}) loaded OK", cfg.layer_kind[li]);
        }
    }
}
