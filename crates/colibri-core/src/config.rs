//! Model hyperparameters — port of the `Cfg` struct and `load_cfg` from
//! `c/glm.c`.
//!
//! Loaded from a snapshot's `config.json`. The validation ranges (`CKR` in C)
//! are a single choke point: `config.json` arrives from untrusted mirrors, so
//! hostile dimensions must not pass this point and reach a downstream alloc.
//!
//! Two architectures are supported, discriminated by [`Config::arch`]:
//!   - [`Arch::GlmMoeDsa`] — GLM-5.2: MLA attention + DSA lightning indexer.
//!   - [`Arch::MinimaxM3`] — MiniMax-M3: standard GQA (partial RoPE, per-head
//!     QK-norm), Gemma-norm, clamped SwiGLU, sigmoid+bias MoE router. The
//!     GQA head geometry reuses the `qk_nope`/`qk_rope`/`v_head` fields
//!     (`qk_rope` = the rotary sub-dim, `qk_nope` = head_dim − rotary).

use colibri_json::Json;
use std::path::Path;

pub const MAX_STOP_IDS: usize = 8;
pub const MAX_LAYERS_IDX: usize = 128;

/// Which model architecture a [`Config`] describes. Selects the attention core,
/// the MoE router, the activation, and the norm variant in the forward pass.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Arch {
    /// GLM-5.2: Multi-head Latent Attention + DSA sparse indexer.
    GlmMoeDsa,
    /// MiniMax-M3: grouped-query attention (partial RoPE, per-head QK-norm),
    /// Gemma-style RMSNorm, clamped OpenAI-SwiGLU, sigmoid+bias top-k router.
    MinimaxM3,
    /// MiniMax-M2: same GQA family as M3 but a flat (non-VL) config — plain SwiGLU
    /// (silu), standard RMSNorm, no shared expert, all-MoE, per-layer QK-norm,
    /// sigmoid+bias top-k router. Shares the M3 GQA forward path (dims differ).
    MinimaxM2,
    /// Nemotron-H: hybrid Mamba2 (SSM) + GQA + latent-MoE. The layer sequence is
    /// heterogeneous by index — see [`Config::layer_kind`]. Mamba2 layers carry a
    /// recurrent conv+ssm state instead of a KV cache; MoE experts run in a low-rank
    /// latent space (`moe_latent`) with a gateless ReLU² activation. Only its 8
    /// attention layers are GQA, so it is NOT blanket [`Arch::is_gqa`] — the
    /// per-layer `layer_kind` dispatch routes each layer to its mixer.
    NemotronH,
    /// Kimi-K3: hybrid Kimi Delta Attention (KDA, a linear/delta-rule mixer) + gated
    /// MLA, with a latent-MoE FFN on every layer after the first.
    ///
    /// Hybrid on the **mixer axis only** — unlike Nemotron-H, where a layer is *either*
    /// Mamba, attention, or MoE, every K3 layer carries both a mixer and an FFN. So
    /// [`Config::layer_kind`] holds [`LayerKind::Kda`]/[`LayerKind::Attn`] and never
    /// [`LayerKind::Moe`]; the MoE layers are the `first_dense` prefix rule instead.
    /// Ask [`Config::moe_layers`] for that count, never `layer_kind` directly.
    KimiK3,
    /// DeepSeek-V4-Flash: MoE on every layer, latent attention with **O-LoRA**, and
    /// **Hyper-Connections** in place of the residual stream.
    ///
    /// Hyper-Connections are the load-bearing difference and the reason this cannot reuse
    /// another arch's block loop. Instead of `x = x + f(norm(x))`, the inter-block state is
    /// `hc_mult` (4) copies of the hidden state, `[b, s, 4, hidden]`. Each block does
    /// `hc_pre` (collapse 4 -> 1 by a learned weighted sum) -> norm -> mixer -> `hc_post`
    /// (expand 1 -> 4, mixing the previous copies through a 4x4 matrix), twice: once around
    /// attention and once around the FFN. The collapse/expand weights are produced per
    /// token by a Sinkhorn normalisation (20 iterations) of a 24-wide projection of the
    /// flattened state — see `hc_split_sinkhorn` in the checkpoint's `inference/kernel.py`.
    ///
    /// Crucially the mixer itself still sees `[b, s, hidden]`, because `hc_pre` collapses
    /// *before* `attn_norm` and `hc_post` expands *after* the block. So attention, the MoE
    /// and every scratch buffer keep their usual shape; only the carried residual is 4-wide
    /// (+201 MB at a 4096-token prefill). Do NOT widen the rest of the engine for this.
    ///
    /// Routed experts are natively **MXFP4** (`fmt=6`, block-32 E8M0), the same format as
    /// [`Arch::KimiK3`] and for the same reason: the checkpoint is trained in it, so a
    /// requantisation to NVFP4 is measured pure loss (6.40% rel-RMS) *and* 5.9% more bytes.
    /// Dense/resident weights are fp8 e4m3 with **128x128** E8M0 block scales — a third
    /// scale layout, neither NVFP4's `ceil(I/16)` nor MXFP4's `ceil(I/32)`.
    DeepseekV4,
    /// Maple (`deepgrove/maple-preview`): the GQA family again — 16Q/4KV, `head_dim` 128,
    /// partial RoPE 64, per-layer QK-norm, clamped SwiGLU, 256 experts top-8, no shared
    /// expert, all-MoE — so it reuses M2/M3's attention and MoE paths. Three things are
    /// its own, and all three are correctness-critical:
    ///
    /// 1. **A softmax router**, not sigmoid+bias. Selection is unaffected (softmax is
    ///    monotone), but the *weights* are not: renormalising the chosen softmax scores
    ///    is `exp(z_i) / Σ_{j∈S} exp(z_j)`, which is a softmax over the top-k logits
    ///    alone. See [`RouterScore`] — and note `sigmoid_route` is NOT that switch.
    /// 2. **A 3:1 sliding/full attention interleave** (`layer_swa`): 18 of 24 layers see
    ///    only the last [`Config::swa_window`] (512) keys. This is architectural, not a
    ///    retrofit — the negative recorded for bolting SWA onto GLM does not apply.
    /// 3. **NoPE on the global layers** ([`Config::nope_on_global`]): the reference
    ///    applies RoPE *only* when a layer has a sliding window
    ///    (`modeling_maple.py`, `MapleAttention::forward`), so the 6 full-attention
    ///    layers carry no positional encoding at all.
    ///
    /// Every expert and attention projection in the released checkpoint is **exactly
    /// per-row ternary** — `{-s, 0, +s}` with one BF16 scale per output row — so they
    /// store as `fmt 3` int2 (`value = field - 2`, using only `{-1, 0, +1}`) **bit-for-bit**,
    /// not approximately. The router, norms, embeddings and `lm_head` are genuinely dense
    /// and are NOT ternary; converting them as if they were is silent quality loss.
    Maple,
}

/// How a MoE router turns logits into the weights of the chosen experts.
///
/// This exists because [`Config::sigmoid_route`] is **not** the switch it appears to be:
/// it is parsed from `scoring_func` and then consumed by nothing in the workspace, while
/// `moe::route` applies `sigmoid` unconditionally. GLM carries `sigmoid_route == false`
/// and runs the sigmoid path regardless — so "wiring up the existing flag" would silently
/// change a shipped model's expert weights. Every pre-Maple arch is [`RouterScore::Sigmoid`]
/// here by construction, which keeps them bit-identical.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum RouterScore {
    /// `sigmoid(logit)`, selected on `sigmoid(logit) + bias`. Every arch before Maple.
    Sigmoid,
    /// `softmax(logit)` over all experts, then top-k, then renormalise over the chosen
    /// set — algebraically a softmax over the top-k logits. No routing bias.
    Softmax,
}

/// Per-layer mixer type for a hybrid architecture. Homogeneous arches leave
/// [`Config::layer_kind`] empty and never consult this.
///
/// Two producers, and they populate different subsets:
/// - Nemotron-H, from `hybrid_override_pattern` (`M`→`Mamba`, `E`→`Moe`, `*`→`Attn`).
///   A layer is exactly one of the three — its MoE layers hold no mixer at all.
/// - Kimi-K3, from `linear_attn_config` (`Kda`/`Attn` only). Every K3 layer *also*
///   has an FFN, which this axis does not describe — see [`Config::moe_layers`].
///
/// So `layer_kind` answers "what mixer does layer `i` run", and only Nemotron-H's
/// encoding additionally answers "does layer `i` hold experts". Anything needing the
/// latter must go through [`Config::moe_layers`].
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum LayerKind {
    /// Mamba2 selective-scan (state-space) mixer.
    Mamba,
    /// Grouped-query attention mixer. Also the gated-MLA layers on Kimi-K3 — the
    /// distinction that matters to the KV cache is "holds a growing KV cache", and
    /// both do.
    Attn,
    /// Mixture-of-experts (routed + shared, latent-space) mixer.
    Moe,
    /// Kimi Delta Attention: a linear-attention mixer carrying a fixed-size recurrent
    /// state (per-head delta-rule matrix + short causal-conv history) instead of a
    /// context-growing KV cache. O(1) in context, like [`LayerKind::Mamba`] — and, like
    /// it, invisible to per-token accounting, so it belongs in `KvCache::fixed_bytes`.
    Kda,
}

impl Arch {
    /// The GQA family (MiniMax M3/M2, Maple): standard q/k/v projections + a KV cache of
    /// `n_kv_heads`, as opposed to GLM's MLA latent attention + DSA indexer. The
    /// engine's attention/KV/loader paths branch on this rather than a specific
    /// variant so every GQA model shares one code path.
    ///
    /// **This is a closed set that a new arch does not join by default**, and the default
    /// is the dangerous one: a `false` here sends a GQA model down the MLA path, which is
    /// a build error nowhere and a wrong answer everywhere. Maple builds and runs with it
    /// omitted. Whenever an `Arch` variant is added, this predicate and
    /// [`Arch::routed_experts_are_latent`] both have to be revisited deliberately —
    /// `matches!` gives no compiler help.
    pub fn is_gqa(&self) -> bool {
        matches!(self, Arch::MinimaxM3 | Arch::MinimaxM2 | Arch::Maple)
    }

    /// Whether routed experts live in the low-rank `moe_latent` space rather than at
    /// the model `hidden`. Nemotron-H and Kimi-K3 both bottleneck the MoE block
    /// (`fc1_latent` down, experts, `fc2_latent` back up), so their expert tensors are
    /// `[moe_inter, moe_latent]` / `[moe_latent, moe_inter]`.
    ///
    /// This is a named predicate, not an inline `matches!`, because it has to agree in
    /// two far-apart places: the loader that reads expert tensors off disk
    /// (`expert_outer_dim`) and the MoE block that computes with them. When they
    /// disagree the expert is read at the wrong width — a shape error at best, and
    /// silently wrong rows at worst.
    pub fn routed_experts_are_latent(&self) -> bool {
        matches!(self, Arch::NemotronH | Arch::KimiK3)
    }
}

/// GLM-5.2 / MiniMax-M3 hyperparameters.
#[derive(Debug, Clone)]
pub struct Config {
    pub hidden: i32,
    pub n_layers: i32,
    pub n_heads: i32,
    pub n_experts: i32,
    pub topk: i32,
    pub moe_inter: i32,
    pub dense_inter: i32,
    pub first_dense: i32,
    pub q_lora: i32,
    /// DeepSeek-V4 output LoRA: rank per group (`o_lora_rank`) and the number of groups
    /// (`o_groups`). V4 replaces the plain o_proj with `o_a` [g*rank, n_heads*head_dim/g]
    /// then `o_b` [hidden, g*rank]. 0 on every other arch.
    ///
    /// These are carried in Config rather than read back from the weights because the
    /// container stores quantised tensors as FLAT blobs — `o_a` arrives as [33554432] with
    /// no shape to recover, and 8192x4096 is not the only factorisation of it.
    pub o_lora: i32,
    pub o_groups: i32,
    pub kv_lora: i32,
    pub qk_nope: i32,
    pub qk_rope: i32,
    /// derived: `qk_nope + qk_rope`
    pub qk_head: i32,
    pub v_head: i32,
    pub n_shared: i32,
    pub vocab: i32,
    /// model's max context (`max_position_embeddings`); 0 if the config omits it
    pub max_ctx: i32,
    pub n_group: i32,
    pub topk_group: i32,
    pub norm_topk: bool,
    /// stop tokens (GLM-5.2 has three: endoftext, user, observation)
    pub stop_ids: Vec<i32>,
    /// DSA lightning indexer params
    pub index_topk: i32,
    pub index_nh: i32,
    pub index_hd: i32,
    /// MiniMax-M3 block-sparse indexer: keys are max-pooled into `index_block_size`
    /// blocks; each query keeps the top `index_topk_blocks` scored blocks plus the
    /// last `index_local_blocks` blocks (always visible). 0 = no block-sparse (GLM).
    pub index_block_size: i32,
    pub index_topk_blocks: i32,
    pub index_local_blocks: i32,
    /// per-layer indexer type: GLM `true` = FULL DSA layer; MiniMax-M3 `true` = a
    /// block-sparse attention layer (from `sparse_attention_freq`).
    pub idx_type: Vec<bool>,
    pub eps: f32,
    pub theta: f32,
    pub attn_scale: f32,
    pub routed_scale: f32,

    // ---- architecture discriminator + MiniMax-M3-specific fields ----
    // (GLM leaves these at the defaults set in `from_json_glm`.)
    /// Which architecture this config describes.
    pub arch: Arch,
    /// GQA key/value head count (MiniMax-M3). For GLM/MLA this mirrors `n_heads`.
    pub n_kv_heads: i32,
    /// Shared-expert intermediate size. MiniMax-M3 sets it explicitly
    /// (`shared_intermediate_size`); GLM derives `n_shared * moe_inter`.
    pub shared_inter: i32,
    /// Per-head QK RMSNorm applied before RoPE (MiniMax-M3 `use_qk_norm`).
    pub qk_norm: bool,
    /// Gemma-style `(1 + weight)` RMSNorm (MiniMax-M3 `use_gemma_norm`).
    pub gemma_norm: bool,
    /// Clamped OpenAI-SwiGLU activation (MiniMax-M3 `hidden_act == "swigluoai"`);
    /// `false` = plain SiLU-gated SwiGLU (GLM).
    pub swiglu_oai: bool,
    /// SwiGLU gate scale (`swiglu_alpha`, MiniMax-M3) — used only when `swiglu_oai`.
    pub swiglu_alpha: f32,
    /// SwiGLU clamp limit (`swiglu_limit`, MiniMax-M3) — used only when `swiglu_oai`.
    pub swiglu_limit: f32,
    /// DeepSeek-V4 Hyper-Connections. `hc_mult` copies of the hidden state replace the
    /// residual, so the stream is `[s, hc_mult, hidden]` and **every activation buffer is
    /// `hc_mult` times wider**. 0 for every other arch, which is also the "no HC" switch:
    /// `hc_mult == 0` must behave exactly as a plain residual.
    ///
    /// `hc_sinkhorn_iters` is the normalisation loop count inside `hc_split_sinkhorn`, and
    /// `hc_eps` is the floor added to the Sinkhorn/sigmoid weights — NOT the RMS epsilon.
    /// The reference uses `norm_eps` for the rsqrt inside `hc_pre` and `hc_eps` only for
    /// the weights; conflating them is silent and shifts every mixing weight slightly.
    /// DeepSeek-V4 Compressor: per-layer `compress_ratio`, 0 where the layer has none
    /// (V4: layers 0-1 and 42+ are 0, the other 41 alternate 4 and 128). A ratio of 4
    /// additionally turns on OVERLAPPING windows, which doubles the projection width —
    /// so this vector determines tensor SHAPES, not just behaviour, and a wrong entry is
    /// a load failure rather than a silent quality loss.
    pub compress_ratios: Vec<i32>,
    /// The Compressor's rope base, SEPARATE from `theta`: V4 uses 160000 here against
    /// 10000 for attention. Reusing `theta` would place every compressed block wrongly.
    pub compress_theta: f32,
    /// DeepSeek-V4 hash routing: the FIRST `num_hash_layers` (3) MoE layers pick their
    /// experts from a `tid2eid[token_id]` table instead of by top-k score. Those layers
    /// still run the router matmul — it supplies the WEIGHTS — and ship no bias at all,
    /// because a bias only ever shifts a comparison and there is no comparison here.
    /// 0 on every other arch.
    pub n_hash_layers: i32,
    /// DeepSeek-V4 DSpark (speculative drafting, stored under `mtp.*`). All 0/empty on
    /// every other arch. `dspark_targets` are the MAIN layers whose hidden states are
    /// concatenated into stage 0's `main_proj` — they are sources, not DSpark layers.
    pub dspark_block: i32,
    pub dspark_noise_id: i32,
    pub markov_rank: i32,
    pub dspark_targets: Vec<i32>,
    pub hc_mult: i32,
    pub hc_sinkhorn_iters: i32,
    pub hc_eps: f32,
    /// DeepSeek-V4 sliding window (`sliding_window`, 128). The raw KV cache is a RING
    /// BUFFER of exactly this many entries — older context is reachable only through the
    /// Compressor. So a build without the Compressor is exact for prompts up to this
    /// length and wrong beyond it, rather than approximate everywhere.
    pub window: i32,
    /// Sigmoid expert scoring with an additive routing bias (MiniMax-M3
    /// `scoring_func == "sigmoid"` + `e_score_correction_bias`); `false` = GLM.
    ///
    /// **Parsed but not consumed.** `moe::route` sigmoids unconditionally; the live
    /// switch is [`Config::router_score`]. Kept because it records what the source
    /// config said, but do not branch on it — see [`RouterScore`].
    pub sigmoid_route: bool,
    /// How the router scores experts. See [`RouterScore`] for why this is a separate
    /// axis from `sigmoid_route` rather than a use of it.
    pub router_score: RouterScore,

    // ---- Maple (sliding/full attention interleave) fields ----
    // Homogeneous-attention arches leave `layer_swa` empty and `swa_window` 0.
    /// Sliding-attention window in tokens (Maple `sliding_window`, 512). A layer with
    /// [`Config::layer_is_swa`] attends only to the last `swa_window` keys, inclusive of
    /// the query's own position.
    ///
    /// Deliberately NOT [`Config::window`], which is DeepSeek-V4's raw-KV **ring size** —
    /// a memory-tier knob, not a mask. Two different quantities sharing one field is the
    /// shape that made M2.7 look unstable for a week; they stay apart.
    pub swa_window: i32,
    /// Per-layer attention span, one entry per layer, from `layer_types`
    /// (`sliding_attention` → `true`, `full_attention` → `false`). Empty means every
    /// layer is full attention, which is every arch before Maple.
    pub layer_swa: Vec<bool>,
    /// Maple `nope_on_global_attention`: apply RoPE **only** on sliding layers, leaving
    /// the full-attention layers with no positional encoding. Ask
    /// [`Config::layer_uses_rope`] rather than combining this with `layer_swa` at the
    /// call site.
    pub nope_on_global: bool,

    // ---- Nemotron-H (hybrid Mamba2/GQA/latent-MoE) fields ----
    // (GLM/MiniMax leave `layer_kind` empty and the Mamba/latent fields at 0.)
    /// Per-layer mixer kind, one entry per layer, from `hybrid_override_pattern`.
    /// Empty for homogeneous arches; drives per-layer dispatch when non-empty.
    pub layer_kind: Vec<LayerKind>,
    /// Per-SUBLAYER mixer kind of the MTP speculative head, from
    /// `mtp_hybrid_override_pattern` (Nemotron-H ships `"*E"`: one attention sublayer
    /// then one latent-MoE sublayer). Empty on every other arch — GLM/M3 heads are a
    /// single sparse block whose shape is implied by the arch, not by a pattern.
    ///
    /// Kept SEPARATE from [`Config::layer_kind`] rather than appended to it: that vector
    /// is `num_hidden_layers` long by contract, and both the KV accounting
    /// (`KvCache::kv_layers`, `fixed_bytes`) and the loader iterate it to describe the
    /// *main stack*. Appending the head would silently inflate every one of those.
    pub mtp_layer_kind: Vec<LayerKind>,
    /// Mamba2 SSM state size (`ssm_state_size`); 0 for non-Mamba arches.
    pub mamba_d_state: i32,
    /// Mamba2 causal-conv kernel width (`conv_kernel`).
    pub mamba_d_conv: i32,
    /// Mamba2 number of SSM heads (`mamba_num_heads`).
    pub mamba_n_heads: i32,
    /// Mamba2 per-head dim (`mamba_head_dim`).
    pub mamba_head_dim: i32,
    /// Mamba2 number of B/C groups (`n_groups`), broadcast to `mamba_n_heads`.
    pub mamba_n_groups: i32,
    /// Mamba2 inner width = `mamba_n_heads * mamba_head_dim`.
    pub mamba_inter: i32,
    /// Mamba2 chunk size for the parallel prefill scan (`chunk_size`).
    pub mamba_chunk: i32,
    /// MoE latent bottleneck dim (`moe_latent_size`); experts run in this space.
    /// 0 = experts run directly in `hidden` (GLM/MiniMax).
    pub moe_latent: i32,
    /// ReLU² expert activation (`mlp_hidden_act == "relu2"`): gateless `down(relu(up·x)²)`.
    /// `false` = the existing gated SwiGLU path.
    pub relu2: bool,
    /// Mamba2 lower clamp on the discretized step `dt` (`time_step_min`); the scan
    /// applies `dt = max(softplus(dt+dt_bias), dt_min)`. 0.0 for non-Mamba arches.
    pub mamba_dt_min: f32,

    // ---- Kimi-K3 (hybrid KDA / gated-MLA) fields ----
    // (0 on every other arch; only read when `arch == Arch::KimiK3`.)
    /// KDA head count (`linear_attn_config.num_heads`).
    pub kda_n_heads: i32,
    /// KDA per-head dim (`linear_attn_config.head_dim`). The recurrent delta-rule
    /// state is `[kda_n_heads, kda_head_dim, kda_head_dim]` per KDA layer — the
    /// square `d_k x d_v` association matrix a delta rule carries between steps.
    pub kda_head_dim: i32,
    /// Kimi-K3 `hidden_act == "situ"`: the gated activation in `math::situ`. Applies to
    /// the dense MLP, the shared experts AND the routed experts.
    pub situ: bool,
    /// `activation_situ_beta` (gate clamp) and `activation_situ_linear_beta` (up clamp).
    pub situ_beta: f32,
    pub situ_linear_beta: f32,
    /// MLA runs WITHOUT rotary embeddings (`mla_use_nope`). The `qk_rope_head_dim`
    /// split still exists in the projections — those dims are simply carried
    /// un-rotated — so this cannot be inferred from the shapes. Kimi-K3 asserts it.
    pub mla_nope: bool,
    /// KDA short causal-conv kernel width (`linear_attn_config.short_conv_kernel_size`).
    /// The conv history is `[kda_d_conv, 3 * kda_n_heads * kda_head_dim]` — q, k and v
    /// each carry their own `*_conv1d`, hence the factor of 3.
    pub kda_d_conv: i32,
    /// `attn_res_block_size` (12 on K3): how often the stack snapshots its accumulator
    /// into the attention-residual candidate set — every `n`-th layer, `layer_idx % n
    /// == 0`. **0 means the arch has no attention residuals at all**, which is what
    /// every non-K3 arch sets and what the reference's `getattr(config,
    /// "attn_res_block_size", None) is not None` tests for. The driver divides by it,
    /// so 0 must never reach a K3 forward pass — `from_json_kimi` range-checks it.
    pub attn_res_block_size: i32,
}

/// Error from loading/validating a config.
#[derive(Debug)]
pub enum ConfigError {
    Io(std::io::Error),
    Parse(String),
    /// A field is outside its accepted `[lo, hi]` range (the `CKR` checks).
    Range {
        name: &'static str,
        value: i64,
        lo: i64,
        hi: i64,
    },
    Unsupported(String),
}

impl std::fmt::Display for ConfigError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            ConfigError::Io(e) => write!(f, "io error: {e}"),
            ConfigError::Parse(s) => write!(f, "parse error: {s}"),
            ConfigError::Range {
                name,
                value,
                lo,
                hi,
            } => write!(f, "config: {name}={value} is outside [{lo},{hi}]"),
            ConfigError::Unsupported(s) => write!(f, "unsupported config: {s}"),
        }
    }
}

impl std::error::Error for ConfigError {}

/// Read integer field `k` from object `r` (0 if absent/non-integer).
fn gi_in(r: &Json, k: &str) -> i32 {
    r.get(k).and_then(Json::as_i64).unwrap_or(0) as i32
}

macro_rules! ckr {
    ($name:literal, $v:expr, $lo:expr, $hi:expr) => {{
        let v = $v as i64;
        if v < ($lo as i64) || v > ($hi as i64) {
            return Err(ConfigError::Range {
                name: $name,
                value: v,
                lo: $lo as i64,
                hi: $hi as i64,
            });
        }
    }};
}

/// Decode a Nemotron-H mixer pattern string (`M`→Mamba, `E`→MoE, `*`→attention) into
/// one [`LayerKind`] per character. `field` names the config key in the error message,
/// since the same grammar describes both the main stack (`hybrid_override_pattern`) and
/// the MTP head (`mtp_hybrid_override_pattern`). An empty/absent string yields an empty
/// vec — for the head that simply means "no MTP module in this checkpoint".
fn parse_hybrid_pattern(pattern: &str, field: &str) -> Result<Vec<LayerKind>, ConfigError> {
    pattern
        .chars()
        .map(|ch| match ch {
            'M' => Ok(LayerKind::Mamba),
            'E' => Ok(LayerKind::Moe),
            '*' => Ok(LayerKind::Attn),
            other => Err(ConfigError::Unsupported(format!(
                "nemotron_h: unknown {field} char '{other}'"
            ))),
        })
        .collect()
}

/// Collect `eos_token_id` (scalar or array) into `out`.
fn parse_stop_ids(r: &Json, out: &mut Vec<i32>) {
    match r.get("eos_token_id") {
        Some(Json::Num(n)) => out.push(*n as i32),
        Some(Json::Arr(a)) => {
            for v in a.iter().take(MAX_STOP_IDS) {
                if let Some(id) = v.as_i64() {
                    out.push(id as i32);
                }
            }
        }
        _ => {}
    }
}

impl Config {
    /// Load and validate `<snap>/config.json`.
    pub fn load(snap: impl AsRef<Path>) -> Result<Config, ConfigError> {
        let snap = snap.as_ref();
        let path = snap.join("config.json");
        let text = std::fs::read_to_string(&path).map_err(ConfigError::Io)?;
        let root = Json::parse(&text)
            .ok_or_else(|| ConfigError::Parse(format!("{}: empty or invalid", path.display())))?;
        let mut cfg = Config::from_json(&root)?;
        // `generation_config.json` carries the inference-time eos, which is often a chat
        // turn-end token distinct from `config.json`'s base eos — MiniMax-M2's is `[e~[`
        // (200020) vs config.json's 2, and without it generation never halts at end of
        // turn. Merge its eos into the stop set (dedup, respecting the MAX_STOP_IDS cap).
        if let Ok(gtext) = std::fs::read_to_string(snap.join("generation_config.json")) {
            if let Some(gc) = Json::parse(&gtext) {
                let mut extra = Vec::new();
                parse_stop_ids(&gc, &mut extra);
                for id in extra {
                    if cfg.stop_ids.len() >= MAX_STOP_IDS {
                        break;
                    }
                    if !cfg.stop_ids.contains(&id) {
                        cfg.stop_ids.push(id);
                    }
                }
            }
        }
        Ok(cfg)
    }

    /// Build a `Config` from an already-parsed `config.json` root object,
    /// dispatching on architecture. MiniMax-M3 nests its hyperparameters under
    /// `text_config` and advertises `model_type == "minimax_m3_vl"`; everything
    /// else is treated as GLM-5.2.
    pub fn from_json(r: &Json) -> Result<Config, ConfigError> {
        let model_type = r.get("model_type").and_then(Json::as_str);
        let arch_is = |name: &str| {
            r.get("architectures")
                .and_then(Json::as_array)
                .map(|a| a.iter().any(|v| v.as_str() == Some(name)))
                .unwrap_or(false)
        };
        // Nemotron-H: hybrid Mamba2/GQA/latent-MoE, flat config at the root.
        if model_type == Some("nemotron_h") || arch_is("NemotronHForCausalLM") {
            return Config::from_json_nemotron(r);
        }
        // MiniMax-M2: flat config (hyperparameters at the root, no vision tower),
        // same GQA family as M3. Parse from the root object.
        if model_type == Some("minimax_m2") || arch_is("MiniMaxM2ForCausalLM") {
            return Config::from_json_minimax(r, r, Arch::MinimaxM2);
        }
        // Kimi-K3: hybrid KDA / gated-MLA, hyperparameters nested under `text_config`.
        //
        // MUST precede the M3 check below. That one treats *any* config carrying a
        // `text_config` as MiniMax-M3, and K3 carries one (it is a VL model too), so
        // reversing these two silently parses K3 as M3 — wrong attention family, wrong
        // expert geometry, and no KDA state, with nothing to signal it.
        // DeepSeek-V4-Flash. Like the K3 arm below, this MUST precede the M3 check: that
        // one claims any config carrying a `text_config`, and matching on `model_type`
        // first is the only thing keeping a new arch from being silently parsed as M3.
        // Keyed on `deepseek_v4` / `DeepseekV4ForCausalLM` — deliberately NOT on the
        // presence of `hc_mult` or `dspark_*`, so a V4 variant that drops one of those
        // still lands here rather than falling through to a wrong family.
        if model_type == Some("deepseek_v4") || arch_is("DeepseekV4ForCausalLM") {
            return Config::from_json_deepseek_v4(r);
        }
        if model_type == Some("kimi_k3")
            || arch_is("KimiK3ForConditionalGeneration")
            || arch_is("KimiLinearForCausalLM")
        {
            let t = r.get("text_config").unwrap_or(r);
            return Config::from_json_kimi(r, t);
        }
        // Maple. Like the arms above, this MUST precede the M3 fallthrough — that one
        // claims any config carrying a `text_config`. Maple carries none today, so it
        // would currently land in `from_json_glm` and be parsed as MLA: wrong attention
        // family entirely, and nothing would say so.
        if model_type == Some("maple") || arch_is("MapleForCausalLM") {
            return Config::from_json_maple(r);
        }
        // MiniMax-M3 VL: hyperparameters nested under `text_config`.
        let is_m3 = model_type == Some("minimax_m3_vl") || r.get("text_config").is_some();
        if is_m3 {
            let t = r.get("text_config").unwrap_or(r);
            Config::from_json_minimax(r, t, Arch::MinimaxM3)
        } else {
            Config::from_json_glm(r)
        }
    }

    /// GLM-5.2 (`glm_moe_dsa`) parse — the original path.
    fn from_json_glm(r: &Json) -> Result<Config, ConfigError> {
        let gi = |k: &str| gi_in(r, k);
        let mut c = Config {
            hidden: gi("hidden_size"),
            n_layers: gi("num_hidden_layers"),
            n_heads: gi("num_attention_heads"),
            n_experts: gi("n_routed_experts"),
            topk: gi("num_experts_per_tok"),
            moe_inter: gi("moe_intermediate_size"),
            dense_inter: gi("intermediate_size"),
            first_dense: gi("first_k_dense_replace"),
            q_lora: gi("q_lora_rank"),
            o_lora: 0,
            o_groups: 0,
            kv_lora: gi("kv_lora_rank"),
            qk_nope: gi("qk_nope_head_dim"),
            qk_rope: gi("qk_rope_head_dim"),
            qk_head: 0,
            v_head: gi("v_head_dim"),
            n_shared: gi("n_shared_experts"),
            vocab: gi("vocab_size"),
            max_ctx: gi("max_position_embeddings"),
            n_group: gi("n_group"),
            topk_group: gi("topk_group"),
            norm_topk: r
                .get("norm_topk_prob")
                .and_then(Json::as_bool)
                .unwrap_or(false),
            stop_ids: Vec::new(),
            index_topk: gi("index_topk"),
            index_nh: gi("index_n_heads"),
            index_hd: gi("index_head_dim"),
            index_block_size: 0,
            index_topk_blocks: 0,
            index_local_blocks: 0,
            idx_type: Vec::new(),
            eps: r.get("rms_norm_eps").and_then(Json::as_f64).unwrap_or(1e-5) as f32,
            theta: 10000.0,
            attn_scale: 0.0,
            routed_scale: r
                .get("routed_scaling_factor")
                .and_then(Json::as_f64)
                .unwrap_or(1.0) as f32,
            // GLM defaults for the MiniMax-only fields.
            arch: Arch::GlmMoeDsa,
            n_kv_heads: gi("num_attention_heads"),
            shared_inter: gi("n_shared_experts") * gi("moe_intermediate_size"),
            qk_norm: false,
            gemma_norm: false,
            swiglu_oai: false,
            swiglu_alpha: 0.0,
            swiglu_limit: 0.0,
            // No Hyper-Connections and no V4 sliding window on this arch: `hc_mult == 0`
            // is the "plain residual" case, and `window == 0` means "no ring buffer".
            compress_ratios: Vec::new(),
            compress_theta: 0.0,
            n_hash_layers: 0,
            dspark_block: 0,
            dspark_noise_id: 0,
            markov_rank: 0,
            dspark_targets: Vec::new(),
            hc_mult: 0,
            hc_sinkhorn_iters: 0,
            hc_eps: 0.0,
            window: 0,
            sigmoid_route: false,
            // Pre-Maple arches: sigmoid scoring, no sliding/full interleave, RoPE on
            // every layer. `RouterScore::Sigmoid` here (rather than deriving it from
            // `sigmoid_route`) is what keeps these bit-identical — see `RouterScore`.
            router_score: RouterScore::Sigmoid,
            swa_window: 0,
            layer_swa: Vec::new(),
            nope_on_global: false,
            // Nemotron-H-only fields (unused by GLM).
            layer_kind: Vec::new(),
            mtp_layer_kind: Vec::new(),
            mamba_d_state: 0,
            mamba_d_conv: 0,
            mamba_n_heads: 0,
            mamba_head_dim: 0,
            mamba_n_groups: 0,
            mamba_inter: 0,
            mamba_chunk: 0,
            moe_latent: 0,
            relu2: false,
            mamba_dt_min: 0.0,
            // Kimi-K3-only fields.
            situ: false,
            situ_beta: 0.0,
            situ_linear_beta: 0.0,
            mla_nope: false,
            kda_n_heads: 0,
            kda_head_dim: 0,
            kda_d_conv: 0,
            attn_res_block_size: 0,
        };

        // rope theta lives under rope_parameters.rope_theta
        if let Some(th) = r
            .get("rope_parameters")
            .and_then(|rp| rp.get("rope_theta"))
            .and_then(Json::as_f64)
        {
            c.theta = th as f32;
        }

        parse_stop_ids(r, &mut c.stop_ids);

        // Per-layer indexer type: explicit list, or a freq/offset formula.
        let n_layers_capped = (c.n_layers.max(0) as usize).min(MAX_LAYERS_IDX);
        c.idx_type = vec![false; n_layers_capped];
        {
            let it = r.get("indexer_types").and_then(Json::as_array);
            let mut freq = gi("index_topk_freq");
            if freq < 1 {
                freq = 1;
            }
            let off = r
                .get("index_skip_topk_offset")
                .and_then(Json::as_i64)
                .unwrap_or(2) as i32;
            for (i, slot) in c.idx_type.iter_mut().enumerate() {
                let ii = i as i32;
                if let Some(arr) = it {
                    if let Some(s) = arr.get(i).and_then(Json::as_str) {
                        *slot = s == "full";
                        continue;
                    }
                }
                let v = (ii - off + 1).max(0);
                *slot = v % freq == 0;
            }
        }

        c.qk_head = c.qk_nope + c.qk_rope;
        c.attn_scale = 1.0 / (c.qk_head as f32).sqrt();

        if c.n_group != 1 {
            return Err(ConfigError::Unsupported(
                "this engine requires n_group=1 (GLM-5.2)".into(),
            ));
        }

        c.validate_common()?;
        // GLM/MLA-specific ranges.
        ckr!("q_lora_rank", c.q_lora, 0, 1 << 20);
        ckr!("kv_lora_rank", c.kv_lora, 1, 1 << 20);
        ckr!("qk_nope_head_dim", c.qk_nope, 1, 1 << 16);
        ckr!("qk_rope_head_dim", c.qk_rope, 1, 1 << 16);
        ckr!("index_topk", c.index_topk, 0, 1 << 20);
        ckr!("index_n_heads", c.index_nh, 0, 1024);
        ckr!("index_head_dim", c.index_hd, 0, 1 << 16);
        Ok(c)
    }

    /// MiniMax GQA parse, shared by M3 and M2. `t` holds the hyperparameters —
    /// `text_config` for the M3 VL config, the root for the flat M2 config — and `r`
    /// is the root (stop-id fallback). `arch` selects the variant. The GQA head
    /// geometry is folded onto `qk_nope`/`qk_rope`/`v_head`: `qk_rope` is the rotary
    /// sub-dimension (`rotary_dim`), `qk_nope = head_dim − rotary_dim`.
    fn from_json_minimax(r: &Json, t: &Json, arch: Arch) -> Result<Config, ConfigError> {
        let gt = |k: &str| gi_in(t, k);

        let head_dim = gt("head_dim");
        let rotary_dim = gt("rotary_dim");
        // Partial RoPE: rotate the first `rotary_dim` of each head, leave the rest.
        let qk_rope = rotary_dim;
        let qk_nope = head_dim - rotary_dim;

        // first-dense count = leading zeros of `moe_layer_freq` (dense layers precede
        // the MoE stack); fall back to `first_k_dense_replace` if the list is absent.
        let first_dense = match t.get("moe_layer_freq").and_then(Json::as_array) {
            Some(arr) => arr.iter().take_while(|v| v.as_i64() == Some(0)).count() as i32,
            None => gt("first_k_dense_replace"),
        };

        let act = t.get("hidden_act").and_then(Json::as_str).unwrap_or("");
        let scoring = t.get("scoring_func").and_then(Json::as_str).unwrap_or("");

        // Block-sparse attention (Lightning Indexer). `sparse_attention_config` carries the
        // indexer geometry and the per-layer on/off list (`sparse_attention_freq`; the
        // leading dense layers are 0). Absent config → dense everywhere.
        let sac = t.get("sparse_attention_config");
        let sg = |k: &str, d: i32| {
            sac.and_then(|s| s.get(k))
                .and_then(Json::as_i64)
                .map(|v| v as i32)
                .unwrap_or(d)
        };
        let index_nh = sg("sparse_num_index_heads", 0);
        let index_hd = sg("sparse_index_dim", 0);
        let index_block_size = sg("sparse_block_size", 0);
        let index_topk_blocks = sg("sparse_topk_blocks", 0);
        let index_local_blocks = sg("sparse_local_block", 0);
        let nlc = (gt("num_hidden_layers").max(0) as usize).min(MAX_LAYERS_IDX);
        let idx_type: Vec<bool> = match sac
            .and_then(|s| s.get("sparse_attention_freq"))
            .and_then(Json::as_array)
        {
            Some(arr) => (0..nlc)
                .map(|i| arr.get(i).and_then(Json::as_i64).unwrap_or(0) != 0)
                .collect(),
            None => vec![false; nlc],
        };

        let mut c = Config {
            hidden: gt("hidden_size"),
            n_layers: gt("num_hidden_layers"),
            n_heads: gt("num_attention_heads"),
            n_experts: gt("num_local_experts"),
            topk: gt("num_experts_per_tok"),
            moe_inter: gt("intermediate_size"), // expert FFN width
            dense_inter: gt("dense_intermediate_size"),
            first_dense,
            q_lora: 0,  // GQA: no query LoRA
            o_lora: 0,  // GQA: plain o_proj, no output LoRA
            o_groups: 0,
            kv_lora: 0, // GQA: no latent KV
            qk_nope,
            qk_rope,
            qk_head: head_dim,
            v_head: head_dim,
            n_shared: gt("n_shared_experts"),
            vocab: gt("vocab_size"),
            max_ctx: gt("max_position_embeddings"),
            n_group: 1,
            topk_group: 1,
            // MiniMax normalizes the top-k gate weights before `routed_scaling`.
            norm_topk: true,
            stop_ids: Vec::new(),
            // Block-sparse Lightning Indexer (see the parse above); `index_topk` is the
            // effective per-query token budget (topk_blocks * block_size), 0 if dense.
            index_topk: index_topk_blocks * index_block_size,
            index_nh,
            index_hd,
            index_block_size,
            index_topk_blocks,
            index_local_blocks,
            idx_type,
            eps: t.get("rms_norm_eps").and_then(Json::as_f64).unwrap_or(1e-6) as f32,
            theta: t
                .get("rope_theta")
                .and_then(Json::as_f64)
                .unwrap_or(10000.0) as f32,
            attn_scale: if head_dim > 0 {
                1.0 / (head_dim as f32).sqrt()
            } else {
                0.0
            },
            routed_scale: t
                .get("routed_scaling_factor")
                .and_then(Json::as_f64)
                .unwrap_or(1.0) as f32,
            arch,
            n_kv_heads: gt("num_key_value_heads"),
            shared_inter: gt("shared_intermediate_size"),
            qk_norm: t
                .get("use_qk_norm")
                .and_then(Json::as_bool)
                .unwrap_or(false),
            gemma_norm: t
                .get("use_gemma_norm")
                .and_then(Json::as_bool)
                .unwrap_or(false),
            swiglu_oai: act == "swigluoai",
            swiglu_alpha: t
                .get("swiglu_alpha")
                .and_then(Json::as_f64)
                .unwrap_or(1.702) as f32,
            swiglu_limit: t.get("swiglu_limit").and_then(Json::as_f64).unwrap_or(7.0) as f32,
            // No Hyper-Connections and no V4 sliding window on this arch: `hc_mult == 0`
            // is the "plain residual" case, and `window == 0` means "no ring buffer".
            compress_ratios: Vec::new(),
            compress_theta: 0.0,
            n_hash_layers: 0,
            dspark_block: 0,
            dspark_noise_id: 0,
            markov_rank: 0,
            dspark_targets: Vec::new(),
            hc_mult: 0,
            hc_sinkhorn_iters: 0,
            hc_eps: 0.0,
            window: 0,
            sigmoid_route: scoring == "sigmoid",
            // Pre-Maple arches: sigmoid scoring, no sliding/full interleave, RoPE on
            // every layer. `RouterScore::Sigmoid` here (rather than deriving it from
            // `sigmoid_route`) is what keeps these bit-identical — see `RouterScore`.
            router_score: RouterScore::Sigmoid,
            swa_window: 0,
            layer_swa: Vec::new(),
            nope_on_global: false,
            // Nemotron-H-only fields (unused by MiniMax).
            layer_kind: Vec::new(),
            mtp_layer_kind: Vec::new(),
            mamba_d_state: 0,
            mamba_d_conv: 0,
            mamba_n_heads: 0,
            mamba_head_dim: 0,
            mamba_n_groups: 0,
            mamba_inter: 0,
            mamba_chunk: 0,
            moe_latent: 0,
            relu2: false,
            mamba_dt_min: 0.0,
            // Kimi-K3-only fields.
            situ: false,
            situ_beta: 0.0,
            situ_linear_beta: 0.0,
            mla_nope: false,
            kda_n_heads: 0,
            kda_head_dim: 0,
            kda_d_conv: 0,
            attn_res_block_size: 0,
        };

        // eos/stop ids may sit in text_config or at the root.
        parse_stop_ids(t, &mut c.stop_ids);
        if c.stop_ids.is_empty() {
            parse_stop_ids(r, &mut c.stop_ids);
        }

        c.validate_common()?;
        // GQA-specific ranges.
        ckr!("head_dim", head_dim, 1, 1 << 16);
        ckr!("rotary_dim", rotary_dim, 1, head_dim);
        ckr!("num_key_value_heads", c.n_kv_heads, 1, c.n_heads);
        ckr!("shared_intermediate_size", c.shared_inter, 0, 1 << 24);
        Ok(c)
    }

    /// Maple (`maple`) parse — a flat config, everything at the root.
    ///
    /// Shares the GQA family's geometry encoding with MiniMax: `qk_rope` is the rotary
    /// sub-dimension and `qk_nope` the remainder, so the attention path needs no new
    /// branch. Maple states the split as a **fraction** (`partial_rotary_factor` 0.5)
    /// where MiniMax states it as an absolute `rotary_dim`, which is the only geometry
    /// difference.
    ///
    /// What is genuinely Maple's, and what a reader should check against
    /// `modeling_maple.py` rather than against another arch here:
    ///
    /// - **`num_experts`**, not MiniMax's `num_local_experts` and not GLM's
    ///   `n_routed_experts`. All three name the same quantity.
    /// - **`moe_intermediate_size` (512) is the expert width**, and `intermediate_size`
    ///   (4096) describes a dense FFN this checkpoint does not contain — every layer is
    ///   MoE (`first_dense == 0`) and there is no shared expert. Reading the wrong one
    ///   sizes every expert 8x too wide.
    /// - **The router is softmax**, with no `e_score_correction_bias`. See
    ///   [`RouterScore`]; `scoring_func` is absent from this config entirely.
    /// - **`layer_types`** gives the sliding/full interleave, and
    ///   **`nope_on_global_attention`** makes the full layers positionless.
    /// - **The activation is the clamped SwiGLU**, expressed as a bare
    ///   `hidden_act: "silu"` plus hard-coded clamps in `MapleMLP.forward`
    ///   (`gate` clamped above at 7.0, `up` clamped to [-7, 7]) — NOT the `swigluoai`
    ///   spelling, and with no `swiglu_alpha` sigmoid gate. That is exactly V4's variant,
    ///   so it sets `swiglu_oai: false` with `swiglu_limit: 7.0`, matching
    ///   `from_json_deepseek_v4`. Taking `hidden_act` at face value and running a plain
    ///   SiLU silently drops both clamps.
    fn from_json_maple(r: &Json) -> Result<Config, ConfigError> {
        let gi = |k: &str| gi_in(r, k);

        let n_heads = gi("num_attention_heads");
        let head_dim = {
            let hd = gi("head_dim");
            if hd > 0 {
                hd
            } else if n_heads > 0 {
                gi("hidden_size") / n_heads
            } else {
                0
            }
        };
        // Partial RoPE as a fraction of head_dim. The reference derives the rotary width
        // from `cos.shape[-1]`, which is `2 * len(inv_freq)` and works out to
        // `head_dim * partial_rotary_factor` — 64 here.
        let frac = r
            .get("partial_rotary_factor")
            .and_then(Json::as_f64)
            .unwrap_or(1.0);
        let rotary_dim = ((head_dim as f64) * frac) as i32;

        let n_layers = gi("num_hidden_layers");
        let nlc = (n_layers.max(0) as usize).min(MAX_LAYERS_IDX);
        // `layer_types` is authoritative. If it is absent, every layer is FULL attention:
        // that is the conservative default (attend to everything) — the opposite guess
        // would silently truncate context on a model that never asked for a window.
        let layer_swa: Vec<bool> = match r.get("layer_types").and_then(Json::as_array) {
            Some(arr) => (0..nlc)
                .map(|i| arr.get(i).and_then(Json::as_str) == Some("sliding_attention"))
                .collect(),
            None => vec![false; nlc],
        };
        // A window only means anything if some layer actually uses one.
        let swa_window = if layer_swa.iter().any(|b| *b) {
            gi("sliding_window")
        } else {
            0
        };

        let mut c = Config {
            hidden: gi("hidden_size"),
            n_layers,
            n_heads,
            n_experts: gi("num_experts"),
            topk: gi("num_experts_per_tok"),
            moe_inter: gi("moe_intermediate_size"),
            // No dense FFN anywhere in this checkpoint; `intermediate_size` describes one
            // that does not exist. Left at 0 so a dense-FFN read fails loudly.
            dense_inter: 0,
            first_dense: 0,
            q_lora: 0,
            o_lora: 0,
            o_groups: 0,
            kv_lora: 0,
            qk_nope: head_dim - rotary_dim,
            qk_rope: rotary_dim,
            qk_head: head_dim,
            v_head: head_dim,
            n_shared: r
                .get("num_shared_experts")
                .and_then(Json::as_i64)
                .unwrap_or(0) as i32,
            vocab: gi("vocab_size"),
            max_ctx: gi("max_position_embeddings"),
            n_group: 1,
            topk_group: 1,
            // `MapleGate.forward` divides the chosen scores by their sum unconditionally;
            // `norm_topk_prob: true` in the config agrees.
            norm_topk: true,
            stop_ids: Vec::new(),
            index_topk: 0,
            index_nh: 0,
            index_hd: 0,
            index_block_size: 0,
            index_topk_blocks: 0,
            index_local_blocks: 0,
            idx_type: Vec::new(),
            eps: r.get("rms_norm_eps").and_then(Json::as_f64).unwrap_or(1e-6) as f32,
            theta: r
                .get("rope_theta")
                .and_then(Json::as_f64)
                .unwrap_or(10000.0) as f32,
            attn_scale: if head_dim > 0 {
                1.0 / (head_dim as f32).sqrt()
            } else {
                0.0
            },
            // No `routed_scaling_factor` in this config, and none in the reference —
            // the renormalised softmax weights are used as-is.
            routed_scale: 1.0,
            arch: Arch::Maple,
            n_kv_heads: gi("num_key_value_heads"),
            shared_inter: 0,
            qk_norm: r.get("use_qk_norm").and_then(Json::as_bool).unwrap_or(false),
            gemma_norm: false,
            // Clamped SwiGLU without the OAI sigmoid gate — V4's variant. See the note
            // on this function.
            swiglu_oai: false,
            swiglu_alpha: 0.0,
            swiglu_limit: 7.0,
            compress_ratios: Vec::new(),
            compress_theta: 0.0,
            n_hash_layers: 0,
            dspark_block: 0,
            dspark_noise_id: 0,
            markov_rank: 0,
            dspark_targets: Vec::new(),
            hc_mult: 0,
            hc_sinkhorn_iters: 0,
            hc_eps: 0.0,
            // NOT the sliding window — `window` is V4's raw-KV ring size. See `swa_window`.
            window: 0,
            // `scoring_func` is absent from this config; recorded as false to match, and
            // unread either way.
            sigmoid_route: false,
            router_score: RouterScore::Softmax,
            swa_window,
            layer_swa,
            nope_on_global: r
                .get("nope_on_global_attention")
                .and_then(Json::as_bool)
                .unwrap_or(false),
            layer_kind: Vec::new(),
            mtp_layer_kind: Vec::new(),
            mamba_d_state: 0,
            mamba_d_conv: 0,
            mamba_n_heads: 0,
            mamba_head_dim: 0,
            mamba_n_groups: 0,
            mamba_inter: 0,
            mamba_chunk: 0,
            moe_latent: 0,
            relu2: false,
            mamba_dt_min: 0.0,
            situ: false,
            situ_beta: 0.0,
            situ_linear_beta: 0.0,
            mla_nope: false,
            kda_n_heads: 0,
            kda_head_dim: 0,
            kda_d_conv: 0,
            attn_res_block_size: 0,
        };

        parse_stop_ids(r, &mut c.stop_ids);

        c.validate_common()?;
        ckr!("head_dim", head_dim, 1, 1 << 16);
        ckr!("rotary_dim", rotary_dim, 1, head_dim);
        ckr!("num_key_value_heads", c.n_kv_heads, 1, c.n_heads);
        // 0 is legal (no sliding layers); a window that exists must be a real span.
        ckr!("sliding_window", c.swa_window, 0, 1 << 24);
        Ok(c)
    }

    /// The MoE layers as `(count, index_of_one)`. The index is somewhere to probe a
    /// real expert on disk, to size one from its true on-disk format.
    ///
    /// Two encodings, because [`LayerKind`] has two producers:
    /// - Nemotron-H marks MoE layers explicitly ([`LayerKind::Moe`]) — there a layer is
    ///   *either* a mixer or an FFN, so the MoE layers are named on that axis.
    /// - GLM, MiniMax and Kimi-K3 put an FFN on *every* layer, so the MoE layers are the
    ///   suffix after the `first_dense` dense prefix.
    ///
    /// K3 reaches the second branch with a **non-empty** `layer_kind` (it carries Kda/Attn
    /// mixer kinds), which is why the test below is "does `layer_kind` name any MoE layer"
    /// rather than "is `layer_kind` empty". The latter reads 0 MoE layers for K3 and
    /// mis-sizes the expert cache by the entire model.
    pub fn moe_layers(&self) -> (usize, usize) {
        if let Some(idx) = self.layer_kind.iter().position(|k| *k == LayerKind::Moe) {
            (
                self.layer_kind
                    .iter()
                    .filter(|k| **k == LayerKind::Moe)
                    .count(),
                idx,
            )
        } else {
            (
                (self.n_layers - self.first_dense).max(0) as usize,
                self.first_dense.max(0) as usize,
            )
        }
    }

    /// Does layer `i` attend only to the last [`Config::swa_window`] keys?
    ///
    /// `false` for every arch that leaves `layer_swa` empty, which is every arch before
    /// Maple — so an out-of-range index reads as full attention, the conservative answer
    /// (attend to everything) rather than the silently-truncating one.
    pub fn layer_is_swa(&self, i: usize) -> bool {
        self.swa_window > 0 && self.layer_swa.get(i).copied().unwrap_or(false)
    }

    /// Does layer `i` apply RoPE?
    ///
    /// Only Maple answers `false` anywhere, and only on its full-attention layers: the
    /// reference gates `apply_rotary_pos_emb` on `self.sliding_window is not None`, so
    /// "global" and "no positional encoding" are the same condition there. This is a
    /// named predicate rather than `!nope_on_global || layer_is_swa(i)` at three call
    /// sites, because the two that agree and the one that doesn't produce a model that
    /// still generates fluent text — just positionally wrong.
    pub fn layer_uses_rope(&self, i: usize) -> bool {
        !self.nope_on_global || self.layer_is_swa(i)
    }

    /// The number of KV entries layer `i` can ever hold, given a context of `ctx` tokens.
    /// Sliding layers are bounded by the window however long the context runs.
    pub fn layer_kv_span(&self, i: usize, ctx: usize) -> usize {
        if self.layer_is_swa(i) {
            ctx.min(self.swa_window.max(0) as usize)
        } else {
            ctx
        }
    }

    /// Kimi-K3 (`kimi_k3`) parse. The hyperparameters are nested under `text_config`
    /// (the root describes the vision-language wrapper); `r` is that root, for the
    /// stop-id fallback.
    ///
    /// Two things differ from every other arch:
    ///
    /// - **The mixer is per-layer.** `linear_attn_config` names which layers run gated
    ///   MLA (`full_attn_layers`) and which run KDA (`kda_layers`). Those lists are
    ///   **1-indexed**: `full_attn_layers` ends at 93 on a 93-layer stack, and checkpoint
    ///   layer 0 is KDA (config index 1). Converted to 0-indexed [`Config::layer_kind`]
    ///   here, so everything downstream indexes normally.
    /// - **MoE is not on that axis.** Every layer after `first_k_dense_replace` carries
    ///   experts regardless of its mixer, so `layer_kind` holds no `Moe` entries and
    ///   [`Config::moe_layers`] falls through to the prefix rule.
    /// DeepSeek-V4-Flash (`deepseek_v4`).
    ///
    /// Everything here is read straight from `config.json`. Two geometry fields are
    /// deliberately left at 0 with no fallback guess — see the note on `qk_nope`/`v_head`
    /// below. A wrong attention geometry does not fail loudly, it produces plausible
    /// garbage, so these stay 0 until they are derived from the checkpoint's own
    /// `inference/model.py` and pinned by a test against real tensor shapes.
    fn from_json_deepseek_v4(r: &Json) -> Result<Config, ConfigError> {
        let g = |k: &str| gi_in(r, k);
        let nlc = (g("num_hidden_layers").max(0) as usize).min(MAX_LAYERS_IDX);

        let mut c = Config {
            hidden: g("hidden_size"),
            n_layers: g("num_hidden_layers"),
            n_heads: g("num_attention_heads"),
            n_experts: g("n_routed_experts"),
            topk: g("num_experts_per_tok"),
            moe_inter: g("moe_intermediate_size"),
            // No `intermediate_size` and no `first_k_dense_replace`: EVERY layer is MoE.
            // Confirmed against the checkpoint inventory, which carries 43 x 256 expert
            // tensor groups (11008) with no dense-FFN layer among them.
            dense_inter: 0,
            first_dense: 0,
            q_lora: g("q_lora_rank"),
            o_lora: g("o_lora_rank"),
            o_groups: g("o_groups").max(1),
            // Attention geometry, taken from the checkpoint's own `inference/model.py`
            // (class `Attention`) rather than inferred, and cross-checked against the
            // released tensor shapes:
            //   nope_head_dim = head_dim - rope_head_dim      => 512 - 64 = 448
            //   wkv  = Linear(dim, head_dim)                  => [512, 4096],  scale [4,32]
            //   kv_norm = RMSNorm(head_dim)                   => [512]
            //   wq_b = Linear(q_lora_rank, n_heads*head_dim)  => [32768, 1024], scale [256,8]
            //
            // There is NO separate V projection: `wkv` emits one `head_dim`-wide latent
            // that serves as both K and V (`kv_cache` is `(batch, t, head_dim)`), which is
            // what `num_key_value_heads: 1` is describing. So the KV latent, the qk head
            // width and the v head width are all `head_dim`, and KV costs 512 f32 per token
            // per layer — cheap for a 43-layer model, which is how it affords 1M context.
            kv_lora: g("head_dim"),
            qk_nope: g("head_dim") - g("qk_rope_head_dim"),
            qk_head: g("head_dim"),
            v_head: g("head_dim"),
            qk_rope: g("qk_rope_head_dim"),
            n_shared: g("n_shared_experts"),
            vocab: g("vocab_size"),
            max_ctx: g("max_position_embeddings"),
            n_group: g("n_group").max(1),
            topk_group: g("topk_group").max(1),
            norm_topk: r
                .get("norm_topk_prob")
                .and_then(Json::as_bool)
                .unwrap_or(true),
            stop_ids: Vec::new(),
            // V4 DOES carry a DSA-family indexer, but on only 21 of 43 layers (the
            // checkpoint has 21 `attn.indexer.*` groups). These dims describe the indexer
            // where it exists; which layers have one is a per-layer fact the loader must
            // read from the weights, not something this scalar config can express.
            index_topk: g("index_topk"),
            index_nh: g("index_n_heads"),
            index_hd: g("index_head_dim"),
            index_block_size: 0,
            index_topk_blocks: 0,
            index_local_blocks: 0,
            idx_type: Vec::new(),
            eps: r.get("rms_norm_eps").and_then(Json::as_f64).unwrap_or(1e-6) as f32,
            theta: r
                .get("rope_theta")
                .and_then(Json::as_f64)
                .unwrap_or(10000.0) as f32,
            attn_scale: 0.0,
            routed_scale: r
                .get("routed_scaling_factor")
                .and_then(Json::as_f64)
                .unwrap_or(1.0) as f32,
            arch: Arch::DeepseekV4,
            n_kv_heads: g("num_key_value_heads"),
            shared_inter: g("n_shared_experts") * g("moe_intermediate_size"),
            qk_norm: false,
            gemma_norm: false,
            // `swiglu_limit` is a clamp on the SwiGLU product, as on M3 — but V4 states it
            // as a bare float with no `swiglu_alpha`, so only the limit is set.
            swiglu_oai: false,
            swiglu_alpha: 0.0,
            swiglu_limit: r
                .get("swiglu_limit")
                .and_then(Json::as_f64)
                .unwrap_or(0.0) as f32,
            // Hyper-Connections: the residual stream becomes `[s, hc_mult, hidden]`.
            // Defaulted to 0 rather than 4 — a V4 variant that drops HC must run as a
            // plain residual, not silently get four copies it has no weights for.
            compress_ratios: r
                .get("compress_ratios")
                .and_then(Json::as_array)
                .map(|a| a.iter().filter_map(Json::as_f64).map(|v| v as i32).collect())
                .unwrap_or_default(),
            compress_theta: r
                .get("compress_rope_theta")
                .and_then(Json::as_f64)
                .unwrap_or(10000.0) as f32,
            n_hash_layers: r
                .get("num_hash_layers")
                .and_then(Json::as_f64)
                .unwrap_or(0.0) as i32,
            dspark_block: r.get("dspark_block_size").and_then(Json::as_f64).unwrap_or(0.0) as i32,
            dspark_noise_id: r
                .get("dspark_noise_token_id")
                .and_then(Json::as_f64)
                .unwrap_or(0.0) as i32,
            markov_rank: r
                .get("dspark_markov_rank")
                .and_then(Json::as_f64)
                .unwrap_or(0.0) as i32,
            dspark_targets: r
                .get("dspark_target_layer_ids")
                .and_then(Json::as_array)
                .map(|a| a.iter().filter_map(Json::as_f64).map(|v| v as i32).collect())
                .unwrap_or_default(),
            hc_mult: g("hc_mult"),
            hc_sinkhorn_iters: g("hc_sinkhorn_iters"),
            // Distinct from `eps` (the RMS epsilon): this one floors the Sinkhorn/sigmoid
            // mixing weights. The reference uses `norm_eps` for the rsqrt and `hc_eps`
            // only for the weights.
            hc_eps: r.get("hc_eps").and_then(Json::as_f64).unwrap_or(1e-6) as f32,
            // The raw KV cache is a ring buffer of this many rows; everything older lives
            // in the Compressor. Until that lands, context is exact to `window` and wrong
            // past it — a hard edge, not a gentle degradation.
            window: g("sliding_window"),
            // Routing is `sqrtsoftplus` + `noaux_tc`, which is neither the sigmoid nor the
            // softmax arm this bool selects between. Left false so it cannot silently take
            // the sigmoid path; the real scorer is a V4-specific one still to be written.
            sigmoid_route: false,
            // Pre-Maple arches: sigmoid scoring, no sliding/full interleave, RoPE on
            // every layer. `RouterScore::Sigmoid` here (rather than deriving it from
            // `sigmoid_route`) is what keeps these bit-identical — see `RouterScore`.
            router_score: RouterScore::Sigmoid,
            swa_window: 0,
            layer_swa: Vec::new(),
            nope_on_global: false,
            // Homogeneous on the mixer axis: every layer is attention + MoE. The
            // heterogeneity in V4 is the Compressor (41/43) and Indexer (21/43), which are
            // sub-modules of attention rather than a different mixer, so `layer_kind`
            // stays empty exactly as it does for GLM.
            layer_kind: Vec::new(),
            mtp_layer_kind: Vec::new(),
            mamba_d_state: 0,
            mamba_d_conv: 0,
            mamba_n_heads: 0,
            mamba_head_dim: 0,
            mamba_n_groups: 0,
            mamba_inter: 0,
            mamba_chunk: 0,
            // Experts are at model `hidden`, not in a latent bottleneck: w1 is
            // [moe_inter, hidden] = [2048, 4096] in the checkpoint. Contrast Nemotron-H
            // and K3, which both route through `moe_latent`.
            moe_latent: 0,
            mamba_dt_min: 0.0,
            situ: false,
            situ_beta: 0.0,
            situ_linear_beta: 0.0,
            mla_nope: false,
            kda_n_heads: 0,
            kda_head_dim: 0,
            kda_d_conv: 0,
            attn_res_block_size: 0,
            // Gated SwiGLU with a clamp, not Nemotron's gateless relu^2.
            relu2: false,
        };
        let _ = nlc;
        parse_stop_ids(r, &mut c.stop_ids);
        Ok(c)
    }

    fn from_json_kimi(r: &Json, t: &Json) -> Result<Config, ConfigError> {
        let gt = |k: &str| gi_in(t, k);
        let lac = t.get("linear_attn_config");
        let lg = |k: &str, d: i32| {
            lac.and_then(|s| s.get(k))
                .and_then(Json::as_i64)
                .map(|v| v as i32)
                .unwrap_or(d)
        };
        let nlc = (gt("num_hidden_layers").max(0) as usize).min(MAX_LAYERS_IDX);

        // Per-layer mixer from the 1-indexed `full_attn_layers` list; anything not named
        // there runs KDA. Reading `full_attn_layers` rather than `kda_layers` makes the
        // *default* the safe one: an unlisted layer becomes KDA, which carries a
        // fixed-size recurrent state that `fixed_bytes` charges for — rather than
        // silently becoming a KV-holding layer nothing reserved for.
        let full: Vec<i64> = lac
            .and_then(|s| s.get("full_attn_layers"))
            .and_then(Json::as_array)
            .map(|a| a.iter().filter_map(Json::as_i64).collect())
            .unwrap_or_default();
        let layer_kind: Vec<LayerKind> = (0..nlc)
            .map(|i| {
                if full.contains(&(i as i64 + 1)) {
                    LayerKind::Attn
                } else {
                    LayerKind::Kda
                }
            })
            .collect();

        let mut c = Config {
            hidden: gt("hidden_size"),
            n_layers: gt("num_hidden_layers"),
            n_heads: gt("num_attention_heads"),
            n_experts: gt("num_experts"),
            topk: gt("num_experts_per_token"),
            moe_inter: gt("moe_intermediate_size"),
            dense_inter: gt("intermediate_size"),
            first_dense: gt("first_k_dense_replace"),
            q_lora: gt("q_lora_rank"),
            o_lora: 0,
            o_groups: 0,
            kv_lora: gt("kv_lora_rank"),
            qk_nope: gt("qk_nope_head_dim"),
            qk_rope: gt("qk_rope_head_dim"),
            // Fixed up below to `qk_nope + qk_rope`, exactly as the GLM parse does.
            qk_head: 0,
            v_head: gt("v_head_dim"),
            n_shared: gt("num_shared_experts"),
            vocab: gt("vocab_size"),
            max_ctx: gt("max_position_embeddings"),
            n_group: gt("num_expert_group").max(1),
            topk_group: gt("topk_group").max(1),
            norm_topk: t
                .get("moe_renormalize")
                .and_then(Json::as_bool)
                .unwrap_or(false),
            stop_ids: Vec::new(),
            // No DSA lightning indexer on K3 — sparsity is the KDA mixer, not a
            // block-sparse index over a dense cache.
            index_topk: 0,
            index_nh: 0,
            index_hd: 0,
            index_block_size: 0,
            index_topk_blocks: 0,
            index_local_blocks: 0,
            idx_type: Vec::new(),
            eps: t.get("rms_norm_eps").and_then(Json::as_f64).unwrap_or(1e-5) as f32,
            theta: t
                .get("rope_theta")
                .and_then(Json::as_f64)
                .unwrap_or(10000.0) as f32,
            attn_scale: 0.0,
            routed_scale: t
                .get("routed_scaling_factor")
                .and_then(Json::as_f64)
                .unwrap_or(1.0) as f32,
            arch: Arch::KimiK3,
            n_kv_heads: gt("num_key_value_heads"),
            // The two shared experts ship fused as one `moe_intermediate_size`-wide pair
            // per layer (the checkpoint carries `[2 * moe_inter, hidden]`).
            shared_inter: gt("num_shared_experts") * gt("moe_intermediate_size"),
            qk_norm: false,
            gemma_norm: false,
            swiglu_oai: false,
            swiglu_alpha: 0.0,
            swiglu_limit: 0.0,
            // No Hyper-Connections and no V4 sliding window on this arch: `hc_mult == 0`
            // is the "plain residual" case, and `window == 0` means "no ring buffer".
            compress_ratios: Vec::new(),
            compress_theta: 0.0,
            n_hash_layers: 0,
            dspark_block: 0,
            dspark_noise_id: 0,
            markov_rank: 0,
            dspark_targets: Vec::new(),
            hc_mult: 0,
            hc_sinkhorn_iters: 0,
            hc_eps: 0.0,
            window: 0,
            sigmoid_route: t.get("moe_router_activation_func").and_then(Json::as_str)
                == Some("sigmoid"),
            // Pre-Maple arches: sigmoid scoring, no sliding/full interleave, RoPE on
            // every layer. `RouterScore::Sigmoid` here (rather than deriving it from
            // `sigmoid_route`) is what keeps these bit-identical — see `RouterScore`.
            router_score: RouterScore::Sigmoid,
            swa_window: 0,
            layer_swa: Vec::new(),
            nope_on_global: false,
            layer_kind,
            mtp_layer_kind: Vec::new(),
            mamba_d_state: 0,
            mamba_d_conv: 0,
            mamba_n_heads: 0,
            mamba_head_dim: 0,
            mamba_n_groups: 0,
            mamba_inter: 0,
            mamba_chunk: 0,
            // Stable LatentMoE: the routed experts run in a `routed_expert_hidden_size`
            // bottleneck, not in `hidden` — same shape as Nemotron-H's `moe_latent`.
            moe_latent: gt("routed_expert_hidden_size"),
            relu2: false,
            mamba_dt_min: 0.0,
            situ: t.get("hidden_act").and_then(Json::as_str) == Some("situ"),
            situ_beta: t
                .get("activation_situ_beta")
                .and_then(Json::as_f64)
                .unwrap_or(1.0) as f32,
            situ_linear_beta: t
                .get("activation_situ_linear_beta")
                .and_then(Json::as_f64)
                .unwrap_or(0.0) as f32,
            mla_nope: t
                .get("mla_use_nope")
                .and_then(Json::as_bool)
                .unwrap_or(false),
            kda_n_heads: lg("num_heads", 0),
            kda_head_dim: lg("head_dim", 0),
            kda_d_conv: lg("short_conv_kernel_size", 0),
            attn_res_block_size: gt("attn_res_block_size"),
        };

        parse_stop_ids(t, &mut c.stop_ids);
        if c.stop_ids.is_empty() {
            parse_stop_ids(r, &mut c.stop_ids);
        }

        // The MLA per-head query width, and its scale — same fixup the GLM parse does,
        // and the same value the reference computes (`q_head_dim ** -0.5`, 192^-0.5).
        //
        // This does NOT reintroduce a GQA full-KV charge: `KvCache::bytes_per_token`
        // gates that term on `allocates_gqa_kv`, which is false for K3, so `qk_head`
        // only ever feeds the attention geometry. Leaving it 0 (as an earlier version
        // did, reasoning "MLA has no GQA head dim") makes the MLA path compute with a
        // per-head width of zero.
        c.qk_head = c.qk_nope + c.qk_rope;
        c.attn_scale = 1.0 / (c.qk_head as f32).sqrt();

        c.validate_common()?;
        if c.layer_kind.len() != c.n_layers as usize {
            return Err(ConfigError::Unsupported(format!(
                "kimi_k3: layer_kind length {} != num_hidden_layers {}",
                c.layer_kind.len(),
                c.n_layers,
            )));
        }
        // Guard the case that would make every KV reservation silently zero.
        if !c.layer_kind.contains(&LayerKind::Attn) {
            return Err(ConfigError::Unsupported(
                "kimi_k3: linear_attn_config.full_attn_layers named no layer of this stack \
                 (the list is 1-indexed); every layer would be KDA and the KV cache would \
                 size to zero"
                    .to_string(),
            ));
        }
        ckr!("linear_attn_config.num_heads", c.kda_n_heads, 1, 1 << 16);
        ckr!("linear_attn_config.head_dim", c.kda_head_dim, 1, 1 << 16);
        ckr!(
            "linear_attn_config.short_conv_kernel_size",
            c.kda_d_conv,
            1,
            1 << 8
        );
        ckr!("routed_expert_hidden_size", c.moe_latent, 0, 1 << 24);
        ckr!("kv_lora_rank", c.kv_lora, 1, 1 << 16);
        // The driver takes `layer_idx % attn_res_block_size`, so 0 would divide by zero,
        // and the whole attention-residual mechanism is mandatory on K3 (there is no
        // ordinary residual stream to fall back to). Reject it here rather than in the
        // forward pass.
        ckr!("attn_res_block_size", c.attn_res_block_size, 1, 1 << 16);
        Ok(c)
    }

    /// Nemotron-H (`nemotron_h`) parse — a hybrid Mamba2 / GQA / latent-MoE model.
    /// The per-layer mixer sequence is read from `hybrid_override_pattern` into
    /// [`Config::layer_kind`] (`M`→Mamba, `E`→MoE, `*`→attention). The 8 attention
    /// layers are GQA with **full** RoPE (`partial_rotary_factor == 1.0`), so the head
    /// geometry folds onto `qk_rope = head_dim`, `qk_nope = 0` (mirroring MiniMax).
    /// The 40 MoE layers route in a low-rank `moe_latent` space with gateless ReLU²
    /// experts; the 40 Mamba2 layers carry recurrent conv+ssm state (no KV).
    fn from_json_nemotron(r: &Json) -> Result<Config, ConfigError> {
        let gi = |k: &str| gi_in(r, k);
        let gf = |k: &str, d: f64| r.get(k).and_then(Json::as_f64).unwrap_or(d);

        let head_dim = gi("head_dim");
        // partial_rotary_factor == 1.0 → full rope over the whole head.
        let rot = gf("partial_rotary_factor", 1.0);
        let qk_rope = ((head_dim as f64) * rot).round() as i32;
        let qk_nope = head_dim - qk_rope;

        // Per-layer mixer kinds from the hybrid pattern; length must == num_hidden_layers.
        let layer_kind = parse_hybrid_pattern(
            r.get("hybrid_override_pattern")
                .and_then(Json::as_str)
                .unwrap_or(""),
            "hybrid_override_pattern",
        )?;
        // The MTP speculative head's own sublayer sequence (`"*E"` on Nemotron-H-MTP:
        // an attention block then a latent-MoE block). Absent on checkpoints without a
        // head, in which case this stays empty and the engine simply never loads one.
        let mtp_layer_kind = parse_hybrid_pattern(
            r.get("mtp_hybrid_override_pattern")
                .and_then(Json::as_str)
                .unwrap_or(""),
            "mtp_hybrid_override_pattern",
        )?;

        let mamba_n_heads = gi("mamba_num_heads");
        let mamba_head_dim = gi("mamba_head_dim");
        let mamba_inter = mamba_n_heads * mamba_head_dim;

        let mut c = Config {
            hidden: gi("hidden_size"),
            n_layers: gi("num_hidden_layers"),
            n_heads: gi("num_attention_heads"),
            n_experts: gi("n_routed_experts"),
            topk: gi("num_experts_per_tok"),
            moe_inter: gi("moe_intermediate_size"),
            // No dense-MLP layers; keep validation happy with a nonzero width (unused).
            dense_inter: gi("moe_intermediate_size").max(1),
            first_dense: 0, // MoE layers are index-selected via layer_kind, not a prefix.
            q_lora: 0,
            o_lora: 0,
            o_groups: 0,
            kv_lora: 0,
            qk_nope,
            qk_rope,
            qk_head: head_dim,
            v_head: head_dim,
            n_shared: gi("n_shared_experts").max(0),
            vocab: gi("vocab_size"),
            max_ctx: gi("max_position_embeddings"),
            n_group: gi("n_group").max(1),
            topk_group: gi("topk_group").max(1),
            norm_topk: r
                .get("norm_topk_prob")
                .and_then(Json::as_bool)
                .unwrap_or(true),
            stop_ids: Vec::new(),
            index_topk: 0,
            index_nh: 0,
            index_hd: 0,
            index_block_size: 0,
            index_topk_blocks: 0,
            index_local_blocks: 0,
            idx_type: Vec::new(),
            // Nemotron-H uses `layer_norm_epsilon` (also mirrored as `norm_eps`), both 1e-5.
            eps: gf("layer_norm_epsilon", gf("norm_eps", 1e-5)) as f32,
            theta: gf("rope_theta", 10000.0) as f32,
            attn_scale: if head_dim > 0 {
                1.0 / (head_dim as f32).sqrt()
            } else {
                0.0
            },
            routed_scale: gf("routed_scaling_factor", 1.0) as f32,
            arch: Arch::NemotronH,
            n_kv_heads: gi("num_key_value_heads"),
            shared_inter: gi("moe_shared_expert_intermediate_size").max(0),
            qk_norm: false,
            gemma_norm: false,
            swiglu_oai: false,
            swiglu_alpha: 0.0,
            swiglu_limit: 0.0,
            // No Hyper-Connections and no V4 sliding window on this arch: `hc_mult == 0`
            // is the "plain residual" case, and `window == 0` means "no ring buffer".
            compress_ratios: Vec::new(),
            compress_theta: 0.0,
            n_hash_layers: 0,
            dspark_block: 0,
            dspark_noise_id: 0,
            markov_rank: 0,
            dspark_targets: Vec::new(),
            hc_mult: 0,
            hc_sinkhorn_iters: 0,
            hc_eps: 0.0,
            window: 0,
            // DeepSeek-style sigmoid router with an additive correction bias.
            sigmoid_route: true,
            // Pre-Maple arches: sigmoid scoring, no sliding/full interleave, RoPE on
            // every layer. `RouterScore::Sigmoid` here (rather than deriving it from
            // `sigmoid_route`) is what keeps these bit-identical — see `RouterScore`.
            router_score: RouterScore::Sigmoid,
            swa_window: 0,
            layer_swa: Vec::new(),
            nope_on_global: false,
            layer_kind,
            mtp_layer_kind,
            mamba_d_state: gi("ssm_state_size"),
            mamba_d_conv: gi("conv_kernel"),
            mamba_n_heads,
            mamba_head_dim,
            mamba_n_groups: gi("n_groups").max(1),
            mamba_inter,
            mamba_chunk: gi("chunk_size").max(1),
            moe_latent: gi("moe_latent_size"),
            relu2: r.get("mlp_hidden_act").and_then(Json::as_str) == Some("relu2"),
            // Scan clamps the discretized step to `time_step_min` (reference:
            // `torch.clamp(dt, self.time_step_min)`); default 0.0 (no floor) if absent.
            mamba_dt_min: gf("time_step_min", 0.0) as f32,
            // Kimi-K3-only fields.
            situ: false,
            situ_beta: 0.0,
            situ_linear_beta: 0.0,
            mla_nope: false,
            kda_n_heads: 0,
            kda_head_dim: 0,
            kda_d_conv: 0,
            attn_res_block_size: 0,
        };

        parse_stop_ids(r, &mut c.stop_ids);

        c.validate_common()?;
        // GQA + Mamba2-specific ranges.
        ckr!("head_dim", head_dim, 1, 1 << 16);
        ckr!("num_key_value_heads", c.n_kv_heads, 1, c.n_heads);
        ckr!("ssm_state_size", c.mamba_d_state, 1, 1 << 16);
        ckr!("conv_kernel", c.mamba_d_conv, 1, 64);
        ckr!("mamba_num_heads", c.mamba_n_heads, 1, 1 << 16);
        ckr!("mamba_head_dim", c.mamba_head_dim, 1, 1 << 16);
        ckr!("n_groups", c.mamba_n_groups, 1, c.mamba_n_heads);
        ckr!("chunk_size", c.mamba_chunk, 1, 1 << 16);
        ckr!("moe_latent_size", c.moe_latent, 1, 1 << 20);
        ckr!(
            "moe_shared_expert_intermediate_size",
            c.shared_inter,
            1,
            1 << 24
        );
        if c.layer_kind.len() != c.n_layers as usize {
            return Err(ConfigError::Unsupported(format!(
                "nemotron_h: hybrid_override_pattern length {} != num_hidden_layers {}",
                c.layer_kind.len(),
                c.n_layers
            )));
        }
        Ok(c)
    }

    /// How many extra layer indices (above `n_layers`) an MTP speculative head occupies.
    ///
    /// GLM/M3's head is a single sparse block living at index `n_layers`, so the answer
    /// is 1 and no pattern is involved. Nemotron-H's head is two sublayers
    /// (`mtp_hybrid_override_pattern == "*E"`: attention then latent-MoE), occupying
    /// `n_layers` and `n_layers + 1`.
    ///
    /// This answers a SHAPE question, not an existence one — it returns 1 for a container
    /// that ships no head at all. Ask `Model::has_mtp` / `Model::mtp` for existence.
    pub fn mtp_head_layers(&self) -> usize {
        self.mtp_layer_kind.len().max(1)
    }

    /// Validation shared by both architectures (the C `CKR` choke point).
    fn validate_common(&self) -> Result<(), ConfigError> {
        ckr!("hidden_size", self.hidden, 1, 1 << 20);
        ckr!("num_hidden_layers", self.n_layers, 1, 128);
        ckr!("num_attention_heads", self.n_heads, 1, 1024);
        ckr!("n_routed_experts", self.n_experts, 1, 4096);
        ckr!("num_experts_per_tok", self.topk, 1, 64);
        ckr!("moe_intermediate_size", self.moe_inter, 1, 1 << 20);
        // The dense FFN width is only used by leading dense layers; an all-MoE model
        // (e.g. MiniMax-M2, first_dense == 0) legitimately omits it → 0.
        if self.first_dense > 0 {
            ckr!("intermediate_size", self.dense_inter, 1, 1 << 24);
        } else {
            ckr!("intermediate_size", self.dense_inter, 0, 1 << 24);
        }
        ckr!("first_k_dense_replace", self.first_dense, 0, self.n_layers);
        ckr!("v_head_dim", self.v_head, 1, 1 << 16);
        ckr!("n_shared_experts", self.n_shared, 0, 64);
        ckr!("vocab_size", self.vocab, 1, 1 << 24);
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    // A minimal GLM-5.2-shaped config (values from the README architecture notes).
    fn glm_json() -> Json {
        let text = r#"{
            "hidden_size": 6144,
            "num_hidden_layers": 78,
            "num_attention_heads": 64,
            "n_routed_experts": 256,
            "num_experts_per_tok": 8,
            "moe_intermediate_size": 2048,
            "intermediate_size": 12288,
            "first_k_dense_replace": 3,
            "q_lora_rank": 2048,
            "kv_lora_rank": 512,
            "qk_nope_head_dim": 128,
            "qk_rope_head_dim": 64,
            "v_head_dim": 128,
            "n_shared_experts": 1,
            "vocab_size": 151552,
            "n_group": 1,
            "topk_group": 1,
            "norm_topk_prob": true,
            "rms_norm_eps": 1e-5,
            "routed_scaling_factor": 2.5,
            "rope_parameters": {"rope_theta": 10000.0},
            "eos_token_id": [151329, 151336, 151338],
            "index_topk": 2048,
            "index_n_heads": 64,
            "index_head_dim": 128
        }"#;
        Json::parse(text).unwrap()
    }

    // A minimal MiniMax-M3-shaped config (values from nvidia/MiniMax-M3-NVFP4).
    fn minimax_json() -> Json {
        let text = r#"{
            "model_type": "minimax_m3_vl",
            "text_config": {
                "hidden_size": 6144,
                "intermediate_size": 3072,
                "num_hidden_layers": 60,
                "num_attention_heads": 64,
                "num_key_value_heads": 4,
                "head_dim": 128,
                "vocab_size": 200064,
                "max_position_embeddings": 1048576,
                "rms_norm_eps": 1e-06,
                "use_gemma_norm": true,
                "rope_theta": 5000000,
                "rotary_dim": 64,
                "partial_rotary_factor": 0.5,
                "hidden_act": "swigluoai",
                "use_qk_norm": true,
                "dense_intermediate_size": 12288,
                "shared_intermediate_size": 3072,
                "num_local_experts": 128,
                "num_experts_per_tok": 4,
                "n_shared_experts": 1,
                "scoring_func": "sigmoid",
                "use_routing_bias": true,
                "moe_layer_freq": [0,0,0,1,1,1],
                "swiglu_alpha": 1.702,
                "swiglu_limit": 7.0,
                "routed_scaling_factor": 2.0,
                "eos_token_id": [200020],
                "sparse_attention_config": {
                    "use_sparse_attention": true,
                    "sparse_index_dim": 128,
                    "sparse_num_index_heads": 4,
                    "sparse_topk_blocks": 16,
                    "sparse_block_size": 128,
                    "sparse_local_block": 1,
                    "sparse_attention_freq": [0,0,0,1,1,1]
                }
            }
        }"#;
        Json::parse(text).unwrap()
    }

    /// The real `deepgrove/maple-preview` config.json, trimmed only of fields no arch
    /// reads (dropout rates, `transformers_version`, `auto_map`). `layer_types` is
    /// verbatim: three sliding then one full, six times over.
    fn maple_json() -> Json {
        let text = r#"{
            "architectures": ["MapleForCausalLM"],
            "bos_token_id": 151643, "eos_token_id": 151645,
            "head_dim": 128, "hidden_act": "silu", "hidden_size": 2048,
            "intermediate_size": 4096,
            "layer_types": [
                "sliding_attention","sliding_attention","sliding_attention","full_attention",
                "sliding_attention","sliding_attention","sliding_attention","full_attention",
                "sliding_attention","sliding_attention","sliding_attention","full_attention",
                "sliding_attention","sliding_attention","sliding_attention","full_attention",
                "sliding_attention","sliding_attention","sliding_attention","full_attention",
                "sliding_attention","sliding_attention","sliding_attention","full_attention"],
            "max_position_embeddings": 131072, "max_window_layers": 24,
            "moe_intermediate_size": 512, "moe_router_enable_expert_bias": false,
            "nope_on_global_attention": true, "norm_topk_prob": true,
            "num_attention_heads": 16, "num_experts": 256, "num_experts_per_tok": 8,
            "num_hidden_layers": 24, "num_key_value_heads": 4, "num_shared_experts": 0,
            "partial_rotary_factor": 0.5, "quantize": true, "rms_norm_eps": 1e-06,
            "rope_scaling": null, "rope_theta": 10000, "router_dtype": "fp32",
            "sliding_window": 512, "tie_word_embeddings": false,
            "use_cache": true, "use_qk_norm": true, "use_rmsnorm": true,
            "vocab_size": 151936
        }"#;
        Json::parse(text).unwrap()
    }

    #[test]
    fn loads_maple_shape() {
        let c = Config::from_json(&maple_json()).unwrap();
        assert_eq!(c.arch, Arch::Maple);
        assert_eq!(c.hidden, 2048);
        assert_eq!(c.n_layers, 24);
        assert_eq!(c.n_heads, 16);
        assert_eq!(c.n_kv_heads, 4);
        // partial_rotary_factor 0.5 of head_dim 128 -> 64 roped, 64 passed through.
        assert_eq!(c.qk_head, 128);
        assert_eq!(c.qk_rope, 64);
        assert_eq!(c.qk_nope, 64);
        assert_eq!(c.v_head, 128);
        assert_eq!(c.n_experts, 256);
        assert_eq!(c.topk, 8);
        // The expert width is `moe_intermediate_size` (512). `intermediate_size` (4096)
        // describes a dense FFN this checkpoint has none of — reading it sizes every
        // expert 8x too wide, which loads and generates rather than failing.
        assert_eq!(c.moe_inter, 512);
        assert_eq!(c.dense_inter, 0);
        // All-MoE, no shared expert.
        assert_eq!(c.first_dense, 0);
        assert_eq!(c.n_shared, 0);
        assert_eq!(c.moe_layers(), (24, 0));
        assert!(c.qk_norm);
        assert!(!c.gemma_norm);
        // Clamped SwiGLU without the OAI sigmoid gate, despite `hidden_act: "silu"`.
        assert!(!c.swiglu_oai);
        assert_eq!(c.swiglu_limit, 7.0);
        assert!(c.norm_topk);
        assert_eq!(c.stop_ids, vec![151645]);
        assert!((c.attn_scale - 1.0 / (128f32).sqrt()).abs() < 1e-6);
    }

    /// The router axis is `router_score`, NOT `sigmoid_route` — which is parsed and read
    /// by nothing. If someone ever wires the old flag up, GLM (false, yet running the
    /// sigmoid path) breaks; this pins both halves of that.
    #[test]
    fn maple_routes_by_softmax_and_sigmoid_route_stays_dead() {
        let m = Config::from_json(&maple_json()).unwrap();
        assert_eq!(m.router_score, RouterScore::Softmax);
        for (name, c) in [
            ("glm", Config::from_json(&glm_json()).unwrap()),
            ("minimax", Config::from_json(&minimax_json()).unwrap()),
        ] {
            assert_eq!(
                c.router_score,
                RouterScore::Sigmoid,
                "{name} must keep the unconditional sigmoid `route()` applies today"
            );
        }
    }

    /// The 3:1 interleave, and the fact that RoPE tracks it. Layers 3, 7, 11, 15, 19, 23
    /// are the global ones, and under `nope_on_global_attention` they carry no positional
    /// encoding — a model that ropes them anyway still emits fluent text.
    #[test]
    fn maple_sliding_full_interleave_and_nope_on_global() {
        let c = Config::from_json(&maple_json()).unwrap();
        assert_eq!(c.swa_window, 512);
        assert!(c.nope_on_global);
        assert_eq!(c.layer_swa.len(), 24);
        for i in 0..24 {
            let global = i % 4 == 3;
            assert_eq!(c.layer_is_swa(i), !global, "layer {i}");
            assert_eq!(c.layer_uses_rope(i), !global, "layer {i} rope");
        }
        assert_eq!(c.layer_swa.iter().filter(|b| **b).count(), 18);

        // A sliding layer's KV is bounded by the window however long the context; a
        // global layer's is not. This is what makes 131k context cheap here.
        assert_eq!(c.layer_kv_span(0, 131_072), 512);
        assert_eq!(c.layer_kv_span(3, 131_072), 131_072);
        assert_eq!(c.layer_kv_span(0, 100), 100, "window is a cap, not a floor");

        // `swa_window` is not V4's raw-KV ring; they must not alias.
        assert_eq!(c.window, 0);
    }

    /// Every pre-Maple arch must answer "full attention, roped" for every layer, so the
    /// new per-layer predicates are inert on the shipped fleet.
    #[test]
    fn swa_predicates_are_inert_on_every_other_arch() {
        for (name, c) in [
            ("glm", Config::from_json(&glm_json()).unwrap()),
            ("minimax", Config::from_json(&minimax_json()).unwrap()),
        ] {
            assert!(c.layer_swa.is_empty(), "{name}");
            assert_eq!(c.swa_window, 0, "{name}");
            assert!(!c.nope_on_global, "{name}");
            for i in 0..(c.n_layers as usize) {
                assert!(!c.layer_is_swa(i), "{name} layer {i}");
                assert!(c.layer_uses_rope(i), "{name} layer {i}");
                assert_eq!(c.layer_kv_span(i, 4096), 4096, "{name} layer {i}");
            }
        }
    }

    /// Maple carries no `text_config`, so without its own arm it lands in
    /// `from_json_glm` and parses as MLA — a different attention family, silently.
    #[test]
    fn maple_is_not_swallowed_by_the_glm_fallthrough() {
        let c = Config::from_json(&maple_json()).unwrap();
        assert_eq!(c.arch, Arch::Maple, "must not fall through to GLM");
        assert_eq!(c.kv_lora, 0, "GQA, not MLA: no latent KV");
        assert_eq!(c.q_lora, 0);
    }

    /// `is_gqa` is a `matches!` over a closed set, so a new arch defaults OUT of it — and
    /// the default is wrong for Maple in the silent direction: a GQA model routed down
    /// the MLA path. Nothing in the build catches that, so it is pinned here.
    #[test]
    fn maple_is_in_the_gqa_family_and_not_latent_moe() {
        let c = Config::from_json(&maple_json()).unwrap();
        assert!(c.arch.is_gqa(), "Maple is q/k/v + a KV cache of n_kv_heads");
        assert!(
            !c.arch.routed_experts_are_latent(),
            "Maple's experts are at model hidden (2048 -> 512), not in a moe_latent bottleneck"
        );
        assert_eq!(c.moe_latent, 0);
    }

    #[test]
    fn loads_glm_shape() {
        let c = Config::from_json(&glm_json()).unwrap();
        assert_eq!(c.arch, Arch::GlmMoeDsa);
        assert_eq!(c.hidden, 6144);
        assert_eq!(c.n_layers, 78);
        assert_eq!(c.qk_head, 128 + 64);
        assert!(!c.mla_nope, "GLM MLA is roped");
        assert_eq!(c.stop_ids, vec![151329, 151336, 151338]);
        assert!(c.norm_topk);
        assert!(!c.gemma_norm && !c.swiglu_oai && !c.sigmoid_route);
        assert!((c.attn_scale - 1.0 / (192f32).sqrt()).abs() < 1e-6);
        assert_eq!(c.idx_type.len(), 78);
    }

    #[test]
    fn loads_minimax_shape() {
        let c = Config::from_json(&minimax_json()).unwrap();
        assert_eq!(c.arch, Arch::MinimaxM3);
        assert_eq!(c.hidden, 6144);
        assert_eq!(c.n_layers, 60);
        assert_eq!(c.n_heads, 64);
        assert_eq!(c.n_kv_heads, 4);
        // GQA head geometry folded onto qk_nope/qk_rope: head_dim 128, rotary 64.
        assert_eq!(c.qk_head, 128);
        assert_eq!(c.qk_rope, 64);
        assert_eq!(c.qk_nope, 64);
        assert_eq!(c.v_head, 128);
        assert_eq!(c.n_experts, 128);
        assert_eq!(c.topk, 4);
        assert_eq!(c.moe_inter, 3072);
        assert_eq!(c.dense_inter, 12288);
        assert_eq!(c.shared_inter, 3072);
        assert_eq!(c.first_dense, 3); // three leading zeros in moe_layer_freq
        assert_eq!(c.vocab, 200064);
        assert_eq!(c.max_ctx, 1048576);
        assert!(c.qk_norm && c.gemma_norm && c.swiglu_oai && c.sigmoid_route);
        // Block-sparse Lightning Indexer geometry + per-layer flags.
        assert_eq!(c.index_nh, 4);
        assert_eq!(c.index_hd, 128);
        assert_eq!(c.index_block_size, 128);
        assert_eq!(c.index_topk_blocks, 16);
        assert_eq!(c.index_local_blocks, 1);
        assert_eq!(c.index_topk, 16 * 128); // effective token budget
        assert_eq!(&c.idx_type[..6], &[false, false, false, true, true, true]); // first 3 dense
        assert!((c.swiglu_alpha - 1.702).abs() < 1e-6);
        assert!((c.swiglu_limit - 7.0).abs() < 1e-6);
        assert!((c.routed_scale - 2.0).abs() < 1e-6);
        assert!((c.attn_scale - 1.0 / (128f32).sqrt()).abs() < 1e-6);
        assert!((c.theta - 5_000_000.0).abs() < 1.0);
        assert_eq!(c.stop_ids, vec![200020]);
        assert_eq!(c.idx_type.len(), 60);
    }

    // A minimal MiniMax-M2-shaped config (flat, values from nvidia/MiniMax-M2.7-NVFP4).
    // Note: no `text_config` nesting, `hidden_act` silu, no gemma-norm, no shared
    // expert, no dense layers, no sparse-attention config, scalar eos.
    fn minimax_m2_json() -> Json {
        let text = r#"{
            "model_type": "minimax_m2",
            "architectures": ["MiniMaxM2ForCausalLM"],
            "hidden_size": 3072,
            "intermediate_size": 1536,
            "num_hidden_layers": 62,
            "num_attention_heads": 48,
            "num_key_value_heads": 8,
            "head_dim": 128,
            "vocab_size": 200064,
            "max_position_embeddings": 196608,
            "rms_norm_eps": 1e-06,
            "rope_theta": 5000000,
            "rotary_dim": 64,
            "partial_rotary_factor": 0.5,
            "hidden_act": "silu",
            "use_qk_norm": true,
            "qk_norm_type": "per_layer",
            "num_local_experts": 256,
            "num_experts_per_tok": 8,
            "shared_intermediate_size": 0,
            "scoring_func": "sigmoid",
            "use_routing_bias": true,
            "eos_token_id": 2
        }"#;
        Json::parse(text).unwrap()
    }

    #[test]
    fn loads_minimax_m2_shape() {
        let c = Config::from_json(&minimax_m2_json()).unwrap();
        assert_eq!(c.arch, Arch::MinimaxM2);
        assert_eq!(c.hidden, 3072);
        assert_eq!(c.n_layers, 62);
        assert_eq!(c.n_heads, 48);
        assert_eq!(c.n_kv_heads, 8);
        // GQA head geometry: head_dim 128, rotary 64 → nope 64.
        assert_eq!(c.qk_head, 128);
        assert_eq!(c.qk_rope, 64);
        assert_eq!(c.qk_nope, 64);
        assert_eq!(c.v_head, 128);
        assert_eq!(c.n_experts, 256);
        assert_eq!(c.topk, 8);
        assert_eq!(c.moe_inter, 1536);
        // No shared expert, no dense layers.
        assert_eq!(c.shared_inter, 0);
        assert_eq!(c.n_shared, 0);
        assert_eq!(c.first_dense, 0);
        assert_eq!(c.vocab, 200064);
        assert_eq!(c.max_ctx, 196608);
        // Flag-driven forward diffs vs M3: qk-norm on, but plain SwiGLU + standard
        // RMSNorm (no gemma), sigmoid+bias router.
        assert!(c.qk_norm && c.sigmoid_route);
        assert!(!c.gemma_norm, "M2 uses standard RMSNorm, not gemma-norm");
        assert!(!c.swiglu_oai, "M2 uses plain SwiGLU (silu), not swigluoai");
        // Dense attention everywhere (no Lightning Indexer).
        assert_eq!(c.index_topk, 0);
        assert_eq!(c.idx_type.len(), 62);
        assert!(c.idx_type.iter().all(|&x| !x));
        assert!((c.theta - 5_000_000.0).abs() < 1.0);
        assert!((c.attn_scale - 1.0 / (128f32).sqrt()).abs() < 1e-6);
        assert_eq!(c.stop_ids, vec![2]); // scalar eos_token_id
    }

    #[test]
    fn rejects_out_of_range() {
        let mut text = glm_json();
        if let Json::Obj(_) = &text {
            // Rebuild with a hostile layer count.
            text = Json::parse(
                &r#"{"hidden_size":6144,"num_hidden_layers":9999,"num_attention_heads":64,
                    "n_routed_experts":256,"num_experts_per_tok":8,"moe_intermediate_size":2048,
                    "intermediate_size":12288,"first_k_dense_replace":3,"q_lora_rank":2048,
                    "kv_lora_rank":512,"qk_nope_head_dim":128,"qk_rope_head_dim":64,"v_head_dim":128,
                    "n_shared_experts":1,"vocab_size":151552,"n_group":1,"index_topk":0,
                    "index_n_heads":0,"index_head_dim":0}"#
                    .to_string(),
            )
            .unwrap();
        }
        match Config::from_json(&text) {
            Err(ConfigError::Range { name, .. }) => assert_eq!(name, "num_hidden_layers"),
            other => panic!("expected range error, got {other:?}"),
        }
    }

    #[test]
    fn requires_n_group_1() {
        let text = Json::parse(r#"{"n_group": 8, "num_hidden_layers": 4}"#).unwrap();
        assert!(matches!(
            Config::from_json(&text),
            Err(ConfigError::Unsupported(_))
        ));
    }

    #[test]
    fn loads_nemotron_h_shape() {
        // Real NVIDIA-Nemotron-3-Super-120B-A12B-NVFP4 hyperparameters, with a short
        // 8-layer hybrid pattern (M/E/*) standing in for the full 88.
        let text = Json::parse(
            r#"{
            "model_type": "nemotron_h",
            "hidden_size": 4096, "num_hidden_layers": 8,
            "num_attention_heads": 32, "num_key_value_heads": 2, "head_dim": 128,
            "partial_rotary_factor": 1.0, "rope_theta": 10000, "layer_norm_epsilon": 1e-5,
            "vocab_size": 131072, "max_position_embeddings": 262144,
            "hybrid_override_pattern": "MEMEMEM*",
            "n_routed_experts": 512, "num_experts_per_tok": 22, "moe_intermediate_size": 2688,
            "moe_latent_size": 1024, "moe_shared_expert_intermediate_size": 5376,
            "n_shared_experts": 1, "routed_scaling_factor": 5.0, "norm_topk_prob": true,
            "mlp_hidden_act": "relu2", "time_step_min": 0.001,
            "ssm_state_size": 128, "conv_kernel": 4, "mamba_num_heads": 128,
            "mamba_head_dim": 64, "n_groups": 8, "chunk_size": 128
        }"#,
        )
        .unwrap();
        let c = Config::from_json(&text).expect("nemotron_h parse");
        assert_eq!(c.arch, Arch::NemotronH);
        assert!(
            !c.arch.is_gqa(),
            "NemotronH is not blanket-GQA (per-layer dispatch)"
        );
        assert_eq!(c.hidden, 4096);
        assert_eq!(c.n_kv_heads, 2);
        assert_eq!((c.qk_rope, c.qk_nope, c.qk_head), (128, 0, 128)); // full rope
        assert_eq!(c.mamba_d_state, 128);
        assert_eq!(
            (c.mamba_n_heads, c.mamba_head_dim, c.mamba_inter),
            (128, 64, 8192)
        );
        assert_eq!(
            (c.mamba_n_groups, c.mamba_d_conv, c.mamba_chunk),
            (8, 4, 128)
        );
        assert_eq!(
            (c.moe_latent, c.moe_inter, c.shared_inter),
            (1024, 2688, 5376)
        );
        assert!(c.relu2 && c.sigmoid_route && c.norm_topk);
        assert_eq!(c.routed_scale, 5.0);
        assert!((c.mamba_dt_min - 0.001).abs() < 1e-9);
        assert_eq!(
            c.layer_kind,
            vec![
                LayerKind::Mamba,
                LayerKind::Moe,
                LayerKind::Mamba,
                LayerKind::Moe,
                LayerKind::Mamba,
                LayerKind::Moe,
                LayerKind::Mamba,
                LayerKind::Attn,
            ]
        );
        // No `mtp_hybrid_override_pattern` in this checkpoint -> no head sublayers, but
        // `mtp_head_layers()` still answers the shape question with the 1-block default.
        assert!(c.mtp_layer_kind.is_empty());
        assert_eq!(c.mtp_head_layers(), 1);
    }

    /// `mtp_hybrid_override_pattern` describes the speculative head, NOT the main stack:
    /// it must land in its own vector (`layer_kind` stays exactly `num_hidden_layers`
    /// long — the KV accounting and the loader both iterate it) and drive
    /// `mtp_head_layers()`.
    #[test]
    fn nemotron_mtp_pattern_is_separate_from_the_main_stack() {
        let text = Json::parse(
            r#"{
            "model_type": "nemotron_h",
            "hidden_size": 4096, "num_hidden_layers": 8,
            "num_attention_heads": 32, "num_key_value_heads": 2, "head_dim": 128,
            "layer_norm_epsilon": 1e-5, "vocab_size": 131072,
            "hybrid_override_pattern": "MEMEMEM*",
            "mtp_hybrid_override_pattern": "*E", "num_nextn_predict_layers": 1,
            "n_routed_experts": 512, "num_experts_per_tok": 22, "moe_intermediate_size": 2688,
            "moe_latent_size": 1024, "moe_shared_expert_intermediate_size": 5376,
            "mlp_hidden_act": "relu2",
            "ssm_state_size": 128, "conv_kernel": 4, "mamba_num_heads": 128,
            "mamba_head_dim": 64, "n_groups": 8, "chunk_size": 128
        }"#,
        )
        .unwrap();
        let c = Config::from_json(&text).expect("nemotron_h + mtp parse");
        assert_eq!(
            c.layer_kind.len(),
            8,
            "main stack unchanged by the head pattern"
        );
        assert_eq!(c.mtp_layer_kind, vec![LayerKind::Attn, LayerKind::Moe]);
        assert_eq!(
            c.mtp_head_layers(),
            2,
            "head occupies n_layers and n_layers+1"
        );
    }

    /// A bad character in the head pattern must be rejected with the HEAD's field name,
    /// so the error points at the right config key.
    #[test]
    fn nemotron_rejects_unknown_mtp_pattern_char() {
        let text = Json::parse(
            r#"{
            "model_type": "nemotron_h",
            "hidden_size": 8, "num_hidden_layers": 1,
            "num_attention_heads": 2, "num_key_value_heads": 1, "head_dim": 2,
            "layer_norm_epsilon": 1e-5, "vocab_size": 8,
            "hybrid_override_pattern": "E", "mtp_hybrid_override_pattern": "*X",
            "n_routed_experts": 4, "num_experts_per_tok": 2, "moe_intermediate_size": 4,
            "moe_latent_size": 2, "moe_shared_expert_intermediate_size": 4,
            "mlp_hidden_act": "relu2",
            "ssm_state_size": 2, "conv_kernel": 2, "mamba_num_heads": 2,
            "mamba_head_dim": 2, "n_groups": 1, "chunk_size": 2
        }"#,
        )
        .unwrap();
        match Config::from_json(&text) {
            Err(ConfigError::Unsupported(m)) => {
                assert!(
                    m.contains("mtp_hybrid_override_pattern"),
                    "wrong field named: {m}"
                )
            }
            other => panic!("expected Unsupported, got {other:?}"),
        }
    }

    #[test]
    fn nemotron_pattern_length_must_match_layers() {
        let text = Json::parse(
            r#"{"model_type":"nemotron_h","hidden_size":4096,"num_hidden_layers":8,
            "num_attention_heads":32,"num_key_value_heads":2,"head_dim":128,
            "vocab_size":131072,"hybrid_override_pattern":"ME",
            "n_routed_experts":512,"num_experts_per_tok":22,"moe_intermediate_size":2688,
            "moe_latent_size":1024,"moe_shared_expert_intermediate_size":5376,
            "ssm_state_size":128,"conv_kernel":4,"mamba_num_heads":128,"mamba_head_dim":64,
            "n_groups":8,"chunk_size":128}"#,
        )
        .unwrap();
        assert!(matches!(
            Config::from_json(&text),
            Err(ConfigError::Unsupported(_))
        ));
    }

    // ---- Kimi-K3 ----------------------------------------------------------------

    /// The real K3 geometry: 93 layers, gated MLA on the 1-indexed `full_attn_layers`,
    /// KDA everywhere else. `attn` lets a test perturb just that list.
    fn kimi_k3_text(attn: &str) -> String {
        format!(
            r#"{{"model_type":"kimi_k3",
                "architectures":["KimiK3ForConditionalGeneration"],
                "text_config":{{
                  "hidden_size":7168,"num_hidden_layers":93,"num_attention_heads":96,
                  "num_key_value_heads":96,"num_experts":896,"num_experts_per_token":16,
                  "num_shared_experts":2,"moe_intermediate_size":3072,
                  "intermediate_size":33792,"routed_expert_hidden_size":3584,
                  "first_k_dense_replace":1,"q_lora_rank":1536,"kv_lora_rank":512,
                  "qk_nope_head_dim":128,"qk_rope_head_dim":64,"v_head_dim":128,
                  "vocab_size":163840,"max_position_embeddings":1048576,
                  "rms_norm_eps":1e-05,"rope_theta":10000.0,"moe_renormalize":true,
                  "mla_use_nope":true,"mla_use_output_gate":true,"hidden_act":"situ",
                  "activation_situ_beta":4.0,"activation_situ_linear_beta":25.0,
                  "moe_router_activation_func":"sigmoid","num_expert_group":1,
                  "topk_group":1,"routed_scaling_factor":1.0,"eos_token_id":163586,
                  "attn_res_block_size":12,
                  "linear_attn_config":{{"head_dim":128,"num_heads":96,
                    "short_conv_kernel_size":4,"full_attn_layers":[{attn}]}}}},
                "vision_config":{{"vt_hidden_size":1024}}}}"#
        )
    }

    const K3_ATTN: &str = "4,8,12,16,20,24,28,32,36,40,44,48,52,56,60,64,68,72,76,80,84,88,92,93";

    fn kimi_k3_cfg(attn: &str) -> Result<Config, ConfigError> {
        Config::from_json(&Json::parse(&kimi_k3_text(attn)).unwrap())
    }

    /// `full_attn_layers` / `kda_layers` are **1-indexed** in the K3 config: the list ends
    /// at 93 on a 93-layer stack, and checkpoint layer 0 is KDA. Off-by-one here shifts
    /// every mixer by one layer, which no later stage can detect.
    #[test]
    fn kimi_k3_converts_the_one_indexed_mixer_lists() {
        let c = kimi_k3_cfg(K3_ATTN).expect("kimi_k3 parse");
        assert_eq!(c.arch, Arch::KimiK3);
        assert!(!c.arch.is_gqa(), "K3 is MLA + KDA, not the GQA family");
        assert_eq!(c.layer_kind.len(), 93);
        assert_eq!(
            c.layer_kind[0],
            LayerKind::Kda,
            "config layer 1 -> index 0, KDA"
        );
        assert_eq!(
            c.layer_kind[3],
            LayerKind::Attn,
            "config layer 4 -> index 3, MLA"
        );
        assert_eq!(c.layer_kind[91], LayerKind::Attn, "config 92");
        assert_eq!(
            c.layer_kind[92],
            LayerKind::Attn,
            "config 93, the trailing MLA layer"
        );
        let n_attn = c
            .layer_kind
            .iter()
            .filter(|k| **k == LayerKind::Attn)
            .count();
        let n_kda = c
            .layer_kind
            .iter()
            .filter(|k| **k == LayerKind::Kda)
            .count();
        assert_eq!((n_attn, n_kda), (24, 69));
    }

    /// The rest of the geometry, including the two places K3 differs from every other
    /// arch: experts run in a `routed_expert_hidden_size` latent, and the two shared
    /// experts ship fused as one `2 * moe_intermediate_size`-wide pair.
    #[test]
    fn kimi_k3_parses_latent_moe_and_fused_shared_experts() {
        let c = kimi_k3_cfg(K3_ATTN).expect("kimi_k3 parse");
        assert_eq!((c.hidden, c.n_layers, c.n_heads), (7168, 93, 96));
        assert_eq!((c.n_experts, c.topk, c.n_shared), (896, 16, 2));
        assert_eq!(c.moe_latent, 3584, "Stable LatentMoE bottleneck");
        assert_eq!(c.shared_inter, 6144, "2 x 3072, fused in the checkpoint");
        assert_eq!(
            (c.kv_lora, c.qk_nope, c.qk_rope, c.v_head),
            (512, 128, 64, 128)
        );
        // The MLA per-head query width, fixed up after the literal like GLM's. This is
        // the attention geometry only — the GQA full-KV charge is gated on
        // `allocates_gqa_kv` (false for K3), so a non-zero `qk_head` costs no KV.
        assert_eq!(c.qk_head, 128 + 64, "q_head_dim = qk_nope + qk_rope");
        // NoPE: the reference asserts `use_nope` and sets `rotary_emb = None`. The 64
        // rope dims still exist in the projections, so nothing about the SHAPES reveals
        // this — only the config flag does.
        assert!(c.mla_nope, "K3 MLA is NoPE");
        // `situ` applies to the dense MLP, the shared experts AND the routed experts.
        assert!(c.situ, "hidden_act = situ");
        assert_eq!((c.situ_beta, c.situ_linear_beta), (4.0, 25.0));
        assert!(
            !c.swiglu_oai && !c.relu2,
            "situ must not also select another variant"
        );
        assert!(
            (c.attn_scale - 1.0 / 192f32.sqrt()).abs() < 1e-9,
            "scale = q_head_dim^-0.5"
        );
        assert_eq!((c.kda_n_heads, c.kda_head_dim, c.kda_d_conv), (96, 128, 4));
        assert_eq!((c.first_dense, c.max_ctx, c.vocab), (1, 1048576, 163840));
        assert!(c.sigmoid_route && c.norm_topk);
        assert_eq!(c.index_topk, 0, "no DSA indexer on K3");
    }

    /// K3 carries a `text_config` because it is a VL model, and the MiniMax-M3 branch
    /// claims *any* config that has one. If the K3 check is ever reordered below it, K3
    /// silently parses as M3 — wrong attention family, wrong expert geometry, no KDA
    /// state, and nothing downstream to notice.
    #[test]
    fn kimi_k3_is_not_swallowed_by_the_minimax_m3_text_config_check() {
        let c = kimi_k3_cfg(K3_ATTN).expect("kimi_k3 parse");
        assert_eq!(c.arch, Arch::KimiK3, "must not fall through to MinimaxM3");
        assert!(
            !c.layer_kind.is_empty(),
            "M3 would have left layer_kind empty"
        );
    }

    /// MoE-ness is NOT on the `layer_kind` axis for K3: every layer past `first_dense`
    /// carries experts regardless of its mixer. Asking `layer_kind` for MoE layers counts
    /// zero and mis-sizes the expert cache by the whole model.
    #[test]
    fn kimi_k3_moe_layers_come_from_the_prefix_rule() {
        let c = kimi_k3_cfg(K3_ATTN).expect("kimi_k3 parse");
        assert!(
            !c.layer_kind.contains(&LayerKind::Moe),
            "no Moe on the mixer axis"
        );
        assert_eq!(c.moe_layers(), (92, 1), "layers 1..=92, probe at the first");
    }

    /// ...while Nemotron-H keeps using the explicit axis, where the prefix rule would be
    /// wrong (its `first_dense` layer is a Mamba layer holding no experts at all).
    #[test]
    fn nemotron_moe_layers_still_come_from_layer_kind() {
        let text = Json::parse(
            r#"{"model_type":"nemotron_h","hidden_size":4096,"num_hidden_layers":8,
            "num_attention_heads":32,"num_key_value_heads":2,"head_dim":128,
            "vocab_size":131072,"hybrid_override_pattern":"MEMEMEM*",
            "n_routed_experts":512,"num_experts_per_tok":22,"moe_intermediate_size":2688,
            "moe_latent_size":1024,"moe_shared_expert_intermediate_size":5376,
            "ssm_state_size":128,"conv_kernel":4,"mamba_num_heads":128,"mamba_head_dim":64,
            "n_groups":8,"chunk_size":128}"#,
        )
        .unwrap();
        let c = Config::from_json(&text).expect("nemotron parse");
        assert_eq!(
            c.moe_layers(),
            (3, 1),
            "the three 'E' layers, first at index 1"
        );
    }

    /// A `full_attn_layers` list that names no layer of this stack (e.g. someone fed it
    /// 0-indexed values against a 1-indexed contract, or an empty list) would make every
    /// layer KDA and size the KV cache to zero. Reject at parse rather than serve a model
    /// whose every reservation is 0.
    #[test]
    fn kimi_k3_rejects_a_full_attn_list_naming_no_layer() {
        assert!(matches!(kimi_k3_cfg(""), Err(ConfigError::Unsupported(_))));
        assert!(matches!(
            kimi_k3_cfg("500,900"),
            Err(ConfigError::Unsupported(_))
        ));
    }
}
