//! Model and per-layer weight structures — port of the `Model`, `Layer`,
//! `ESlot`, and `KVState` structs from `c/glm.c`.
//!
//! # Status: SKELETON
//!
//! The field layout mirrors the C structs so the loader and forward pass can be
//! filled in against a known shape. Buffers that are hot-path detail (profiling
//! counters, GPU shadow caches) are elided until their subsystem is ported.

use crate::mamba2::SsmState;
use colibri_core::{Arch, Config, LayerKind, QTensor};
use colibri_safetensors::Shards;

/// A transformer layer: MLA attention (dense, quantized) plus either a dense MLP
/// (`sparse == false`) or the MoE block (`sparse == true`).
#[derive(Default)]
pub struct Layer {
    pub in_ln: Vec<f32>,
    pub post_ln: Vec<f32>,

    // MLA (dense, quantized) — GLM (arch == GlmMoeDsa). `o` is shared with GQA.
    pub q_a: QTensor,
    pub q_b: QTensor,
    pub kv_a: QTensor,
    pub kv_b: QTensor,
    pub o: QTensor,
    pub q_a_ln: Vec<f32>,
    pub kv_a_ln: Vec<f32>,

    // DeepSeek-V4 output projection: a LoRA PAIR replacing the single `o` above.
    // `o_a` is [n_groups*o_lora_rank, n_heads*head_dim/n_groups] and `o_b` is
    // [hidden, n_groups*o_lora_rank] (`o_groups` 8, `o_lora_rank` 1024). V3 and every
    // other arch here have a plain `o_proj`, so these are None elsewhere and `o` is
    // left empty on V4 — the two are mutually exclusive, never both populated.
    pub o_a: Option<QTensor>,
    pub o_b: Option<QTensor>,
    /// `o_a` pre-split into its `o_groups` row-blocks. V4's O-LoRA is block-diagonal —
    /// group `gi`'s rows read only group `gi`'s slice of the attention output — so the
    /// operation is `g` independent matmuls, not one big one. Split once at load so each
    /// takes the ordinary `matmul_qt` path; dequantizing `o_a` whole would be 134 MB per
    /// layer. Empty unless the arch has an O-LoRA.
    pub o_a_groups: Vec<QTensor>,
    // Per-head attention sink, f32 [n_heads] (DeepSeek-V4). Empty elsewhere.
    pub attn_sink: Vec<f32>,

    // DeepSeek-V4 Hyper-Connections, one set per sublayer (attention and FFN). These are
    // the weights of the RESIDUAL STREAM itself, not of a sublayer: `hc_pre` collapses the
    // `hc_mult` copies into one using them, and `hc_post` expands back. All f32 in the
    // checkpoint and small enough to stay dense:
    //   `*_fn`    [mix_width(hc), hc*hidden]  = [24, 16384] at hc=4, hidden=4096
    //   `*_base`  [mix_width(hc)]             = [24]
    //   `*_scale` [3]
    // Empty on every other arch. Without them a V4 model loads and computes garbage, so
    // the loader treats a missing set on a V4 checkpoint as an error rather than a default.
    // DeepSeek-V4 Compressor (41 of 43 layers). Learned gated pooling over `comp_ratio`
    // consecutive tokens, producing the compressed KV that carries ALL context beyond the
    // 128-token raw window — so this is not an optimisation, it is how V4 has long context
    // at all. `comp_ratio == 0` means this layer has no compressor.
    //
    // Ratio 4 turns on OVERLAPPING windows, which doubles the projection width: `coff` is
    // `1 + (ratio == 4)`, so `comp_wkv`/`comp_wgate` are [coff*head_dim, hidden] and
    // `comp_ape` is [ratio, coff*head_dim]. The shapes therefore depend on the ratio, and
    // getting the ratio wrong is a load error rather than a silent numeric one.
    // DeepSeek-V4 Indexer (21 of 43 layers): picks which compressed blocks attention
    // actually keys on. It carries its OWN Compressor — same algorithm as the main one but
    // at `index_head_dim` (128) instead of `head_dim` (512), so `compress_prefill` /
    // `compress_decode` serve both unchanged — plus a q projection off the SHARED q-LoRA
    // bottleneck and a per-head weighting.
    //
    // Distinct from `ix_wk`/`ix_wq`/`ix_wp` above, which are GLM's DSA indexer: that one
    // scores raw keys, this one scores compressed blocks and has no `wk` at all. V4 is
    // excluded from the GLM arm for exactly that reason.
    pub idx_wq_b: Option<QTensor>,
    pub idx_wproj: Option<QTensor>,
    pub idx_comp_wkv: Option<QTensor>,
    pub idx_comp_wgate: Option<QTensor>,
    pub idx_comp_ape: Vec<f32>,
    pub idx_comp_norm: Vec<f32>,

    pub comp_ratio: i32,
    pub comp_wkv: Option<QTensor>,
    pub comp_wgate: Option<QTensor>,
    pub comp_ape: Vec<f32>,
    pub comp_norm: Vec<f32>,
    /// DeepSeek-V4 hash routing: `[vocab, topk]` flattened, expert IDs. Non-empty only on
    /// the first `n_hash_layers` (3) layers. `u32` rather than `Vec<f32>` because these
    /// are indices — the container stores them as exactly-representable f32 only to avoid
    /// an integer tensor path used by one tensor, and they are converted once, at load.
    pub tid2eid: Vec<u32>,

    pub hc_attn_fn: Vec<f32>,
    pub hc_attn_base: Vec<f32>,
    pub hc_attn_scale: Vec<f32>,
    pub hc_ffn_fn: Vec<f32>,
    pub hc_ffn_base: Vec<f32>,
    pub hc_ffn_scale: Vec<f32>,
    // V4's per-sublayer input norms are NOT separate fields: the converter canonicalizes
    // `attn_norm`/`ffn_norm` to `input_layernorm`/`post_attention_layernorm`, so they land
    // in `in_ln`/`post_ln` like every other arch. The reference applies each AFTER
    // `hc_pre` and before the sublayer, which is the only V4-specific part.

    // GQA (MiniMax-M3, arch == MinimaxM3): standard q/k/v projections with per-head
    // QK-norm; RoPE is partial (see Config::qk_rope). `None`/empty on GLM, which
    // uses the MLA fields above instead. `o` (above) is the shared output proj.
    pub q_proj: Option<QTensor>, // hidden -> n_heads * head_dim
    pub k_proj: Option<QTensor>, // hidden -> n_kv_heads * head_dim
    pub v_proj: Option<QTensor>, // hidden -> n_kv_heads * head_dim
    /// q/k/v concatenated row-wise (`[n_heads*head_dim + 2*n_kv_heads*head_dim, hidden]`),
    /// built at load time so `attention_gqa` runs ONE fused matmul per layer instead of
    /// three (q/k/v share the same input). When set, `q_proj`/`k_proj`/`v_proj` are dropped.
    pub qkv_proj: Option<QTensor>,
    pub q_norm: Vec<f32>, // per-head RMSNorm weight [head_dim] (gemma-folded)
    pub k_norm: Vec<f32>, // per-head RMSNorm weight [head_dim] (gemma-folded)

    // MiniMax-M3 block-sparse Lightning Indexer (present only on sparse attention
    // layers; see `Config::idx_type` for M3). Empty/None on GLM and on M3 dense layers.
    pub idx_q_proj: Option<QTensor>, // hidden -> index_n_heads * index_head_dim
    pub idx_k_proj: Option<QTensor>, // hidden -> index_head_dim (MQA: one key head)
    pub idx_q_norm: Vec<f32>,        // per-head index RMSNorm [index_head_dim] (gemma-folded)
    pub idx_k_norm: Vec<f32>,        // index-key RMSNorm [index_head_dim] (gemma-folded)

    pub sparse: bool,

    // dense mlp (sparse == false)
    pub gate_proj: QTensor,
    pub up_proj: QTensor,
    pub down_proj: QTensor,

    // moe (sparse == true) — router weights stay f32 (numerically sensitive)
    pub router: Vec<f32>,
    pub router_bias: Vec<f32>,
    pub sh_gate: QTensor,
    pub sh_up: QTensor,
    pub sh_down: QTensor,

    // DSA lightning indexer (present only on FULL indexer layers, i.e. when the
    // checkpoint was converted with the indexer weights). `None`/empty → no DSA on
    // this layer, so attention runs the dense path. See `crate::dsa`.
    pub ix_wk: Option<QTensor>, // key proj: hidden -> index_hd
    pub ix_wq: Option<QTensor>, // query proj: q_lora -> index_nh*index_hd
    pub ix_wp: Option<QTensor>, // per-head weight proj: hidden -> index_nh
    pub ix_knorm_w: Vec<f32>,   // key LayerNorm weight (eps 1e-6)
    pub ix_knorm_b: Vec<f32>,   // key LayerNorm bias

    // ---- Nemotron-H Mamba2 mixer (present on `LayerKind::Mamba` layers) ----
    // `in_ln` above is the single block-input RMSNorm; Nemotron has no `post_ln`.
    pub mamba_in_proj: Option<QTensor>, // hidden -> d_inner + conv_dim + n_heads (18560)
    pub mamba_out_proj: Option<QTensor>, // d_inner -> hidden
    pub mamba_conv_w: Vec<f32>,         // depthwise conv [conv_dim, k] (from [conv_dim,1,k])
    pub mamba_conv_b: Vec<f32>,         // conv bias [conv_dim]
    pub mamba_a_log: Vec<f32>,          // [n_heads]; A = -exp(a_log)
    pub mamba_d: Vec<f32>,              // skip connection [n_heads]
    pub mamba_dt_bias: Vec<f32>,        // step bias [n_heads]
    pub mamba_norm: Vec<f32>,           // gated RMSNorm weight [d_inner]

    // ---- Nemotron-H latent-MoE projections (present on `LayerKind::Moe` layers) ----
    // Routed experts (`sh_*` unused here; routed via the expert provider) run in the
    // `moe_latent` space between these two projections; `router`/`router_bias` above
    // are reused for the gate. `up_proj`/`down_proj` reused for the shared expert.
    pub fc1_latent: Option<QTensor>, // hidden -> moe_latent
    pub fc2_latent: Option<QTensor>, // moe_latent -> hidden

    // ---- Kimi-K3: the output gate, on BOTH mixers ----
    // `g_proj` is present on all 93 layers (confirmed by sweeping the checkpoint
    // headers), gating the mixer output before `o`. KDA and gated MLA share it.
    pub attn_gate: Option<QTensor>, // hidden -> n_heads * head_dim

    // ---- Kimi-K3 KDA mixer (present on `LayerKind::Kda` layers) ----
    // q/k/v reuse `q_proj`/`k_proj`/`v_proj` above; `o` is the shared output proj.
    /// per-head decay coefficient projection, `hidden -> n_heads`
    pub kda_b_proj: Option<QTensor>,
    /// low-rank forget gate, factored: `hidden -> r` then `r -> n_heads * head_dim`
    pub kda_f_a: Option<QTensor>,
    pub kda_f_b: Option<QTensor>,
    /// short causal depthwise conv over q/k/v, each `[n_heads*head_dim, d_conv]`
    /// (stored `[C, 1, k]` on disk; read flat it is exactly `[C, k]`).
    pub kda_conv_q: Vec<f32>,
    pub kda_conv_k: Vec<f32>,
    pub kda_conv_v: Vec<f32>,
    /// `[head_dim]`; the decay is derived from this as Mamba's is from `A_log`
    pub kda_a_log: Vec<f32>,
    /// `[n_heads * head_dim]` step bias
    pub kda_dt_bias: Vec<f32>,
    /// `[head_dim]` output RMSNorm weight, applied per head
    pub kda_o_norm: Vec<f32>,

    // ---- Kimi-K3 "attention residuals" (every layer) ----
    // NOT a residual add, despite the name. K3 has NO ordinary residual stream: these
    // drive a softmax attention over a stack of saved hidden states, which REPLACES the
    // running state rather than adding to it (reference `_apply_attn_res`):
    //
    //     v      = [block_residual ; prefix_sum]        // [tokens, blocks+1, hidden]
    //     k      = v * rsqrt(mean(v^2) + eps)           // RMSNorm each candidate
    //     scores = sum(k * (res_norm * res_proj))       // the two [hidden] vecs multiplied
    //     out    = softmax(scores) @ v                  // weighted avg of the RAW v
    //
    // So `*_res_norm` and `*_res_proj` are not a norm and a projection applied in
    // sequence — they are multiplied elementwise into ONE `[hidden]` score vector. The
    // `[1, hidden]` shape of `*_res_proj` is why it looks like a projection.
    //
    // `block_residual` grows by one entry every `attn_res_block_size` (12) layers and
    // `prefix_sum` accumulates sublayer outputs across the whole stack, so this cannot
    // be evaluated per-layer — it needs stack-level state (see the driver).
    pub attn_res_norm: Vec<f32>,
    pub attn_res_proj: Vec<f32>,
    pub mlp_res_norm: Vec<f32>,
    pub mlp_res_proj: Vec<f32>,

    /// Kimi-K3 latent-MoE: RMSNorm applied in the `moe_latent` space, `[moe_latent]`.
    pub routed_expert_norm: Vec<f32>,
}
/// DeepSeek-V4's DSpark speculative head — the reference's "DSpark stages stored under the
/// `mtp.*` checkpoint namespace", at layer indices `n_layers .. n_layers + stages.len()`.
///
/// A stage IS a V4 block (attention + MoE + Hyper-Connections), so each one loads through
/// the ordinary layer loader. What makes it its own type is the extras: stage 0 projects
/// the concatenated hidden states of `cfg.dspark_targets` down to one hidden width, and the
/// LAST stage owns the vocabulary heads.
///
/// **Each stage carries its own 256-expert pool** — 10.3 GB of the head's ~11.2 GB. On a
/// box that is already short of full residency for the main model, loading this competes
/// with it. That is why it is opt-in.
pub struct DsparkHead {
    /// the stages in execution order; each a full V4 block
    pub stages: Vec<Layer>,
    /// stage 0: `[dim, dim * dspark_targets.len()]`, then `main_norm`
    pub main_proj: Option<QTensor>,
    pub main_norm: Vec<f32>,
    /// last stage: final norm before the vocabulary head
    pub norm: Vec<f32>,
    /// last stage: rank-`markov_rank` token embedding and its head, a per-position logit
    /// bias applied while sampling the drafted block
    pub markov_w1: Option<QTensor>,
    pub markov_w2: Option<QTensor>,
    /// last stage: `[1, dim + markov_rank]`, scores the drafted block. Kept f32 — it is
    /// one row, so quantising it buys nothing.
    pub confidence: Vec<f32>,
    /// last stage: its own Hyper-Connection head (plain sigmoid gate, no Sinkhorn)
    pub hc_head_fn: Vec<f32>,
    pub hc_head_base: Vec<f32>,
    pub hc_head_scale: f32,
}


/// The MTP (multi-token prediction) speculative head — port of the `mtpL` /
/// `eh_proj` / `enorm` / `hnorm` / `mtp_norm` members of the C `Model`.
///
/// Structurally it is a **normal sparse [`Layer`]** living at the extra layer
/// index `n_layers` (its routed experts stream like any other layer's), plus four
/// tensors that fuse the main model's hidden state with the next token's
/// embedding before that layer runs:
///
/// ```text
/// e  = rmsnorm(embed(next_tok), enorm)
/// h  = rmsnorm(rmsnorm(hidden, final_norm), hnorm)   // hidden is POST model.norm
/// hx = eh_proj · [e ; h]                             // [D, 2D] · [2D] -> [D]
/// hx = layer_forward(mtp_layer, hx, pos)
/// draft = argmax(lm_head · rmsnorm(hx, mtp_norm))
/// ```
///
/// The head is trained to predict token `t+2` from the state at `t` and the
/// embedding of `t+1`, which is what makes its drafts worth verifying.
///
/// # Why the block is a `Vec`
///
/// GLM's head is exactly one block, but Nemotron-H's is **two** sublayers
/// (`mtp_hybrid_override_pattern == "*E"`: a NoPE-GQA attention block then a latent-MoE
/// block), so `hx` runs through both before the final norm. Rather than bolt an
/// `Option<Layer>` onto the side — which would leave every consumer branching on "one or
/// two?" and would not extend to a three-sublayer head — the head owns an ordered
/// `Vec<MtpBlock>` and the forward path is a loop. GLM builds a one-element vec, so its
/// loop body runs exactly the calls it always did, in the same order, at the same layer
/// index; [`MtpHead::layer`] keeps the "the block" reading concise for that case.
pub struct MtpHead {
    /// The head's transformer block(s), in execution order, occupying layer indices
    /// `n_layers .. n_layers + blocks.len()`. Each carries its own KV row.
    pub blocks: Vec<MtpBlock>,
    /// `[D, 2D]` — projects the concatenated `[e ; h]` back to hidden width
    pub eh_proj: QTensor,
    /// RMSNorm weight applied to the next token's embedding
    pub enorm: Vec<f32>,
    /// RMSNorm weight applied to the (already final_norm'd) hidden state
    pub hnorm: Vec<f32>,
    /// `shared_head.norm.weight` — the head's own final norm before `lm_head`
    pub mtp_norm: Vec<f32>,
}

/// One sublayer of the MTP head: the weights, plus (on a hybrid arch) which mixer runs.
pub struct MtpBlock {
    /// the sublayer's resident weights, loaded exactly like a main-stack layer's
    pub layer: Layer,
    /// Which mixer this sublayer is, for hybrid architectures whose per-layer dispatch
    /// normally reads `cfg.layer_kind[li]`. **Must** be `Some` on Nemotron-H: the head
    /// lives at `li >= n_layers`, which is past the end of `layer_kind` (that vector is
    /// `num_hidden_layers` long by contract and describes the main stack only — see
    /// `Config::mtp_layer_kind`). `None` on GLM/M3, where the block shape is implied by
    /// the arch and nothing consults a kind.
    pub kind: Option<LayerKind>,
}

impl MtpHead {
    /// The head's first (on GLM/M3, only) block. Convenience for the single-block case
    /// and for tests; the forward path iterates [`MtpHead::blocks`] instead.
    pub fn layer(&self) -> &Layer {
        &self.blocks[0].layer
    }
}

/// The compressed MLA KV-cache — port of the `Lc`/`Rc` per-layer buffers in
/// `c/glm.c`.
///
/// Only the normalized latent `[kv_lora]` and the rotary key `[qk_rope]` are kept
/// per token (576 vs 32768 values/token for GLM-5.2); k_nope and value are
/// reconstructed on the fly via `kv_b`. This is what makes the context tractable
/// in ~10 GB (64 heads, no GQA).
pub struct KvCache {
    pub max_t: usize,
    kv_lora: usize,
    qk_rope: usize,
    /// per-layer latent buffer, each `[max_t * kv_lora]`
    latent: Vec<Vec<f32>>,
    /// per-layer rotary-key buffer, each `[max_t * qk_rope]`
    k_rot: Vec<Vec<f32>>,
    /// GQA full-KV width (`n_kv_heads * head_dim`); 0 on the MLA (GLM) path.
    kv_dim: usize,
    /// DeepSeek-V4 compressed KV: per-layer `[max_blocks * head_dim]`, empty on layers
    /// without a Compressor and on every other arch. See `comp_init`.
    comp: Vec<Vec<f32>>,
    comp_len: Vec<usize>,
    comp_state: Vec<Option<crate::dsv4::CompressorState>>,
    comp_dim: usize,
    /// DeepSeek-V4 **Indexer** compressed KV. Same shape as `comp` but at
    /// `index_head_dim` (128, not 512) and only on the layers whose `compress_ratio` is 4.
    /// The Indexer owns a SEPARATE Compressor constructed with `rotate=True`, so these
    /// rows are Hadamard-rotated and FP4-simulated — they are not a view of `comp`, and
    /// scoring against `comp` instead would silently compare against the wrong space.
    icomp: Vec<Vec<f32>>,
    icomp_len: Vec<usize>,
    icomp_state: Vec<Option<crate::dsv4::CompressorState>>,
    icomp_dim: usize,
    /// per-layer full key buffer, each `[max_t * kv_dim]` — GQA only (else empty).
    k_full: Vec<Vec<f32>>,
    /// per-layer full value buffer, each `[max_t * kv_dim]` — GQA only (else empty).
    v_full: Vec<Vec<f32>>,

    // ---- Nemotron-H Mamba2 recurrent state (per Mamba layer; empty otherwise) ----
    // Unlike the KV buffers above these are **fixed-size**, not `max_t`-scaled: a
    // selective-scan carries a bounded recurrent state, so its memory is O(1) in
    // context length. Populated by [`KvCache::enable_mamba2`]; other layer rows (and
    // every non-Nemotron cache) leave both empty.
    /// per-Mamba-layer causal-conv history, each `[d_conv * conv_dim]`, time-major:
    /// the last `d_conv` input columns of `hidden_B_C`, row `d_conv-1` = most recent.
    mamba_conv: Vec<Vec<f32>>,
    /// per-Mamba-layer selective-scan state `[n_heads, head_dim, d_state]`.
    mamba_ssm: Vec<SsmState>,
    /// conv-kernel width `d_conv` (rows per `mamba_conv` entry); 0 if unused.
    mamba_d_conv: usize,
    /// conv channel count `conv_dim` (cols per `mamba_conv` entry); 0 if unused.
    mamba_conv_dim: usize,

    // ---- Kimi-K3 KDA recurrent state (per KDA layer; empty otherwise) ----
    // Fixed-size like the Mamba2 state above and for the same reason: a delta rule
    // carries a bounded association matrix between steps, so its memory is O(1) in
    // context. Populated by [`KvCache::enable_kda`].
    /// per-KDA-layer causal-conv history, each `[d_conv, 3 * n_heads * head_dim]`
    /// (q, k and v each have their own `*_conv1d`), time-major.
    kda_conv: Vec<Vec<f32>>,
    /// per-KDA-layer delta-rule association matrix, each `[n_heads, head_dim, head_dim]`.
    kda_state: Vec<Vec<f32>>,

    /// Raw-KV **ring width**: when non-zero, `latent` and `k_rot` hold exactly this many
    /// physical rows and absolute position `p` lives at `p % ring`. `0` means "no ring" —
    /// row `p` is at index `p`, which is what every arch except DeepSeek-V4 does.
    ///
    /// V4's raw KV *is* a ring by construction: a query at `p` can only reach back
    /// `sliding_window` positions, and everything older is reachable solely through the
    /// Compressor. Retaining raw rows past that window is dead storage, and it was the
    /// term that made context cost 206.9 KB/token instead of the 13.4 KB the compressed
    /// rows actually need.
    ///
    /// Sized by [`KvCache::ring_ensure`] from the widest span a call will read, and never
    /// shrunk: every live row's address is `p % ring`, so changing `ring` re-homes all of
    /// them (which `ring_ensure` does explicitly, rather than leaving the mapping stale).
    ring: usize,
    /// Highest absolute position + 1 ever written to a ringed row. Only maintained while
    /// `ring != 0`; it is what tells [`KvCache::ring_ensure`] which rows are still live
    /// and therefore have to be carried across a resize.
    ring_hi: usize,

    /// first valid position per layer (MTP partial caches start mid-sequence)
    pub kv_start: Vec<usize>,
    /// device-side KV shadow (persistent-KV GPU decode path); lazily allocated
    #[cfg(feature = "cuda")]
    dev: Option<crate::gpu::DeviceKv>,
}

/// `kv_start` value meaning "this layer's cache has not started yet" — the MTP
/// row's state until the first draft establishes its first position.
///
/// The C uses `-1` in an `int` array and tests `kv_start[li] < 0 || kv_start[li] > p`.
/// `usize::MAX` collapses that to just `kv_start[li] > p`, since the sentinel is
/// greater than every real position — same semantics, no signed type needed.
pub const KV_UNSET: usize = usize::MAX;

/// `n_rows` independently-allocated zero buffers of `len` f32 each.
///
/// Deliberately NOT `vec![vec![0.0; len]; n_rows]`: that clones one buffer `n_rows`
/// times, and each clone memcpies — faulting in (committing) every page. Here each row
/// is a fresh `vec![0.0; len]`, which lowers to `alloc_zeroed`; on Linux that is a
/// zero-on-demand `mmap`, so the pages stay uncommitted until written. The KV cache is
/// then sized to `max_t` in *address space* but only resident for the tokens actually
/// produced — a request that stops early never commits the tail.
#[inline]
fn lazy_zeros(n_rows: usize, len: usize) -> Vec<Vec<f32>> {
    (0..n_rows).map(|_| vec![0.0f32; len]).collect()
}

/// Physical row index of absolute position `pos` in a raw-KV buffer of `ring` rows;
/// `ring == 0` means "not a ring", so the position indexes itself.
///
/// Free-standing because [`KvCache::ring_ensure`] has to apply it for TWO widths at once
/// (old and new) while resizing, and a second copy of this rule living there is exactly
/// how a resize ends up disagreeing with every read that follows it.
#[inline]
fn ring_slot(pos: usize, ring: usize) -> usize {
    if ring == 0 { pos } else { pos % ring }
}

impl KvCache {
    /// Allocate a cache for `n_rows` layer rows holding up to `max_t` tokens.
    ///
    /// Prefer [`KvCache::for_model`], which sizes the rows (including the MTP
    /// head's extra row) from the model itself.
    ///
    /// The row buffers are sized to `max_t` but **committed lazily** ([`lazy_zeros`]):
    /// a full-window cache is virtual address space, not resident RAM, until tokens are
    /// actually written. So a request's KV footprint grows with the tokens it produces,
    /// not with `max_t` — one that stops early never pays for the unused tail.
    pub fn new(n_rows: usize, kv_lora: usize, qk_rope: usize, max_t: usize) -> KvCache {
        KvCache {
            // DeepSeek-V4 compressed KV — empty until `comp_init`; every other arch
            // leaves them empty for the cache's lifetime.
            comp: Vec::new(),
            comp_len: Vec::new(),
            comp_state: Vec::new(),
            comp_dim: 0,
            icomp: Vec::new(),
            icomp_len: Vec::new(),
            icomp_state: Vec::new(),
            icomp_dim: 0,
            max_t,
            kv_lora,
            qk_rope,
            latent: lazy_zeros(n_rows, max_t * kv_lora),
            k_rot: lazy_zeros(n_rows, max_t * qk_rope),
            kv_dim: 0,
            k_full: vec![Vec::new(); n_rows],
            v_full: vec![Vec::new(); n_rows],
            mamba_conv: vec![Vec::new(); n_rows],
            mamba_ssm: (0..n_rows).map(|_| SsmState::zeros(0, 0, 0)).collect(),
            mamba_d_conv: 0,
            mamba_conv_dim: 0,
            kda_conv: vec![Vec::new(); n_rows],
            kda_state: vec![Vec::new(); n_rows],
            ring: 0,
            ring_hi: 0,
            kv_start: vec![0; n_rows],
            #[cfg(feature = "cuda")]
            dev: None,
        }
    }

    /// Enable the GQA full-KV cache (MiniMax-M3): allocate per-layer key/value
    /// buffers of width `kv_dim = n_kv_heads * head_dim`. No-op for the MLA path.
    /// Lazily committed like the rest of the cache (see [`KvCache::new`]).
    pub(crate) fn enable_gqa(&mut self, kv_dim: usize) {
        self.kv_dim = kv_dim;
        let rows = self.k_full.len();
        self.k_full = lazy_zeros(rows, self.max_t * kv_dim);
        self.v_full = lazy_zeros(rows, self.max_t * kv_dim);
    }

    /// Enable the Nemotron-H Mamba2 recurrent state: for each `LayerKind::Mamba`
    /// layer, allocate a fixed-size conv history (`[d_conv, conv_dim]`, time-major)
    /// and selective-scan state (`[n_heads, head_dim, d_state]`), both zeroed. Non-Mamba
    /// rows (attention/MoE) keep empty buffers. These are O(1) in context length, unlike
    /// the `max_t`-scaled KV rows, so they are allocated eagerly (small + always resident).
    pub(crate) fn enable_mamba2(&mut self, cfg: &Config) {
        let conv_dim =
            cfg.mamba_inter as usize + 2 * cfg.mamba_n_groups as usize * cfg.mamba_d_state as usize;
        let k = cfg.mamba_d_conv as usize;
        let (nh, hd, ds) = (
            cfg.mamba_n_heads as usize,
            cfg.mamba_head_dim as usize,
            cfg.mamba_d_state as usize,
        );
        self.mamba_d_conv = k;
        self.mamba_conv_dim = conv_dim;
        let rows = self.mamba_conv.len();
        for li in 0..rows {
            // Only the actual Mamba layers carry state; index by layer so the mixer can
            // address `mamba_*(li)` directly. The MTP row (if any) is never Mamba.
            if cfg.layer_kind.get(li).copied() == Some(LayerKind::Mamba) {
                self.mamba_conv[li] = vec![0.0f32; k * conv_dim];
                self.mamba_ssm[li] = SsmState::zeros(nh, hd, ds);
            }
        }
    }

    /// This layer's KDA conv history: `3 * (d_conv - 1) * c` floats laid out
    /// `[stream][token][channel]` for q, k, v (oldest token first). Zeros for a fresh
    /// cache or a non-KDA layer, which is exactly the "convolve against nothing"
    /// boundary a prefill wants.
    pub(crate) fn kda_conv_take(&self, layer: usize, c: usize, k: usize) -> Vec<f32> {
        let need = 3 * k.saturating_sub(1) * c;
        match self.kda_conv.get(layer) {
            Some(v) if v.len() >= need && need > 0 => v[..need].to_vec(),
            _ => vec![0.0f32; need],
        }
    }

    /// Replace this layer's KDA conv history. `carries` is what
    /// [`KvCache::kda_conv_take`] returns, advanced by one step.
    pub(crate) fn kda_conv_store(&mut self, layer: usize, carries: &[f32]) {
        let Some(slot) = self.kda_conv.get_mut(layer) else {
            return;
        };
        if slot.len() < carries.len() {
            slot.resize(carries.len(), 0.0);
        }
        slot[..carries.len()].copy_from_slice(carries);
    }

    /// This layer's delta-rule association matrix, `[h, dk, dk]` flattened K-major.
    /// Allocated by [`KvCache::enable_kda`]; empty on a non-KDA layer.
    pub(crate) fn kda_state_mut(&mut self, layer: usize) -> &mut [f32] {
        &mut self.kda_state[layer]
    }

    /// Element counts of one KDA layer's two fixed-size buffers, as f32 counts:
    /// `(conv_history, delta_rule_state)`.
    ///
    /// The single source of truth for both [`KvCache::enable_kda`], which allocates
    /// them, and [`KvCache::fixed_bytes`], which charges for them. Every past KV
    /// accounting bug in this file was a second copy of a shape drifting from the
    /// allocation, so there is deliberately only one copy of this one.
    fn kda_state_lens(cfg: &Config) -> (usize, usize) {
        let (nh, hd) = (cfg.kda_n_heads as usize, cfg.kda_head_dim as usize);
        // q, k and v each carry their own `*_conv1d` over `nh * hd` channels.
        let conv = cfg.kda_d_conv as usize * 3 * nh * hd;
        // A delta rule keeps one `[head_dim, head_dim]` association matrix per head.
        let state = nh * hd * hd;
        (conv, state)
    }

    /// Enable the Kimi-K3 KDA recurrent state: for each [`LayerKind::Kda`] layer,
    /// allocate the short causal-conv history and the per-head delta-rule matrix, both
    /// zeroed. Gated-MLA rows keep empty buffers here and use the KV rows instead.
    ///
    /// O(1) in context length, like the Mamba2 state — so these are allocated eagerly
    /// and charged to [`KvCache::fixed_bytes`], never to the per-token figure. K3
    /// carries ~475 MB of this per *sequence* across its 69 KDA layers; a reservation
    /// counting only per-token bytes under-commits by that much for every concurrent
    /// sequence, and the shortfall is worst for SHORT requests, where the per-token
    /// term is too small to accidentally cover it.
    pub(crate) fn enable_kda(&mut self, cfg: &Config) {
        let (conv, state) = Self::kda_state_lens(cfg);
        let rows = self.kda_conv.len();
        for li in 0..rows {
            // Index by layer so the mixer can address `kda_*(li)` directly. The MTP row
            // (if any) is past the end of `layer_kind` and is never KDA.
            if cfg.layer_kind.get(li).copied() == Some(LayerKind::Kda) {
                self.kda_conv[li] = vec![0.0f32; conv];
                self.kda_state[li] = vec![0.0f32; state];
            }
        }
    }

    /// Allocate a cache sized for `model`, holding up to `max_t` tokens.
    ///
    /// When the model carries an MTP head this allocates one extra row **per head
    /// sublayer** (C: `NR = c->n_layers + 1`, which assumed GLM's single block; a
    /// Nemotron-H head is two sublayers, so it needs two). Each is a real layer at index
    /// `n_layers + j` with its own KV. Those rows start [`KV_UNSET`] rather than 0 (C:
    /// `kv_start[i] = -1`): unlike the main stack, the head's cache begins at the
    /// first *decode* position, not at the start of the prompt, so it holds only a
    /// partial suffix of the sequence.
    pub fn for_model(model: &Model, max_t: usize) -> KvCache {
        let n_layers = model.cfg.n_layers as usize;
        let head_rows = model.mtp.as_ref().map_or(0, |m| m.blocks.len());
        let rows = n_layers + head_rows;
        let mut kv = KvCache::new(
            rows,
            model.cfg.kv_lora as usize,
            model.cfg.qk_rope as usize,
            max_t,
        );
        for r in n_layers..n_layers + head_rows {
            kv.kv_start[r] = KV_UNSET;
        }
        // One predicate for "has GQA full-KV", shared with `bytes_per_token` — if these
        // two ever disagree the reservation silently mis-sizes (see `allocates_gqa_kv`).
        // Nemotron-H is hybrid: its 8 attention layers need the GQA full-KV cache AND its
        // 40 Mamba layers need recurrent conv+ssm state. `enable_gqa` allocates KV rows for
        // every layer (only attention rows are ever written — the rest stay lazily
        // uncommitted), and `enable_mamba2` allocates state on just the Mamba rows.
        if Self::allocates_gqa_kv(&model.cfg) {
            kv.enable_gqa(model.cfg.n_kv_heads as usize * model.cfg.qk_head as usize);
        }
        if model.cfg.arch == Arch::NemotronH {
            kv.enable_mamba2(&model.cfg);
        }
        // Kimi-K3 is hybrid on the mixer axis: its 24 gated-MLA layers use the latent
        // KV rows allocated above, its 69 KDA layers a fixed-size recurrent state.
        if model.cfg.arch == Arch::KimiK3 {
            kv.enable_kda(&model.cfg);
        }
        kv
    }

    /// Does [`KvCache::for_model`] give this architecture the GQA full-KV buffers?
    ///
    /// Deliberately NOT `cfg.arch.is_gqa()`: Nemotron-H is excluded from `is_gqa()`
    /// (its hybrid stack is not a GQA transformer) yet `for_model` still calls
    /// `enable_gqa` for its 8 attention layers. Callers that size or reserve memory
    /// must ask this, not `is_gqa()`, or they silently omit `k_full`/`v_full` — which
    /// is exactly how the serve reservation under-counted the largest KV term.
    fn allocates_gqa_kv(cfg: &Config) -> bool {
        cfg.arch.is_gqa() || cfg.arch == Arch::NemotronH
    }

    /// How many layers actually hold KV. For a hybrid stack only the attention layers
    /// do (Nemotron-H: 8 of 88; Kimi-K3: 24 of 93, the gated-MLA layers) — `for_model`
    /// allocates rows for every layer, but the non-attention rows are never written and
    /// stay lazily uncommitted, so they cost no physical memory and must not be charged
    /// for.
    ///
    /// `pub` so the capacity planner reports the same number the reservation uses; it
    /// used to keep its own copy of this predicate, which is exactly how the earlier
    /// accounting bugs got in.
    pub fn kv_layers(cfg: &Config) -> usize {
        if cfg.layer_kind.is_empty() {
            cfg.n_layers as usize
        } else {
            cfg.layer_kind
                .iter()
                .filter(|k| **k == LayerKind::Attn)
                .count()
        }
    }

    /// Resident KV bytes per token — host cache **plus** the CUDA device shadow.
    ///
    /// Per KV-holding layer: latent (`kv_lora`) + roped key (`qk_rope`), and, when the
    /// GQA buffers are allocated, the full K and V at `kv_dim = n_kv_heads * qk_head`
    /// each. The device shadow (`DeviceKv`, CUDA only) mirrors **only** latent + rope —
    /// `k_full`/`v_full` are read from host over GB10's unified memory — so it doubles
    /// just the MLA-style terms.
    ///
    /// This lives beside [`KvCache::for_model`] on purpose: it is the accounting twin of
    /// the allocation, and the two drifting apart is what produced every past error here.
    /// Three fixed so far: the original omitted GQA `k_full`/`v_full` (~17× undercount);
    /// the interim fix doubled the *whole* host figure, over-counting GQA ~2×; and the
    /// hybrid case charged all 88 Nemotron layers while omitting its `k_full`/`v_full`
    /// (a net ~3.7× over-count). Excludes [`KvCache::fixed_bytes`], which is per-sequence.
    /// **Worst case, and deliberately so.** It is the cost of a sequence that is all
    /// prompt. On a ringed arch (DeepSeek-V4) the raw rows stop accumulating at the ring
    /// width, so a sequence that GENERATES most of its tokens costs far less — but the
    /// consumers of this figure (`context_in_kv_budget`, serve's `ctx` planning) ask "how
    /// long a context fits", and the honest answer to that is the all-prompt one. Lowering
    /// it here would let a long prompt be admitted into memory that cannot hold it.
    /// [`Self::bytes_for_split`] is the form that knows the prompt/generation split and
    /// can charge the ring for what it really costs.
    pub fn bytes_per_token(cfg: &Config) -> usize {
        Self::raw_row_bytes(cfg) + Self::compressed_bytes_per_token(cfg)
    }

    /// Bytes for ONE raw KV row across every KV-holding layer: latent (`kv_lora`) + roped
    /// key (`qk_rope`), the GQA full K and V when those buffers exist, and the CUDA device
    /// shadow, which mirrors **only** latent + rope (`k_full`/`v_full` are read from host
    /// over GB10's unified memory) and so doubles just the MLA-style terms.
    ///
    /// Split out from [`Self::bytes_per_token`] because a ring makes "per row" and "per
    /// token" different questions: V4 retains a bounded number of rows however many tokens
    /// go past.
    pub fn raw_row_bytes(cfg: &Config) -> usize {
        // `latent` needs no predicate: `kv_lora` is already 0 on every arch that does not
        // keep one, and it is the ONE row DeepSeek-V4 writes.
        let latent = cfg.kv_lora as usize;
        // `k_rot` and the device shadow do: both exist only on the MLA reader.
        let mla = if Self::mirrors_kv_on_device(cfg) {
            cfg.qk_rope as usize
        } else {
            0
        };
        let gqa_full = if Self::allocates_gqa_kv(cfg) {
            2 * cfg.n_kv_heads as usize * cfg.qk_head as usize // k_full + v_full: host only
        } else {
            0
        };
        let device_shadow = if cfg!(feature = "cuda") && Self::mirrors_kv_on_device(cfg) {
            latent + mla
        } else {
            0
        };
        Self::kv_layers(cfg) * (latent + mla + gqa_full + device_shadow) * 4
    }

    /// Does this architecture keep the MLA-style `latent` + `k_rot` rows on the **device**
    /// as well as the host — the [`DeviceKv`](crate::gpu::DeviceKv) shadow that
    /// [`Self::sync_device`] uploads?
    ///
    /// Only the weight-absorption reader (`attention_with_heads`, i.e. GLM-5.2 and Kimi-K3's
    /// gated-MLA layers) syncs; it is the sole caller. Two architectures were being charged
    /// for a shadow they never allocate, and for the host `k_rot` rows behind it:
    ///
    /// * **GQA** (MiniMax-M3/M2.7, Nemotron-H's attention layers) — `attention_gqa` ropes
    ///   the key in place and stores the whole thing in `k_full`/`v_full`. It never touches
    ///   `latent` or `k_rot`, so both buffers stay lazily uncommitted and cost nothing. Worth
    ///   ~12% of their per-token figure.
    /// * **DeepSeek-V4** — `dsv4_attention` builds its key block on the host and hands it
    ///   straight to the sparse kernel. It writes `latent` and nothing else. Worth ~1.9x on
    ///   its raw term, the single largest error this accounting has carried.
    ///
    /// **A deny-list on purpose.** An allow-list (`matches!(arch, Glm | KimiK3)`) would
    /// silently *under*-charge a future MLA architecture, and under-charging KV is how a
    /// request gets admitted into memory that cannot hold it. Charging is the safe default,
    /// so a new arch has to opt out here deliberately.
    ///
    /// The `window` clause is mechanism, not a proxy for "is V4": a ringed raw tier
    /// **cannot** carry this shadow, because the shadow is absolute-indexed and uploads
    /// incrementally — which is exactly what `sync_device` asserts.
    fn mirrors_kv_on_device(cfg: &Config) -> bool {
        !Self::allocates_gqa_kv(cfg) && cfg.window == 0
    }

    /// How many raw KV rows a sequence of `n_prompt` prompt tokens and `n_new` generated
    /// ones keeps resident.
    ///
    /// Without a ring that is every position, so both terms count. With one
    /// (`cfg.window > 0`, DeepSeek-V4) it is `max(window, n_prompt)` and **`n_new` does
    /// not appear**: prefill is the one call that reads all of its own rows, and every
    /// decode step after it reads only the window, so the ring never grows again. That
    /// asymmetry is the shape of the whole feature — generation became free, prompts did
    /// not, and they stay expensive until prefill is chunked.
    pub fn raw_rows(cfg: &Config, n_prompt: usize, n_new: usize) -> usize {
        if cfg.window > 0 {
            n_prompt.max(cfg.window as usize)
        } else {
            n_prompt.saturating_add(n_new)
        }
    }

    /// DeepSeek-V4's compressed KV, per token.
    ///
    /// A compressor layer emits one `qk_head`-wide row every `compress_ratio` tokens, so
    /// it costs `qk_head / ratio` floats per token — 128 floats on a ratio-4 layer, 4 on a
    /// ratio-128 one. Layers with an Indexer (`ratio == 4`) carry a SECOND set at
    /// `index_hd`. Zero on every other arch, where `compress_ratios` is empty.
    ///
    /// This was missing entirely: the compressed rows are the only reason V4 has context
    /// past its 128-token window, so leaving them out of the reservation under-commits by
    /// more the longer the sequence — exactly backwards.
    fn compressed_bytes_per_token(cfg: &Config) -> usize {
        let n = cfg.layer_kind.len().max(cfg.n_layers as usize);
        cfg.compress_ratios
            .iter()
            .take(n)
            .filter(|&&r| r > 0)
            .map(|&r| {
                let r = r as usize;
                let main = cfg.qk_head as usize / r;
                let idx = if r == 4 { cfg.index_hd as usize / r } else { 0 };
                (main + idx) * 4
            })
            .sum()
    }

    /// Per-sequence KV bytes that do **not** scale with context length: the recurrent
    /// state carried by every non-attention mixer in a hybrid stack.
    ///
    /// Two contributors, one per hybrid arch:
    /// - **Mamba2** ([`LayerKind::Mamba`], Nemotron-H): conv history + selective-scan
    ///   state, ~174 MB/sequence across its 40 Mamba layers.
    /// - **KDA** ([`LayerKind::Kda`], Kimi-K3): conv history + the per-head delta-rule
    ///   association matrix, ~475 MB/sequence across its 69 KDA layers — dominated by
    ///   the `[n_heads, head_dim, head_dim]` matrix.
    ///
    /// O(1) in context, but far from free. A reservation that counts only per-token
    /// bytes under-commits by this much for every concurrent sequence, and the shortfall
    /// is worst for SHORT requests, where the per-token term is too small to accidentally
    /// cover it. The shapes come from [`KvCache::kda_state_lens`] and `enable_mamba2`'s
    /// own arithmetic, so this cannot drift from what is actually allocated.
    pub fn fixed_bytes(cfg: &Config) -> usize {
        if cfg.layer_kind.is_empty() {
            return 0;
        }
        let count = |want: LayerKind| cfg.layer_kind.iter().filter(|k| **k == want).count();
        // Mamba2 (Nemotron-H). Zero elsewhere: no `Mamba` layers and zeroed dims.
        let n_mamba = count(LayerKind::Mamba);
        let conv_dim =
            cfg.mamba_inter as usize + 2 * cfg.mamba_n_groups as usize * cfg.mamba_d_state as usize;
        let conv = cfg.mamba_d_conv as usize * conv_dim;
        let ssm =
            cfg.mamba_n_heads as usize * cfg.mamba_head_dim as usize * cfg.mamba_d_state as usize;
        // KDA (Kimi-K3). Same: zero unless the stack actually has `Kda` layers.
        let n_kda = count(LayerKind::Kda);
        let (kda_conv, kda_state) = Self::kda_state_lens(cfg);
        (n_mamba * (conv + ssm) + n_kda * (kda_conv + kda_state)) * 4
    }

    /// Total resident bytes for a sequence of `n_tokens`: the per-token KV plus the
    /// fixed per-sequence state. This is what a reservation should ask for.
    pub fn bytes_for(cfg: &Config, n_tokens: usize) -> usize {
        // All prompt: the worst split, and the only safe reading when the caller has not
        // said how the tokens divide. Identical to `bytes_per_token * n + fixed` on every
        // arch, ringed or not — `raw_rows(cfg, n, 0)` is `n` for a prompt at least a
        // window long, and rounds UP to the window below that.
        Self::bytes_for_split(cfg, n_tokens, 0)
    }

    /// [`Self::bytes_for`] for a caller that knows the prompt/generation split.
    ///
    /// The distinction only bites on a ringed arch, where it is the difference between
    /// charging a 40k-token generation for 40k raw rows and charging it for the window's
    /// worth it actually holds. Over-charging here is not free: `gen` reports this figure
    /// as `Class::Kv`, and every byte of it is subtracted from the expert cache's budget.
    pub fn bytes_for_split(cfg: &Config, n_prompt: usize, n_new: usize) -> usize {
        let rows = Self::raw_rows(cfg, n_prompt, n_new);
        Self::raw_row_bytes(cfg)
            .saturating_mul(rows)
            .saturating_add(
                Self::compressed_bytes_per_token(cfg)
                    .saturating_mul(n_prompt.saturating_add(n_new)),
            )
            .saturating_add(Self::fixed_bytes(cfg))
    }

    /// Record that the layer's cache covers positions from `pos` onward, if that
    /// is earlier than what it already covers. Port of the C's
    /// `if(kv_start[li] < 0 || kv_start[li] > p) kv_start[li] = p;` — the
    /// [`KV_UNSET`] sentinel makes the `< 0` arm unnecessary.
    pub fn start_at(&mut self, layer: usize, pos: usize) {
        if self.kv_start[layer] > pos {
            self.kv_start[layer] = pos;
        }
    }

    /// Sync the device KV shadow for `layer` up to `tk` rows and return the
    /// device `(latent, rope)` base pointers. Uploads only the missing rows.
    #[cfg(feature = "cuda")]
    pub fn sync_device(
        &mut self,
        layer: usize,
        pos_base: usize,
        tk: usize,
    ) -> Option<(*const f32, *const f32)> {
        let n_layers = self.latent.len();
        // The device shadow mirrors the host rows at ABSOLUTE positions and uploads
        // `pos_base..tk` incrementally, so it cannot read a ringed buffer. Only the MLA
        // path syncs, and it never rings — assert rather than leave the two layouts to
        // disagree silently if that ever changes.
        assert_eq!(self.ring, 0, "sync_device on a ringed cache: the shadow is absolute-indexed");
        let (max_t, kvl, r) = (self.max_t, self.kv_lora, self.qk_rope);
        let dev = self
            .dev
            .get_or_insert_with(|| crate::gpu::DeviceKv::new(n_layers, max_t));
        dev.sync(
            layer,
            &self.latent[layer],
            &self.k_rot[layer],
            kvl,
            r,
            pos_base,
            tk,
        )
    }

    pub fn kv_lora(&self) -> usize {
        self.kv_lora
    }
    pub fn qk_rope(&self) -> usize {
        self.qk_rope
    }

    // ---- raw-KV ring (DeepSeek-V4) ----------------------------------------
    //
    // See the `ring` field. Everything below is a no-op while `ring == 0`, which is the
    // state every other architecture stays in for the cache's whole lifetime.

    /// The ring width, or 0 when the raw rows are indexed by absolute position.
    pub fn ring(&self) -> usize {
        self.ring
    }

    /// Physical index of absolute position `pos` in the raw row buffers.
    #[inline]
    fn slot(&self, pos: usize) -> usize {
        ring_slot(pos, self.ring)
    }

    /// Guarantee the raw rows can hold `rows` consecutive positions at once, turning
    /// them into a ring the first time this is called.
    ///
    /// `rows` is the widest span a *single* call will read back — for V4 that is
    /// `total - raw_lo`, computed by the one expression the reader also uses, so the two
    /// cannot drift. Under-sizing here does not fault: it silently returns a row written
    /// by some other position, which is why the caller derives `rows` rather than
    /// guessing it.
    ///
    /// Growing is supported because it must be: `p % ring` is a different address for a
    /// different `ring`, so a resize that left the existing rows where they were would
    /// quietly re-point every live position. The live window is the last `ring` positions
    /// below `ring_hi`; they are re-homed into the new buffer before it is installed.
    /// Never shrinks — the win comes from the first sizing, and a shrink would be all
    /// risk for no memory (the pages are already committed).
    pub fn ring_ensure(&mut self, rows: usize) {
        if rows == 0 || rows <= self.ring {
            return;
        }
        let (kvl, r) = (self.kv_lora, self.qk_rope);
        let n_rows = self.latent.len();
        let old = self.ring;
        // Say once, out loud, that the ring engaged. Its success case is "tokens
        // unchanged" — identical output is equally consistent with the ring working and
        // with it never being reached — so an A/B alone cannot tell the two apart. The
        // Indexer needed the same line for the same reason.
        eprintln!(
            "[kv] raw ring {old} -> {rows} rows x {n_rows} layers ({:.1} MB); \
             generated tokens add none",
            (n_rows * rows * (kvl + r) * 4) as f64 / (1024.0 * 1024.0),
        );
        if old == 0 {
            // First call: the buffers still hold nothing (V4 sizes the ring before its
            // first write), so there is no live data to carry — just re-allocate small.
            // Deliberately re-allocating rather than truncating: `Vec::truncate` keeps the
            // `max_t`-sized allocation, and shedding that address space is the point.
            debug_assert_eq!(self.ring_hi, 0, "ring_ensure must size the ring before the first write");
            self.latent = lazy_zeros(n_rows, rows * kvl);
            self.k_rot = lazy_zeros(n_rows, rows * r);
            self.ring = rows;
            return;
        }
        let lo = self.ring_hi.saturating_sub(old);
        let mut latent = lazy_zeros(n_rows, rows * kvl);
        let mut k_rot = lazy_zeros(n_rows, rows * r);
        for li in 0..n_rows {
            for p in lo..self.ring_hi {
                let (from, to) = (ring_slot(p, old), ring_slot(p, rows));
                if kvl > 0 {
                    let src = &self.latent[li][from * kvl..(from + 1) * kvl];
                    latent[li][to * kvl..(to + 1) * kvl].copy_from_slice(src);
                }
                if r > 0 {
                    let src = &self.k_rot[li][from * r..(from + 1) * r];
                    k_rot[li][to * r..(to + 1) * r].copy_from_slice(src);
                }
            }
        }
        self.latent = latent;
        self.k_rot = k_rot;
        self.ring = rows;
    }

    /// Record that `pos` now holds live data, so a later [`Self::ring_ensure`] carries it.
    #[inline]
    fn ring_touch(&mut self, pos: usize) {
        if self.ring != 0 && pos + 1 > self.ring_hi {
            self.ring_hi = pos + 1;
        }
    }

    /// Append raw rows `[start, end)` of `layer` to `out`.
    ///
    /// The read side of the ring: a span that wraps is two runs, not one, so this cannot
    /// be a `&[f32]` the way [`Self::latent_rows`] is. Callers on a ringed cache must use
    /// this — `latent_rows` asserts it is not ringed rather than returning a slice whose
    /// rows belong to the wrong positions.
    pub fn extend_latent_rows(&self, layer: usize, start: usize, end: usize, out: &mut Vec<f32>) {
        self.extend_rows(&self.latent[layer], self.kv_lora, start, end, out);
    }

    /// [`Self::extend_latent_rows`] for the roped-key rows.
    pub fn extend_krot_rows(&self, layer: usize, start: usize, end: usize, out: &mut Vec<f32>) {
        self.extend_rows(&self.k_rot[layer], self.qk_rope, start, end, out);
    }

    fn extend_rows(&self, buf: &[f32], w: usize, start: usize, end: usize, out: &mut Vec<f32>) {
        if self.ring == 0 {
            out.extend_from_slice(&buf[start * w..end * w]);
            return;
        }
        assert!(
            end - start <= self.ring,
            "ring holds {} rows but {}..{} was asked for — ring_ensure was not given this span",
            self.ring,
            start,
            end,
        );
        let lo = self.slot(start);
        let n = end - start;
        // One run if the span ends before the wrap, two if it straddles it.
        let head = n.min(self.ring - lo);
        out.extend_from_slice(&buf[lo * w..(lo + head) * w]);
        if head < n {
            out.extend_from_slice(&buf[..(n - head) * w]);
        }
    }

    /// Normalized-latent row for `(layer, pos)`.
    pub fn latent_row(&self, layer: usize, pos: usize) -> &[f32] {
        let p = self.slot(pos);
        &self.latent[layer][p * self.kv_lora..(p + 1) * self.kv_lora]
    }
    pub fn latent_row_mut(&mut self, layer: usize, pos: usize) -> &mut [f32] {
        self.ring_touch(pos);
        let p = self.slot(pos);
        &mut self.latent[layer][p * self.kv_lora..(p + 1) * self.kv_lora]
    }

    /// DeepSeek-V4 compressed-KV rows and the per-layer Compressor carry state.
    ///
    /// Lazily sized: only the 41 layers that HAVE a compressor get buffers, and a model
    /// without one keeps these empty. `comp_len[layer]` is how many compressed rows exist
    /// so far — it advances once every `compress_ratio` tokens, not once per token, which
    /// is the whole point of the mechanism.
    pub fn comp_init(&mut self, n_layers: usize, ratios: &[i32], head_dim: usize, max_blocks: usize) {
        self.comp = (0..n_layers)
            .map(|i| {
                let r = ratios.get(i).copied().unwrap_or(0);
                if r > 0 { vec![0f32; max_blocks * head_dim] } else { Vec::new() }
            })
            .collect();
        self.comp_len = vec![0usize; n_layers];
        self.comp_state = (0..n_layers)
            .map(|i| {
                let r = ratios.get(i).copied().unwrap_or(0);
                (r > 0).then(|| crate::dsv4::CompressorState::new(r as usize, head_dim))
            })
            .collect();
        self.comp_dim = head_dim;
    }

    /// Append one compressed row to `layer`. Silently drops past capacity rather than
    /// panicking mid-generation — the caller sizes `max_blocks` from `max_t / min_ratio`.
    pub fn comp_push(&mut self, layer: usize, row: &[f32]) {
        let d = self.comp_dim;
        let n = self.comp_len[layer];
        if (n + 1) * d <= self.comp[layer].len() {
            self.comp[layer][n * d..(n + 1) * d].copy_from_slice(row);
            self.comp_len[layer] = n + 1;
        }
    }

    /// The compressed rows written so far for `layer`. Empty when the Compressor is off
    /// or the layer has none — returning a slice rather than indexing, because the caller
    /// asks unconditionally and `comp` is only allocated when `comp_init` runs.
    pub fn comp_rows(&self, layer: usize) -> &[f32] {
        match self.comp.get(layer) {
            Some(v) => &v[..self.comp_len[layer] * self.comp_dim],
            None => &[],
        }
    }

    /// Whether `comp_init` has run (buffers allocated).
    pub fn comp_rows_len(&self) -> usize {
        self.comp.len()
    }

    /// Mutable access to a layer's Compressor carry state.
    pub fn comp_state_mut(&mut self, layer: usize) -> Option<&mut crate::dsv4::CompressorState> {
        self.comp_state.get_mut(layer).and_then(|o| o.as_mut())
    }

    /// Allocate the Indexer's compressed KV — only on layers where `ratios[i] == want`
    /// (V4 builds an Indexer exactly when `compress_ratio == 4`), at `index_head_dim`.
    pub fn icomp_init(
        &mut self,
        n_layers: usize,
        ratios: &[i32],
        want: i32,
        head_dim: usize,
        max_blocks: usize,
    ) {
        let has = |i: usize| ratios.get(i).copied().unwrap_or(0) == want;
        self.icomp = (0..n_layers)
            .map(|i| if has(i) { vec![0f32; max_blocks * head_dim] } else { Vec::new() })
            .collect();
        self.icomp_len = vec![0usize; n_layers];
        self.icomp_state = (0..n_layers)
            .map(|i| has(i).then(|| crate::dsv4::CompressorState::new(want as usize, head_dim)))
            .collect();
        self.icomp_dim = head_dim;
    }

    /// Append one Indexer compressed row. Drops past capacity, like [`Self::comp_push`].
    pub fn icomp_push(&mut self, layer: usize, row: &[f32]) {
        let d = self.icomp_dim;
        let n = self.icomp_len[layer];
        if (n + 1) * d <= self.icomp[layer].len() {
            self.icomp[layer][n * d..(n + 1) * d].copy_from_slice(row);
            self.icomp_len[layer] = n + 1;
        }
    }

    /// The Indexer compressed rows written so far for `layer`; empty when it has none.
    pub fn icomp_rows(&self, layer: usize) -> &[f32] {
        match self.icomp.get(layer) {
            Some(v) => &v[..self.icomp_len[layer] * self.icomp_dim],
            None => &[],
        }
    }

    /// Whether [`Self::icomp_init`] has run.
    pub fn icomp_ready(&self) -> bool {
        !self.icomp.is_empty()
    }

    /// Mutable access to a layer's Indexer-Compressor carry state.
    pub fn icomp_state_mut(&mut self, layer: usize) -> Option<&mut crate::dsv4::CompressorState> {
        self.icomp_state.get_mut(layer).and_then(|o| o.as_mut())
    }

    /// Roped k_rot row for `(layer, pos)`.
    pub fn krot_row(&self, layer: usize, pos: usize) -> &[f32] {
        let p = self.slot(pos);
        &self.k_rot[layer][p * self.qk_rope..(p + 1) * self.qk_rope]
    }
    pub fn krot_row_mut(&mut self, layer: usize, pos: usize) -> &mut [f32] {
        self.ring_touch(pos);
        let p = self.slot(pos);
        &mut self.k_rot[layer][p * self.qk_rope..(p + 1) * self.qk_rope]
    }

    /// Contiguous latent rows `[start, end)` for a layer — a single slice the
    /// batched `kv_b` reconstruction multiplies against.
    ///
    /// Asserts the cache is not ringed. On a ring the rows are contiguous only until the
    /// wrap, and `[start*w..end*w]` would then be in bounds but hold *other positions'*
    /// data — a wrong answer, not a panic. Ringed callers use
    /// [`Self::extend_latent_rows`]; today only the MLA path (which never rings) is here.
    pub fn latent_rows(&self, layer: usize, start: usize, end: usize) -> &[f32] {
        assert_eq!(self.ring, 0, "latent_rows on a ringed cache — use extend_latent_rows");
        &self.latent[layer][start * self.kv_lora..end * self.kv_lora]
    }

    /// Contiguous roped-key rows `[start, end)` for a layer. Not ring-safe; see
    /// [`Self::latent_rows`].
    pub fn krot_rows(&self, layer: usize, start: usize, end: usize) -> &[f32] {
        assert_eq!(self.ring, 0, "krot_rows on a ringed cache — use extend_krot_rows");
        &self.k_rot[layer][start * self.qk_rope..end * self.qk_rope]
    }

    // ---- GQA full-KV accessors (MiniMax-M3) ----
    /// GQA full-KV width (`n_kv_heads * head_dim`), 0 on the MLA path.
    pub fn kv_dim(&self) -> usize {
        self.kv_dim
    }
    /// Writable full-key row for `(layer, pos)` (`[kv_dim]`).
    pub fn k_full_row_mut(&mut self, layer: usize, pos: usize) -> &mut [f32] {
        &mut self.k_full[layer][pos * self.kv_dim..(pos + 1) * self.kv_dim]
    }
    /// Writable full-value row for `(layer, pos)` (`[kv_dim]`).
    pub fn v_full_row_mut(&mut self, layer: usize, pos: usize) -> &mut [f32] {
        &mut self.v_full[layer][pos * self.kv_dim..(pos + 1) * self.kv_dim]
    }
    /// Contiguous full-key rows `[start, end)` for a layer.
    pub fn k_full_rows(&self, layer: usize, start: usize, end: usize) -> &[f32] {
        &self.k_full[layer][start * self.kv_dim..end * self.kv_dim]
    }
    /// Contiguous full-value rows `[start, end)` for a layer.
    pub fn v_full_rows(&self, layer: usize, start: usize, end: usize) -> &[f32] {
        &self.v_full[layer][start * self.kv_dim..end * self.kv_dim]
    }

    // ---- Nemotron-H Mamba2 state accessors ----
    /// Conv-kernel width `d_conv` (rows in a `mamba_conv` entry); 0 if unused.
    pub fn mamba_d_conv(&self) -> usize {
        self.mamba_d_conv
    }
    /// Conv channel count `conv_dim` (cols in a `mamba_conv` entry); 0 if unused.
    pub fn mamba_conv_dim(&self) -> usize {
        self.mamba_conv_dim
    }
    /// The layer's `[d_conv, conv_dim]` time-major conv history (read).
    pub fn mamba_conv_row(&self, layer: usize) -> &[f32] {
        &self.mamba_conv[layer]
    }
    /// The layer's `[d_conv, conv_dim]` time-major conv history (write).
    pub fn mamba_conv_row_mut(&mut self, layer: usize) -> &mut [f32] {
        &mut self.mamba_conv[layer]
    }
    /// The layer's selective-scan state, carried and updated across steps.
    pub fn mamba_ssm_mut(&mut self, layer: usize) -> &mut SsmState {
        &mut self.mamba_ssm[layer]
    }

    /// True when this cache carries Nemotron-H Mamba2 recurrent state.
    pub fn has_mamba(&self) -> bool {
        self.mamba_conv_dim != 0
    }

    /// Copy all Mamba2 recurrent state (conv history + SSM state) into `snap`, reusing
    /// its buffers so repeated calls don't reallocate. Used to roll back a speculative
    /// (MTP) verify forward: that forward advances every Mamba layer by the whole draft
    /// batch, but a partial accept must keep only the accepted prefix. Attention KV
    /// self-heals by positional overwrite; Mamba's running recurrence has no position
    /// index, so it must be saved and restored explicitly.
    pub fn snapshot_mamba_into(&self, snap: &mut MambaSnapshot) {
        let n = self.mamba_conv.len();
        if snap.conv.len() != n {
            snap.conv = vec![Vec::new(); n];
            snap.ssm = vec![Vec::new(); n];
        }
        for li in 0..n {
            snap.conv[li].clear();
            snap.conv[li].extend_from_slice(&self.mamba_conv[li]);
            snap.ssm[li].clear();
            snap.ssm[li].extend_from_slice(&self.mamba_ssm[li].data);
        }
    }

    /// Restore all Mamba2 recurrent state from a prior [`KvCache::snapshot_mamba_into`].
    /// Per-layer lengths are invariant across a sequence, so this is a plain copy.
    pub fn restore_mamba_from(&mut self, snap: &MambaSnapshot) {
        for li in 0..self.mamba_conv.len() {
            self.mamba_conv[li].copy_from_slice(&snap.conv[li]);
            self.mamba_ssm[li].data.copy_from_slice(&snap.ssm[li]);
        }
    }
}

/// Saved copy of all Mamba2 recurrent state, for speculative-decode rollback. Buffers
/// are reused across verify steps. See [`KvCache::snapshot_mamba_into`].
#[derive(Default)]
pub struct MambaSnapshot {
    conv: Vec<Vec<f32>>,
    ssm: Vec<Vec<f32>>,
}

/// A fully loaded model.
///
/// Fields present so the loader has a target; heavy runtime state (expert LRU,
/// pinned hot-store, DSA indexer, MTP head, profiling) is added as each
/// subsystem is ported. See PORTING.md.
pub struct Model {
    pub cfg: Config,
    pub shards: Shards,
    /// bits/param for experts and for the dense part
    pub ebits: i32,
    pub dbits: i32,

    pub embed: QTensor,
    pub lm_head: QTensor,
    pub final_norm: Vec<f32>,
    pub layers: Vec<Layer>,

    /// Kimi-K3's model-level attention-residual score vectors (`model.output_attn_res_*`),
    /// applied once after the last layer and before [`Model::final_norm`]. Same
    /// norm-times-proj form as the per-layer pair on [`Layer`]. Empty on every other
    /// arch — K3 is the only one with attention residuals.
    pub output_attn_res_norm: Vec<f32>,
    pub output_attn_res_proj: Vec<f32>,

    /// DeepSeek-V4's model-level Hyper-Connection head: the final `[hc, hidden] -> [hidden]`
    /// collapse before `final_norm` and the LM head. Unlike the per-layer `hc_pre` this has
    /// **no Sinkhorn** — a plain sigmoid gate — and `hc_head_scale` is a single scalar where
    /// the per-layer one is a 3-vector. Empty on every other arch.
    pub hc_head_fn: Vec<f32>,
    pub hc_head_base: Vec<f32>,
    pub hc_head_scale: f32,

    /// whether the DSA lightning indexer weights are present
    pub has_dsa: bool,
    /// whether the native MTP speculative head is present and loaded
    /// (mirrors `mtp.is_some()`; both are set together by the loader)
    pub has_mtp: bool,
    /// the loaded MTP head, when the container ships a complete one and `MTP=0`
    /// was not set. `None` on the default containers, which are converted without
    /// `--mtp`.
    pub mtp: Option<MtpHead>,
    /// DeepSeek-V4's DSpark drafting head. `None` unless `COLI_DSPARK=1` AND the container
    /// actually ships it — it costs ~11.2 GB, most of it three private 256-expert pools,
    /// so it is never loaded just because the weights are there.
    pub dspark: Option<DsparkHead>,
}

impl Layer {
    /// Resident bytes this layer holds — every weight and norm it owns. Routed experts
    /// are not here (they stream), which is exactly the split `Model::resident_bytes`
    /// needs to budget the expert cache against.
    ///
    /// Enumerated per field like `mark_gpu_eligible`, and wrong in the same silent way if
    /// a new arch's tensors are forgotten: the budget simply comes out too generous. The
    /// unit test asserts the count of fields covered here matches the struct.
    pub fn resident_bytes(&self) -> u64 {
        let q = |t: &QTensor| t.bytes().max(0) as u64;
        let oq = |t: &Option<QTensor>| t.as_ref().map(&q).unwrap_or(0);
        let v = |x: &Vec<f32>| (x.len() * 4) as u64;
        0 + q(&self.q_a)
            + q(&self.q_b)
            + q(&self.kv_a)
            + q(&self.kv_b)
            + q(&self.o)
            + oq(&self.o_a)
            + oq(&self.o_b)
            + self.o_a_groups.iter().map(&q).sum::<u64>()
            // Not via `v`: this is a Vec<u32>, and the field-count guard below keys on the
            // `Vec<f32>` declarations. 3.1 MB on each of the three hash layers.
            + (self.tid2eid.len() * 4) as u64
            + v(&self.attn_sink)
            // Hyper-Connections. `*_fn` is the big one: [24, 16384] f32 = 1.5 MB per
            // sublayer, so 3.1 MB per layer and ~135 MB across V4's 43 layers — small
            // against 145 GB, but omitting it is the exact silent over-generous budget
            // this function's doc warns about.
            + oq(&self.idx_wq_b)
            + oq(&self.idx_wproj)
            + oq(&self.idx_comp_wkv)
            + oq(&self.idx_comp_wgate)
            + v(&self.idx_comp_ape)
            + v(&self.idx_comp_norm)
            + oq(&self.comp_wkv)
            + oq(&self.comp_wgate)
            + v(&self.comp_ape)
            + v(&self.comp_norm)
            + v(&self.hc_attn_fn)
            + v(&self.hc_attn_base)
            + v(&self.hc_attn_scale)
            + v(&self.hc_ffn_fn)
            + v(&self.hc_ffn_base)
            + v(&self.hc_ffn_scale)
            + q(&self.gate_proj)
            + q(&self.up_proj)
            + q(&self.down_proj)
            + q(&self.sh_gate)
            + q(&self.sh_up)
            + q(&self.sh_down)
            + oq(&self.q_proj)
            + oq(&self.k_proj)
            + oq(&self.v_proj)
            + oq(&self.qkv_proj)
            + oq(&self.idx_q_proj)
            + oq(&self.idx_k_proj)
            + oq(&self.ix_wk)
            + oq(&self.ix_wq)
            + oq(&self.ix_wp)
            + oq(&self.mamba_in_proj)
            + oq(&self.mamba_out_proj)
            + oq(&self.fc1_latent)
            + oq(&self.fc2_latent)
            + oq(&self.attn_gate)
            + oq(&self.kda_b_proj)
            + oq(&self.kda_f_a)
            + oq(&self.kda_f_b)
            + v(&self.in_ln)
            + v(&self.post_ln)
            + v(&self.q_a_ln)
            + v(&self.kv_a_ln)
            + v(&self.q_norm)
            + v(&self.k_norm)
            + v(&self.idx_q_norm)
            + v(&self.idx_k_norm)
            + v(&self.router)
            + v(&self.router_bias)
            + v(&self.ix_knorm_w)
            + v(&self.ix_knorm_b)
            + v(&self.mamba_conv_w)
            + v(&self.mamba_conv_b)
            + v(&self.mamba_a_log)
            + v(&self.mamba_d)
            + v(&self.mamba_dt_bias)
            + v(&self.mamba_norm)
            + v(&self.kda_conv_q)
            + v(&self.kda_conv_k)
            + v(&self.kda_conv_v)
            + v(&self.kda_a_log)
            + v(&self.kda_dt_bias)
            + v(&self.kda_o_norm)
            + v(&self.attn_res_norm)
            + v(&self.attn_res_proj)
            + v(&self.mlp_res_norm)
            + v(&self.mlp_res_proj)
            + v(&self.routed_expert_norm)
    }
}

impl Model {
    /// Convenience accessor for the config.
    pub fn config(&self) -> &Config {
        &self.cfg
    }

    /// Bytes of dense weight this model holds resident for its lifetime: embeddings,
    /// lm_head, and every per-layer tensor. Routed experts are NOT counted — they stream
    /// through the [`crate::ExpertCache`], which is precisely what this number is used to
    /// budget.
    ///
    /// Exists because `MemAvailable` is the wrong input for that budget. Read right after
    /// the load it looks generous — the expert cache has not filled yet, and reclaimable
    /// page cache from reading the weights counts as available — so a budget derived from
    /// it overshoots. This is the number that does not move: Kimi-K3 is ~63 GB of a
    /// 121 GB box, where GLM is ~19, and that difference is the whole reason the same
    /// `MemTotal/3` cache budget fits one and gets the other OOM-killed.
    pub fn resident_bytes(&self) -> u64 {
        let mut n = self.embed.bytes().max(0) as u64 + self.lm_head.bytes().max(0) as u64;
        n += (self.final_norm.len() * 4) as u64;
        for l in &self.layers {
            n += l.resident_bytes();
        }
        // DSpark's stages are Layers too, and they are NOT in `self.layers`. Omitting them
        // is the same silent under-count this function's doc warns about: the budget comes
        // out too generous by the head's dense weight and the expert cache overcommits.
        if let Some(d) = &self.dspark {
            for l in &d.stages {
                n += l.resident_bytes();
            }
            n += [&d.main_norm, &d.norm, &d.confidence, &d.hc_head_fn, &d.hc_head_base]
                .iter()
                .map(|v| (v.len() * 4) as u64)
                .sum::<u64>();
            for t in [&d.main_proj, &d.markov_w1, &d.markov_w2] {
                n += t.as_ref().map(|t| t.bytes().max(0) as u64).unwrap_or(0);
            }
        }
        n
    }
}

#[cfg(test)]
mod resident_bytes_tests {
    use super::*;

    /// `Layer::resident_bytes` must cover EVERY weight field on the struct.
    ///
    /// It is enumerated per field, like `mark_gpu_eligible`, and fails the same silent
    /// way: forget a new arch's tensors and the number simply comes out too small, the
    /// expert-cache budget too generous, and the process gets OOM-killed on a model
    /// nobody re-checked. This counts the fields in the struct definition and compares
    /// against the number the function sums, so adding a field without adding it there
    /// fails here rather than in production.
    #[test]
    fn resident_bytes_covers_every_weight_field() {
        let src = include_str!("model.rs");
        let start = src.find("pub struct Layer {").expect("Layer struct");
        let body = &src[start..start + src[start..].find("\n}").expect("struct end")];
        let n_fields = body.matches(": QTensor").count()
            + body.matches(": Option<QTensor>").count()
            + body.matches(": Vec<f32>").count();

        let fstart = src
            .find("pub fn resident_bytes(&self) -> u64 {")
            .expect("fn");
        let fbody = &src[fstart..fstart + src[fstart..].find("\n    }").expect("fn end")];
        let n_summed = fbody.matches("q(&self.").count() + fbody.matches("v(&self.").count();

        assert_eq!(
            n_summed, n_fields,
            "Layer::resident_bytes sums {n_summed} fields but the struct has {n_fields} \
             weight fields — a new one was added without being counted"
        );
    }

    /// A layer with nothing in it costs nothing; one with a weight costs its bytes.
    #[test]
    fn resident_bytes_counts_what_is_present() {
        let empty = Layer::default();
        assert_eq!(empty.resident_bytes(), 0, "an empty layer holds nothing");

        let mut l = Layer::default();
        l.in_ln = vec![0.0; 16]; // 64 bytes
        l.attn_gate = Some(QTensor {
            fmt_code: 1,
            o: 4,
            i: 8,
            ..Default::default()
        });
        // int8: o*i codes + o f32 scales = 32 + 16 = 48
        assert_eq!(l.resident_bytes(), 64 + 48);
    }
}

#[cfg(test)]
mod kv_accounting_tests {
    use super::*;
    use colibri_core::Config;

    fn cfg_from(json: &str) -> Config {
        Config::from_json(&colibri_json::Json::parse(json).unwrap()).unwrap()
    }

    /// A hybrid stack must charge only its ATTENTION layers for per-token KV, and must
    /// include `k_full`/`v_full` (which `for_model` really allocates via `enable_gqa`,
    /// even though `Arch::is_gqa()` is false for Nemotron-H).
    ///
    /// This pins the bug that made serve quote 88 KB/token for the real model when the
    /// true figure is 24 KB: it charged all 88 layers and, because it keyed off
    /// `is_gqa()`, dropped the largest term entirely.
    #[test]
    fn hybrid_kv_counts_only_attention_layers_and_includes_full_kv() {
        // 4 layers: Mamba, MoE, Attn, Mamba — exactly one carries KV.
        let cfg = cfg_from(
            r#"{"model_type":"nemotron_h","hidden_size":8,"num_hidden_layers":4,
                "num_attention_heads":4,"num_key_value_heads":2,"head_dim":4,"vocab_size":8,
                "hybrid_override_pattern":"ME*M","n_routed_experts":4,"num_experts_per_tok":2,
                "moe_intermediate_size":6,"moe_latent_size":4,
                "moe_shared_expert_intermediate_size":6,"norm_topk_prob":false,
                "routed_scaling_factor":1.0,"mlp_hidden_act":"relu2","ssm_state_size":2,
                "conv_kernel":2,"mamba_num_heads":2,"mamba_head_dim":2,"n_groups":1,
                "chunk_size":2,"layer_norm_epsilon":1e-5}"#,
        );
        assert_eq!(
            KvCache::kv_layers(&cfg),
            1,
            "only the single '*' layer holds KV"
        );
        assert!(
            KvCache::allocates_gqa_kv(&cfg),
            "for_model calls enable_gqa for NemotronH"
        );

        // per attn layer: k_full/v_full (2*2*4 = 16) and NOTHING ELSE.
        //
        // `attention_gqa` ropes the key in place and stores the whole thing in
        // `k_full`/`v_full`; it never touches `latent` or `k_rot`, and never calls
        // `sync_device`. Those buffers are allocated but stay lazily uncommitted, so
        // charging `qk_rope` for them — and again for a device shadow of them — was two
        // phantom terms, ~12% of the real fleet figure. See `mirrors_kv_on_device`.
        assert!(!KvCache::mirrors_kv_on_device(&cfg), "GQA keeps no latent/rope rows");
        assert_eq!(KvCache::bytes_per_token(&cfg), 16 * 4);

        // Counting all 4 layers — the old behaviour — would be 4x too big.
        assert!(
            KvCache::bytes_per_token(&cfg) * 4
                > KvCache::bytes_per_token(&cfg) * cfg.n_layers as usize / 2,
            "sanity: per-token figure must not scale with total layer count"
        );
    }

    /// The Mamba2 recurrent state is per-SEQUENCE and O(1) in context. It must be
    /// reported separately, never folded into the per-token figure — otherwise it
    /// scales with context length and the short-prompt case stays under-reserved.
    #[test]
    fn hybrid_reports_fixed_mamba_state_separately() {
        let cfg = cfg_from(
            r#"{"model_type":"nemotron_h","hidden_size":8,"num_hidden_layers":4,
                "num_attention_heads":4,"num_key_value_heads":2,"head_dim":4,"vocab_size":8,
                "hybrid_override_pattern":"ME*M","n_routed_experts":4,"num_experts_per_tok":2,
                "moe_intermediate_size":6,"moe_latent_size":4,
                "moe_shared_expert_intermediate_size":6,"norm_topk_prob":false,
                "routed_scaling_factor":1.0,"mlp_hidden_act":"relu2","ssm_state_size":2,
                "conv_kernel":2,"mamba_num_heads":2,"mamba_head_dim":2,"n_groups":1,
                "chunk_size":2,"layer_norm_epsilon":1e-5}"#,
        );
        // 2 Mamba layers x (conv: d_conv * (inter + 2*groups*state)  +  ssm: nh*hd*state)
        let conv_dim = cfg.mamba_inter as usize + 2 * 1 * 2;
        let per_layer = 2 * conv_dim + 2 * 2 * 2;
        assert_eq!(KvCache::fixed_bytes(&cfg), 2 * per_layer * 4);

        // bytes_for = per-token * n + fixed, and the fixed part does NOT scale with n.
        let pt = KvCache::bytes_per_token(&cfg);
        let fx = KvCache::fixed_bytes(&cfg);
        assert_eq!(KvCache::bytes_for(&cfg, 100), pt * 100 + fx);
        assert_eq!(KvCache::bytes_for(&cfg, 1), pt + fx);
        assert!(fx > 0, "a hybrid model must report non-zero fixed state");
    }

    /// A uniform GQA transformer: every layer holds KV, there is no fixed state, and the
    /// per-token cost is `k_full` + `v_full` and nothing else.
    ///
    /// This used to be named `..._is_unchanged` and assert the older formula, which added
    /// `qk_rope` plus a device mirror of it. Both were phantom — `attention_gqa` writes
    /// neither (`gqa_writes_no_latent_or_roped_key_rows` poisons the rows and watches them
    /// survive) — so "unchanged" was guarding the wrong number.
    #[test]
    fn uniform_gqa_charges_k_full_and_v_full_only() {
        let cfg = cfg_from(
            r#"{"model_type":"minimax_m2","hidden_size":8,"intermediate_size":6,
                "num_hidden_layers":3,"num_attention_heads":4,"num_key_value_heads":2,
                "head_dim":4,"partial_rotary_factor":0.5,"rotary_dim":2,"vocab_size":8,
                "num_local_experts":4,"num_experts_per_tok":2,"shared_intermediate_size":0,
                "scoring_func":"sigmoid","use_routing_bias":true,"hidden_act":"silu",
                "rms_norm_eps":1e-5,"eos_token_id":2}"#,
        );
        assert!(
            cfg.layer_kind.is_empty(),
            "uniform archs carry no layer_kind"
        );
        assert_eq!(
            KvCache::kv_layers(&cfg),
            cfg.n_layers as usize,
            "every layer holds KV"
        );
        assert_eq!(KvCache::fixed_bytes(&cfg), 0, "no recurrent state");
        assert_eq!(
            KvCache::bytes_for(&cfg, 7),
            KvCache::bytes_per_token(&cfg) * 7
        );

        // A GQA transformer's KV is `k_full` + `v_full`, full stop — no latent, no roped-key
        // row, no device shadow (`attention_gqa` writes neither and never syncs).
        assert!(!KvCache::mirrors_kv_on_device(&cfg));
        let expect =
            cfg.n_layers as usize * (2 * cfg.n_kv_heads as usize * cfg.qk_head as usize) * 4;
        assert_eq!(KvCache::bytes_per_token(&cfg), expect);
        // The `qk_rope` term this used to add is not zero on this config, so its removal is
        // a real change and not a no-op the test could pass through inattention.
        assert!(cfg.qk_rope > 0, "the dropped term was non-zero here");
    }

    /// The real Kimi-K3 geometry (93 layers = 69 KDA + 24 gated-MLA), so the figures the
    /// admission path depends on are pinned to the model actually served, not a toy.
    fn kimi_k3_cfg() -> Config {
        cfg_from(
            r#"{"model_type":"kimi_k3",
                "architectures":["KimiK3ForConditionalGeneration"],
                "text_config":{
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
                  "linear_attn_config":{"head_dim":128,"num_heads":96,
                    "short_conv_kernel_size":4,
                    "full_attn_layers":[4,8,12,16,20,24,28,32,36,40,44,48,52,56,60,64,68,
                                        72,76,80,84,88,92,93]}},
                "vision_config":{"vt_hidden_size":1024}}"#,
        )
    }

    /// K3 splits its stack across two mixers, and each is accounted on a different axis:
    /// the 24 gated-MLA layers hold a context-growing KV cache, the 69 KDA layers a
    /// fixed-size recurrent state. Charging all 93 layers per token would be ~3.9x over;
    /// omitting the KDA state (which is what happens if `fixed_bytes` only knows about
    /// Mamba) leaves ~475 MB per concurrent sequence unreserved.
    #[test]
    fn kimi_k3_charges_mla_per_token_and_kda_per_sequence() {
        let cfg = kimi_k3_cfg();
        assert_eq!(cfg.n_layers, 93);
        assert_eq!(
            KvCache::kv_layers(&cfg),
            24,
            "only the gated-MLA layers hold KV"
        );
        assert!(
            !KvCache::allocates_gqa_kv(&cfg),
            "K3 is MLA: no k_full/v_full"
        );

        // Per token: 24 layers x (kv_lora 512 + qk_rope 64) x f32, doubled by the device
        // shadow under cuda. 108 KiB/token there, 54 KiB on the host-only build.
        let mla = cfg.kv_lora as usize + cfg.qk_rope as usize;
        let shadow = if cfg!(feature = "cuda") { mla } else { 0 };
        assert_eq!(KvCache::bytes_per_token(&cfg), 24 * (mla + shadow) * 4);

        // Per sequence: 69 KDA layers x (conv history + delta-rule matrix), O(1) in ctx.
        let conv = 4 * 3 * 96 * 128; // d_conv x (q,k,v) x n_heads*head_dim
        let state = 96 * 128 * 128; // [n_heads, head_dim, head_dim]
        assert_eq!(KvCache::fixed_bytes(&cfg), 69 * (conv + state) * 4);
        assert_eq!(
            KvCache::fixed_bytes(&cfg),
            474_808_320,
            "~475 MB per sequence"
        );

        // The fixed term must not scale with context — that is the whole point of it
        // being separate, and the reason short requests are where it bites.
        let (pt, fx) = (KvCache::bytes_per_token(&cfg), KvCache::fixed_bytes(&cfg));
        assert_eq!(KvCache::bytes_for(&cfg, 1), pt + fx);
        assert_eq!(KvCache::bytes_for(&cfg, 200_000), pt * 200_000 + fx);
    }

    /// The drift guard. Every KV accounting bug in this file has been a formula that
    /// disagreed with what was actually allocated, so assert the two against each other
    /// directly rather than against a second hand-derived constant.
    #[test]
    fn kimi_k3_fixed_bytes_equals_what_enable_kda_allocates() {
        let cfg = kimi_k3_cfg();
        let mut kv = KvCache::new(
            cfg.n_layers as usize,
            cfg.kv_lora as usize,
            cfg.qk_rope as usize,
            1,
        );
        kv.enable_kda(&cfg);

        let allocated: usize = kv.kda_conv.iter().map(Vec::len).sum::<usize>()
            + kv.kda_state.iter().map(Vec::len).sum::<usize>();
        assert_eq!(allocated * 4, KvCache::fixed_bytes(&cfg));

        // ...and it landed on the KDA rows specifically, not on all 93.
        let populated = kv.kda_state.iter().filter(|v| !v.is_empty()).count();
        assert_eq!(populated, 69, "only KDA layers carry recurrent state");
        for (li, k) in cfg.layer_kind.iter().enumerate() {
            let has_state = !kv.kda_state[li].is_empty();
            assert_eq!(
                has_state,
                *k == LayerKind::Kda,
                "layer {li} state/kind mismatch"
            );
        }
    }

    /// A KDA layer must never be charged as a Mamba one or vice versa: the two live in
    /// the same `fixed_bytes` sum and are distinguished only by `LayerKind`.
    #[test]
    fn kda_and_mamba_fixed_state_do_not_bleed_into_each_other() {
        let k3 = kimi_k3_cfg();
        assert_eq!(k3.mamba_n_heads, 0, "K3 sets no Mamba dims");
        assert!(!k3.layer_kind.contains(&LayerKind::Mamba));

        let nemo = cfg_from(
            r#"{"model_type":"nemotron_h","hidden_size":8,"num_hidden_layers":4,
                "num_attention_heads":4,"num_key_value_heads":2,"head_dim":4,"vocab_size":8,
                "hybrid_override_pattern":"ME*M","n_routed_experts":4,"num_experts_per_tok":2,
                "moe_intermediate_size":6,"moe_latent_size":4,
                "moe_shared_expert_intermediate_size":6,"norm_topk_prob":false,
                "routed_scaling_factor":1.0,"mlp_hidden_act":"relu2","ssm_state_size":2,
                "conv_kernel":2,"mamba_num_heads":2,"mamba_head_dim":2,"n_groups":1,
                "chunk_size":2,"layer_norm_epsilon":1e-5}"#,
        );
        assert_eq!(nemo.kda_n_heads, 0, "Nemotron sets no KDA dims");
        assert!(!nemo.layer_kind.contains(&LayerKind::Kda));
        // Nemotron's figure is unchanged by the KDA term existing.
        let conv_dim = nemo.mamba_inter as usize + 2 * 2;
        assert_eq!(
            KvCache::fixed_bytes(&nemo),
            2 * (2 * conv_dim + 2 * 2 * 2) * 4
        );
    }

    // ---- the raw-KV ring (DeepSeek-V4, task #69) --------------------------

    const RING_LAYERS: usize = 2;
    const RING_KV_LORA: usize = 4;
    const RING_QK_ROPE: usize = 2;
    const RING_WINDOW: usize = 8;
    /// Far past the window, so a ring of `RING_WINDOW` rows wraps several times.
    const RING_TOTAL: usize = 40;

    /// Position- and lane-derived, so a span returned shifted by one row — the likeliest
    /// ring bug — fails instead of coincidentally matching.
    fn ring_lat(p: usize, j: usize, li: usize) -> f32 {
        (p * 100 + j) as f32 + li as f32 * 0.5
    }
    fn ring_rot(p: usize, j: usize, li: usize) -> f32 {
        -((p * 10 + j) as f32) - li as f32 * 0.5
    }

    /// Write position `p` on every layer, then read back every span that a query at `p`
    /// could ask for, checking each row against the position that wrote it.
    ///
    /// Interleaved on purpose: this is the order `dsv4_attention` actually runs in, and it
    /// is the only order a ring can satisfy — writing position 40 into an 8-row ring has
    /// already destroyed position 32, so a write-everything-then-read-everything test
    /// would be asserting a property the design does not have.
    fn ring_sweep(kv: &mut KvCache, window: usize, total: usize) {
        for p in 0..total {
            for li in 0..RING_LAYERS {
                for (j, v) in kv.latent_row_mut(li, p).iter_mut().enumerate() {
                    *v = ring_lat(p, j, li);
                }
                for (j, v) in kv.krot_row_mut(li, p).iter_mut().enumerate() {
                    *v = ring_rot(p, j, li);
                }
            }
            let (start, end) = ((p + 1).saturating_sub(window), p + 1);
            for li in 0..RING_LAYERS {
                let (mut lat, mut rot) = (Vec::new(), Vec::new());
                kv.extend_latent_rows(li, start, end, &mut lat);
                kv.extend_krot_rows(li, start, end, &mut rot);
                assert_eq!(lat.len(), (end - start) * RING_KV_LORA);
                assert_eq!(rot.len(), (end - start) * RING_QK_ROPE);
                for (i, q) in (start..end).enumerate() {
                    for j in 0..RING_KV_LORA {
                        assert_eq!(
                            lat[i * RING_KV_LORA + j],
                            ring_lat(q, j, li),
                            "layer {li} span [{start},{end}) row {q} lane {j}"
                        );
                    }
                    for j in 0..RING_QK_ROPE {
                        assert_eq!(
                            rot[i * RING_QK_ROPE + j],
                            ring_rot(q, j, li),
                            "layer {li} krot span [{start},{end}) row {q} lane {j}"
                        );
                    }
                }
                // With no ring the borrowed slice must agree with the appended one; that
                // equivalence is what lets the ring be swapped in under the reader.
                if kv.ring() == 0 {
                    assert_eq!(kv.latent_rows(li, start, end), &lat[..]);
                    assert_eq!(kv.krot_rows(li, start, end), &rot[..]);
                }
            }
        }
    }

    /// The storage contract DeepSeek-V4's 128-token sliding window depends on, in the
    /// pre-ring layout: rows indexed by absolute position, buffers sized to `max_t`.
    #[test]
    fn latent_and_krot_spans_read_back_the_positions_written() {
        let mut kv = KvCache::new(RING_LAYERS, RING_KV_LORA, RING_QK_ROPE, RING_TOTAL);
        assert_eq!(kv.ring(), 0, "no arch rings by default");
        ring_sweep(&mut kv, RING_WINDOW, RING_TOTAL);
    }

    /// ...and the same contract once the tier is a ring. This is the test the ring exists
    /// to pass: an off-by-one in `pos % ring` returns a row that is in bounds and
    /// plausible, so it would surface as wrong TOKENS, never as a crash.
    #[test]
    fn a_ringed_cache_reads_back_the_same_spans_as_a_linear_one() {
        let mut kv = KvCache::new(RING_LAYERS, RING_KV_LORA, RING_QK_ROPE, RING_TOTAL);
        kv.ring_ensure(RING_WINDOW);
        assert_eq!(kv.ring(), RING_WINDOW);
        ring_sweep(&mut kv, RING_WINDOW, RING_TOTAL);
    }

    /// The point of the exercise: a ring holds `R` rows however long the sequence runs.
    /// Without this the buffers are `max_t`-sized and every generated token commits
    /// another row — the term that made V4 context cost 206.9 KB/token.
    #[test]
    fn ring_storage_is_constant_in_context_length() {
        let mut short = KvCache::new(1, RING_KV_LORA, RING_QK_ROPE, 1_000);
        let mut long = KvCache::new(1, RING_KV_LORA, RING_QK_ROPE, 1_000_000);
        short.ring_ensure(RING_WINDOW);
        long.ring_ensure(RING_WINDOW);
        assert_eq!(short.latent[0].len(), RING_WINDOW * RING_KV_LORA);
        assert_eq!(long.latent[0].len(), RING_WINDOW * RING_KV_LORA);
        assert_eq!(long.k_rot[0].len(), RING_WINDOW * RING_QK_ROPE);
        // Same width at a thousand times the context — and far below what the linear
        // layout would have addressed for the larger one.
        assert!(long.latent[0].len() * 1000 < 1_000_000 * RING_KV_LORA);
    }

    /// Growing the ring re-homes the live rows. `p % ring` is a different address for a
    /// different `ring`, so a resize that only reallocated would silently re-point every
    /// position it kept. Reachable if one sequence ever prefills wider than the first call
    /// sized for (an MTP verify batch, say) — rare, and silent if wrong.
    #[test]
    fn growing_the_ring_carries_the_live_rows_to_their_new_slots() {
        let mut kv = KvCache::new(RING_LAYERS, RING_KV_LORA, RING_QK_ROPE, RING_TOTAL);
        kv.ring_ensure(RING_WINDOW);
        for p in 0..RING_TOTAL {
            for li in 0..RING_LAYERS {
                for (j, v) in kv.latent_row_mut(li, p).iter_mut().enumerate() {
                    *v = ring_lat(p, j, li);
                }
                for (j, v) in kv.krot_row_mut(li, p).iter_mut().enumerate() {
                    *v = ring_rot(p, j, li);
                }
            }
        }
        // 8 -> 13: not a multiple, so every live row lands in a different slot.
        kv.ring_ensure(13);
        assert_eq!(kv.ring(), 13);
        let (lo, hi) = (RING_TOTAL - RING_WINDOW, RING_TOTAL);
        for li in 0..RING_LAYERS {
            let (mut lat, mut rot) = (Vec::new(), Vec::new());
            kv.extend_latent_rows(li, lo, hi, &mut lat);
            kv.extend_krot_rows(li, lo, hi, &mut rot);
            for (i, p) in (lo..hi).enumerate() {
                for j in 0..RING_KV_LORA {
                    assert_eq!(lat[i * RING_KV_LORA + j], ring_lat(p, j, li), "row {p} lane {j}");
                }
                for j in 0..RING_QK_ROPE {
                    assert_eq!(rot[i * RING_QK_ROPE + j], ring_rot(p, j, li), "krot {p} lane {j}");
                }
            }
        }
    }

    /// A span wider than the ring is not representable — the rows it names no longer all
    /// exist. Loud, because in bounds and wrong is the failure this whole design guards
    /// against; `dsv4_attention` sizes the ring from the very span it then reads, so a
    /// caller reaching this has broken that pairing.
    #[test]
    #[should_panic(expected = "ring holds")]
    fn reading_more_rows_than_the_ring_holds_panics() {
        let mut kv = KvCache::new(1, RING_KV_LORA, RING_QK_ROPE, RING_TOTAL);
        kv.ring_ensure(RING_WINDOW);
        let mut out = Vec::new();
        kv.extend_latent_rows(0, 0, RING_WINDOW + 1, &mut out);
    }

    /// The borrowed-slice accessors are absolute-indexed and cannot serve a ring. They
    /// refuse rather than return the wrong positions' data, which is what makes it safe
    /// for the MLA path to keep using them.
    #[test]
    #[should_panic(expected = "use extend_latent_rows")]
    fn latent_rows_refuses_a_ringed_cache() {
        let mut kv = KvCache::new(1, RING_KV_LORA, RING_QK_ROPE, RING_TOTAL);
        kv.ring_ensure(RING_WINDOW);
        let _ = kv.latent_rows(0, 0, 4);
    }

    /// V4's accounting shape: a **latent** attention stack (so `allocates_gqa_kv` is false,
    /// as it is for the real V4) with `sliding_window` set. K3's real geometry is the base
    /// because `window` and `arch` are the only fields these figures key off, which is what
    /// lets the ring be reasoned about without a V4 checkpoint.
    ///
    /// **Must not be GQA-based.** It was, briefly, and that silently disarmed the whole
    /// point: `allocates_gqa_kv` was already true, so the `window` clause in
    /// `mirrors_kv_on_device` became redundant and deleting it broke nothing. A mutation
    /// run caught it. Any config used to test the ring's *accounting* has to reach the
    /// window clause to exercise it.
    fn ringed_cfg(window: i32) -> Config {
        let mut cfg = kimi_k3_cfg();
        assert!(!KvCache::allocates_gqa_kv(&cfg), "must reach the window clause");
        cfg.window = window;
        cfg
    }

    /// The GQA/hybrid shape, for the arm of the accounting that has no latent rows at all.
    fn gqa_cfg_for_accounting() -> Config {
        cfg_from(
            r#"{"model_type":"nemotron_h","hidden_size":8,"num_hidden_layers":4,
                "num_attention_heads":4,"num_key_value_heads":2,"head_dim":4,"vocab_size":8,
                "hybrid_override_pattern":"ME*M","n_routed_experts":4,"num_experts_per_tok":2,
                "moe_intermediate_size":6,"moe_latent_size":4,
                "moe_shared_expert_intermediate_size":6,"norm_topk_prob":false,
                "routed_scaling_factor":1.0,"mlp_hidden_act":"relu2","ssm_state_size":2,
                "conv_kernel":2,"mamba_num_heads":2,"mamba_head_dim":2,"n_groups":1,
                "chunk_size":2,"layer_norm_epsilon":1e-5}"#,
        )
    }

    /// The point of the ring, in the ledger: generated tokens add no raw rows.
    ///
    /// The residual growth is the COMPRESSED tier, which is what carries context past the
    /// window and genuinely is per-token — so the assertion is "grows far slower", not
    /// "does not grow".
    #[test]
    fn generated_tokens_stop_adding_raw_kv_rows_under_a_ring() {
        let ring = ringed_cfg(128);
        let flat = ringed_cfg(0);

        // Same prompt, 100x the generation.
        let (short, long) = (
            KvCache::bytes_for_split(&ring, 512, 400),
            KvCache::bytes_for_split(&ring, 512, 40_000),
        );
        assert_eq!(short, long, "no compressor on this cfg: generation is free");

        // Without a ring the same pair scales with every token. Net of `fixed_bytes`, which
        // is the per-sequence recurrent state and constant in both arms — leaving it in
        // would dilute the ratio with a term that has nothing to do with the ring.
        let fx = KvCache::fixed_bytes(&flat);
        let flat_short = KvCache::bytes_for_split(&flat, 512, 400) - fx;
        let flat_long = KvCache::bytes_for_split(&flat, 512, 40_000) - fx;
        assert_eq!(flat_long, flat_short * 40_512 / 912, "unringed: linear in total tokens");
        // And the ring is the whole reason: same prompt, same generation, ~79x apart.
        assert!(
            flat_long > (long - KvCache::fixed_bytes(&ring)) * 40,
            "ring should be far cheaper: {long} vs {flat_long}"
        );
    }

    /// Prompts are NOT cheaper. The ring holds `max(window, prompt)` rows, so a long
    /// prompt costs the same *rows* it did — the honest limit until prefill is chunked, and
    /// the reason `bytes_per_token` stays at its all-prompt worst case.
    ///
    /// Stated in ROWS, not bytes: a ringed arch also drops the device shadow (it cannot
    /// carry one — see `mirrors_kv_on_device`), so comparing bytes against an unringed
    /// config would fold that separate saving in and stop testing this claim.
    #[test]
    fn a_long_prompt_still_costs_full_price_under_a_ring() {
        let ring = ringed_cfg(128);
        let flat = ringed_cfg(0);
        for n in [512usize, 4_096, 40_000] {
            assert_eq!(KvCache::raw_rows(&ring, n, 0), n, "all-prompt {n}: every row kept");
            assert_eq!(KvCache::raw_rows(&flat, n, 0), n);
            // ...and the charge is linear in those rows, so doubling the prompt doubles it.
            assert_eq!(
                KvCache::bytes_for(&ring, 2 * n) - KvCache::fixed_bytes(&ring),
                2 * (KvCache::bytes_for(&ring, n) - KvCache::fixed_bytes(&ring)),
                "all-prompt cost is linear in n at {n}"
            );
        }
        // Below the window the ring rounds UP to it — it cannot hold fewer rows than the
        // span every decode step reads.
        assert_eq!(KvCache::raw_rows(&ring, 10, 0), 128);
        assert_eq!(KvCache::raw_rows(&flat, 10, 0), 10);
    }

    /// Exactly one family syncs KV to the device, and it is the one whose reader calls
    /// `sync_device`. Pinned per-arch because the predicate is a deny-list: if a new
    /// architecture is added and left un-audited it lands in the CHARGED arm, which
    /// over-reserves — the safe direction — and this test says so out loud.
    #[test]
    fn only_the_mla_reader_is_charged_for_a_device_shadow() {
        // Charged: the weight-absorption reader, the sole caller of `sync_device`.
        assert!(KvCache::mirrors_kv_on_device(&kimi_k3_cfg()), "K3's gated-MLA layers sync");

        // Not charged, for two independent reasons — and BOTH arms must be exercised by a
        // config that actually reaches them, or one clause of the predicate is untested.
        let gqa = gqa_cfg_for_accounting();
        assert!(KvCache::allocates_gqa_kv(&gqa));
        assert!(
            !KvCache::mirrors_kv_on_device(&gqa),
            "GQA writes k_full/v_full only — no latent, no rope row, no shadow"
        );
        let v4 = ringed_cfg(128);
        assert!(!KvCache::allocates_gqa_kv(&v4), "V4 is latent, not GQA: the GQA clause misses it");
        assert!(
            !KvCache::mirrors_kv_on_device(&v4),
            "a ringed raw tier cannot carry an absolute-indexed shadow — sync_device asserts it"
        );

        // And what was dropped is exactly the two phantom terms — `k_rot` and its mirror —
        // not some other quantity that happens to make the arithmetic work. Shown on V4,
        // where flipping `window` back to 0 reproduces the old charge on the same config.
        let mut charged = v4.clone();
        charged.window = 0;
        let shadow = if cfg!(feature = "cuda") {
            v4.kv_lora as usize + v4.qk_rope as usize
        } else {
            0
        };
        assert_eq!(
            KvCache::bytes_per_token(&charged) - KvCache::bytes_per_token(&v4),
            KvCache::kv_layers(&v4) * (v4.qk_rope as usize + shadow) * 4,
            "the delta must be k_rot + its device mirror, exactly"
        );
    }

    /// The MLA models must be UNCHANGED by that fix. They are the ones that really do
    /// mirror, so any movement in their figure would be the fix over-reaching.
    #[test]
    fn the_mla_models_keep_their_full_latent_rope_and_shadow_charge() {
        let cfg = kimi_k3_cfg();
        let mla = cfg.kv_lora as usize + cfg.qk_rope as usize;
        let shadow = if cfg!(feature = "cuda") { mla } else { 0 };
        assert_eq!(
            KvCache::bytes_per_token(&cfg),
            KvCache::kv_layers(&cfg) * (mla + shadow) * 4,
            "K3: latent + rope on host, mirrored on device"
        );
    }

    /// `bytes_for` is the split-free form and must stay the all-prompt worst case: it
    /// feeds serve's admission and `context_in_kv_budget`, where guessing a cheaper split
    /// would admit a prompt into memory that cannot hold it.
    #[test]
    fn bytes_for_is_still_per_token_times_n_plus_fixed() {
        for cfg in [ringed_cfg(128), ringed_cfg(0), kimi_k3_cfg()] {
            let (pt, fx) = (KvCache::bytes_per_token(&cfg), KvCache::fixed_bytes(&cfg));
            for n in [1usize, 128, 100_000] {
                // Only exact once the prompt reaches the window; below it the ring rounds
                // up, which is a larger charge, never a smaller one.
                let want = pt * n + fx;
                let got = KvCache::bytes_for(&cfg, n);
                assert!(got >= want, "cfg window {}: {got} < {want} at n={n}", cfg.window);
                if n >= cfg.window.max(0) as usize {
                    assert_eq!(got, want, "cfg window {}: n={n}", cfg.window);
                }
            }
        }
    }
}
