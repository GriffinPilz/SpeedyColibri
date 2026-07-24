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
    pub q_norm: Vec<f32>,        // per-head RMSNorm weight [head_dim] (gemma-folded)
    pub k_norm: Vec<f32>,        // per-head RMSNorm weight [head_dim] (gemma-folded)

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
    pub ix_wk: Option<QTensor>,     // key proj: hidden -> index_hd
    pub ix_wq: Option<QTensor>,     // query proj: q_lora -> index_nh*index_hd
    pub ix_wp: Option<QTensor>,     // per-head weight proj: hidden -> index_nh
    pub ix_knorm_w: Vec<f32>,       // key LayerNorm weight (eps 1e-6)
    pub ix_knorm_b: Vec<f32>,       // key LayerNorm bias

    // ---- Nemotron-H Mamba2 mixer (present on `LayerKind::Mamba` layers) ----
    // `in_ln` above is the single block-input RMSNorm; Nemotron has no `post_ln`.
    pub mamba_in_proj: Option<QTensor>, // hidden -> d_inner + conv_dim + n_heads (18560)
    pub mamba_out_proj: Option<QTensor>, // d_inner -> hidden
    pub mamba_conv_w: Vec<f32>,     // depthwise conv [conv_dim, k] (from [conv_dim,1,k])
    pub mamba_conv_b: Vec<f32>,     // conv bias [conv_dim]
    pub mamba_a_log: Vec<f32>,      // [n_heads]; A = -exp(a_log)
    pub mamba_d: Vec<f32>,          // skip connection [n_heads]
    pub mamba_dt_bias: Vec<f32>,    // step bias [n_heads]
    pub mamba_norm: Vec<f32>,       // gated RMSNorm weight [d_inner]

    // ---- Nemotron-H latent-MoE projections (present on `LayerKind::Moe` layers) ----
    // Routed experts (`sh_*` unused here; routed via the expert provider) run in the
    // `moe_latent` space between these two projections; `router`/`router_bias` above
    // are reused for the gate. `up_proj`/`down_proj` reused for the shared expert.
    pub fc1_latent: Option<QTensor>, // hidden -> moe_latent
    pub fc2_latent: Option<QTensor>, // moe_latent -> hidden
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
pub struct MtpHead {
    /// the MTP transformer block (always sparse), at layer index `n_layers`
    pub layer: Layer,
    /// `[D, 2D]` — projects the concatenated `[e ; h]` back to hidden width
    pub eh_proj: QTensor,
    /// RMSNorm weight applied to the next token's embedding
    pub enorm: Vec<f32>,
    /// RMSNorm weight applied to the (already final_norm'd) hidden state
    pub hnorm: Vec<f32>,
    /// `shared_head.norm.weight` — the head's own final norm before `lm_head`
    pub mtp_norm: Vec<f32>,
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
        let conv_dim = cfg.mamba_inter as usize
            + 2 * cfg.mamba_n_groups as usize * cfg.mamba_d_state as usize;
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

    /// Allocate a cache sized for `model`, holding up to `max_t` tokens.
    ///
    /// When the model carries an MTP head this allocates **`n_layers + 1`** rows
    /// (C: `NR = c->n_layers + 1`) — the head is a real layer at index `n_layers`
    /// with its own KV. That row starts [`KV_UNSET`] rather than 0 (C:
    /// `kv_start[i] = -1`): unlike the main stack, the head's cache begins at the
    /// first *decode* position, not at the start of the prompt, so it holds only a
    /// partial suffix of the sequence.
    pub fn for_model(model: &Model, max_t: usize) -> KvCache {
        let n_layers = model.cfg.n_layers as usize;
        let rows = n_layers + usize::from(model.has_mtp);
        let mut kv = KvCache::new(
            rows,
            model.cfg.kv_lora as usize,
            model.cfg.qk_rope as usize,
            max_t,
        );
        if model.has_mtp {
            kv.kv_start[n_layers] = KV_UNSET;
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
    /// do (Nemotron-H: 8 of 88) — `for_model` allocates rows for every layer, but the
    /// non-attention rows are never written and stay lazily uncommitted, so they cost
    /// no physical memory and must not be charged for.
    fn kv_layers(cfg: &Config) -> usize {
        if cfg.layer_kind.is_empty() {
            cfg.n_layers as usize
        } else {
            cfg.layer_kind.iter().filter(|k| **k == LayerKind::Attn).count()
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
    pub fn bytes_per_token(cfg: &Config) -> usize {
        let mla = cfg.kv_lora as usize + cfg.qk_rope as usize; // mirrored on device
        let gqa_full = if Self::allocates_gqa_kv(cfg) {
            2 * cfg.n_kv_heads as usize * cfg.qk_head as usize // k_full + v_full: host only
        } else {
            0
        };
        let device_shadow = if cfg!(feature = "cuda") { mla } else { 0 };
        Self::kv_layers(cfg) * (mla + gqa_full + device_shadow) * 4
    }

    /// Per-sequence KV bytes that do **not** scale with context length: the Mamba2
    /// recurrent state (conv history + selective-scan state) on each Mamba layer.
    ///
    /// O(1) in context, but far from free — Nemotron-H carries ~174 MB per sequence
    /// across its 40 Mamba layers, dominated by the `[n_heads, head_dim, d_state]` scan
    /// state. A reservation that counts only per-token bytes under-commits by that much
    /// for every concurrent sequence, and the shortfall is worst for SHORT requests,
    /// where the per-token term is too small to accidentally cover it.
    pub fn fixed_bytes(cfg: &Config) -> usize {
        if cfg.layer_kind.is_empty() {
            return 0;
        }
        let n_mamba = cfg.layer_kind.iter().filter(|k| **k == LayerKind::Mamba).count();
        let conv_dim =
            cfg.mamba_inter as usize + 2 * cfg.mamba_n_groups as usize * cfg.mamba_d_state as usize;
        let conv = cfg.mamba_d_conv as usize * conv_dim;
        let ssm = cfg.mamba_n_heads as usize * cfg.mamba_head_dim as usize
            * cfg.mamba_d_state as usize;
        n_mamba * (conv + ssm) * 4
    }

    /// Total resident bytes for a sequence of `n_tokens`: the per-token KV plus the
    /// fixed per-sequence state. This is what a reservation should ask for.
    pub fn bytes_for(cfg: &Config, n_tokens: usize) -> usize {
        Self::bytes_per_token(cfg)
            .saturating_mul(n_tokens)
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
        let (max_t, kvl, r) = (self.max_t, self.kv_lora, self.qk_rope);
        let dev = self
            .dev
            .get_or_insert_with(|| crate::gpu::DeviceKv::new(n_layers, max_t));
        dev.sync(layer, &self.latent[layer], &self.k_rot[layer], kvl, r, pos_base, tk)
    }

    pub fn kv_lora(&self) -> usize {
        self.kv_lora
    }
    pub fn qk_rope(&self) -> usize {
        self.qk_rope
    }

    /// Normalized-latent row for `(layer, pos)`.
    pub fn latent_row(&self, layer: usize, pos: usize) -> &[f32] {
        &self.latent[layer][pos * self.kv_lora..(pos + 1) * self.kv_lora]
    }
    pub fn latent_row_mut(&mut self, layer: usize, pos: usize) -> &mut [f32] {
        &mut self.latent[layer][pos * self.kv_lora..(pos + 1) * self.kv_lora]
    }

    /// Roped k_rot row for `(layer, pos)`.
    pub fn krot_row(&self, layer: usize, pos: usize) -> &[f32] {
        &self.k_rot[layer][pos * self.qk_rope..(pos + 1) * self.qk_rope]
    }
    pub fn krot_row_mut(&mut self, layer: usize, pos: usize) -> &mut [f32] {
        &mut self.k_rot[layer][pos * self.qk_rope..(pos + 1) * self.qk_rope]
    }

    /// Contiguous latent rows `[start, end)` for a layer — a single slice the
    /// batched `kv_b` reconstruction multiplies against.
    pub fn latent_rows(&self, layer: usize, start: usize, end: usize) -> &[f32] {
        &self.latent[layer][start * self.kv_lora..end * self.kv_lora]
    }

    /// Contiguous roped-key rows `[start, end)` for a layer.
    pub fn krot_rows(&self, layer: usize, start: usize, end: usize) -> &[f32] {
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

    /// whether the DSA lightning indexer weights are present
    pub has_dsa: bool,
    /// whether the native MTP speculative head is present and loaded
    /// (mirrors `mtp.is_some()`; both are set together by the loader)
    pub has_mtp: bool,
    /// the loaded MTP head, when the container ships a complete one and `MTP=0`
    /// was not set. `None` on the default containers, which are converted without
    /// `--mtp`.
    pub mtp: Option<MtpHead>,
}

impl Model {
    /// Convenience accessor for the config.
    pub fn config(&self) -> &Config {
        &self.cfg
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
        assert_eq!(KvCache::kv_layers(&cfg), 1, "only the single '*' layer holds KV");
        assert!(KvCache::allocates_gqa_kv(&cfg), "for_model calls enable_gqa for NemotronH");

        // per attn layer: mla(kv_lora 0 + qk_rope 4) + k_full/v_full(2*2*4=16) + shadow(mla)
        let mla = cfg.kv_lora as usize + cfg.qk_rope as usize;
        let shadow = if cfg!(feature = "cuda") { mla } else { 0 };
        assert_eq!(KvCache::bytes_per_token(&cfg), (mla + 16 + shadow) * 4);

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

    /// Non-hybrid models keep their previous accounting exactly: all layers hold KV and
    /// there is no fixed state. Guards the refactor against changing GLM/M3/M2.7 figures.
    #[test]
    fn uniform_transformer_accounting_is_unchanged() {
        let cfg = cfg_from(
            r#"{"model_type":"minimax_m2","hidden_size":8,"intermediate_size":6,
                "num_hidden_layers":3,"num_attention_heads":4,"num_key_value_heads":2,
                "head_dim":4,"partial_rotary_factor":0.5,"rotary_dim":2,"vocab_size":8,
                "num_local_experts":4,"num_experts_per_tok":2,"shared_intermediate_size":0,
                "scoring_func":"sigmoid","use_routing_bias":true,"hidden_act":"silu",
                "rms_norm_eps":1e-5,"eos_token_id":2}"#,
        );
        assert!(cfg.layer_kind.is_empty(), "uniform archs carry no layer_kind");
        assert_eq!(KvCache::kv_layers(&cfg), cfg.n_layers as usize, "every layer holds KV");
        assert_eq!(KvCache::fixed_bytes(&cfg), 0, "no recurrent state");
        assert_eq!(KvCache::bytes_for(&cfg, 7), KvCache::bytes_per_token(&cfg) * 7);

        // Matches the long-standing formula for a GQA transformer.
        let mla = cfg.kv_lora as usize + cfg.qk_rope as usize;
        let shadow = if cfg!(feature = "cuda") { mla } else { 0 };
        let expect = cfg.n_layers as usize
            * (mla + 2 * cfg.n_kv_heads as usize * cfg.qk_head as usize + shadow)
            * 4;
        assert_eq!(KvCache::bytes_per_token(&cfg), expect);
    }
}
