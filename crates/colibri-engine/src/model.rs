//! Model and per-layer weight structures — port of the `Model`, `Layer`,
//! `ESlot`, and `KVState` structs from `c/glm.c`.
//!
//! # Status: SKELETON
//!
//! The field layout mirrors the C structs so the loader and forward pass can be
//! filled in against a known shape. Buffers that are hot-path detail (profiling
//! counters, GPU shadow caches) are elided until their subsystem is ported.

use colibri_core::{Config, QTensor};
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

impl KvCache {
    /// Allocate a cache for `n_rows` layer rows holding up to `max_t` tokens.
    ///
    /// Prefer [`KvCache::for_model`], which sizes the rows (including the MTP
    /// head's extra row) from the model itself.
    pub fn new(n_rows: usize, kv_lora: usize, qk_rope: usize, max_t: usize) -> KvCache {
        KvCache {
            max_t,
            kv_lora,
            qk_rope,
            latent: vec![vec![0.0; max_t * kv_lora]; n_rows],
            k_rot: vec![vec![0.0; max_t * qk_rope]; n_rows],
            kv_dim: 0,
            k_full: vec![Vec::new(); n_rows],
            v_full: vec![Vec::new(); n_rows],
            kv_start: vec![0; n_rows],
            #[cfg(feature = "cuda")]
            dev: None,
        }
    }

    /// Enable the GQA full-KV cache (MiniMax-M3): allocate per-layer key/value
    /// buffers of width `kv_dim = n_kv_heads * head_dim`. No-op for the MLA path.
    pub(crate) fn enable_gqa(&mut self, kv_dim: usize) {
        self.kv_dim = kv_dim;
        let rows = self.k_full.len();
        self.k_full = vec![vec![0.0; self.max_t * kv_dim]; rows];
        self.v_full = vec![vec![0.0; self.max_t * kv_dim]; rows];
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
        if model.cfg.arch == colibri_core::Arch::MinimaxM3 {
            kv.enable_gqa(model.cfg.n_kv_heads as usize * model.cfg.qk_head as usize);
        }
        kv
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
