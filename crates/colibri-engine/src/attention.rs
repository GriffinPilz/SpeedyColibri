//! MLA attention with a compressed KV-cache — port of `attention_rows` from
//! `c/glm.c` (the CPU path).
//!
//! GLM-5.2 uses Multi-head Latent Attention: per token only a normalized latent
//! `[kv_lora]` and a shared rotary key `[qk_rope]` are cached; the per-head
//! `k_nope` and `value` are recovered from the latent through `kv_b`. Two cores
//! compute the same result:
//!
//!   - [`AttnCore::Reconstruct`] — rebuild `k_nope`/`value` for every cached token
//!     via one `kv_b` matmul, then do standard causal attention. The reference.
//!   - [`AttnCore::Absorb`] — DeepSeek weight absorption: fold `W_K` into the
//!     query and apply `W_V` after the softmax, so nothing per-token is
//!     reconstructed (`O(T·kv_lora)` instead of `O(T·H·(nope+vh))`). The decode
//!     fast path.
//!
//! They are algebraically identical; a test asserts they agree numerically.
//!
//! DSA sparse attention: `attention_with` takes an optional per-query selection (the
//! DSA lightning indexer's top-k, computed in [`crate::dsa`]). The reconstruct core
//! then attends only to the selected cached positions. `sel == None` is dense, and
//! selecting *all* positions reproduces the dense output exactly (tested) — so DSA is
//! a no-op for context ≤ `index_topk` and a strict speedup above it.

use crate::linear::{matmul_qt, qt_addrow, qt_matvec_rows};
use crate::math::{rmsnorm_inplace, rope_interleave, rope_neox, softmax};
use crate::model::{KvCache, Layer};
use colibri_cluster::{AttnRequest, ExpertSharding, NodeId, Transport};
use colibri_core::Config;
use std::io;

/// Add `t.elapsed()` to `acc` when `COLI_PROFILE` is on (else near-free). Backs the
/// per-phase attention breakdown.
#[inline]
fn atime(acc: &std::sync::atomic::AtomicU64, t: std::time::Instant) {
    if crate::forward::profile_on() {
        acc.fetch_add(t.elapsed().as_micros() as u64, std::sync::atomic::Ordering::Relaxed);
    }
}

/// Which attention core to use. Both give the same result.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum AttnCore {
    /// Reconstruct k_nope/value from the latent (the reference path).
    Reconstruct,
    /// DeepSeek weight absorption (the decode fast path).
    Absorb,
}

/// DSA sparse attention. **On by default** (`COLI_DSA=0` disables): the GPU sparse
/// kernel (`attention_absorb_sparse`) makes it a strict prefill win at long context —
/// measured at 8192 tok, 2721 s vs dense 3933 s (1.45×, −31%), with attention alone
/// ~1.55× (dense attn is O(n²), sparse ~O(n·topk), so the gap widens with context).
/// Only fires for single-shot prefill (`pos_base == 0`) at or above `dsa_min_prefill()`
/// tokens; decode stays dense (disk-bound anyway), and short prefills stay dense to
/// skip the ~index_topk break-even band (at 3072 tok sparse ≈ dense).
fn dsa_enabled() -> bool {
    static ON: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *ON.get_or_init(|| std::env::var("COLI_DSA").ok().as_deref() != Some("0"))
}

/// Minimum prefill length (new tokens) at which DSA activates. Default 4096 — above the
/// 2048–~3500 break-even band, where GPU-sparse is a clear win and grows with context.
/// Override with `COLI_DSA_MIN` (e.g. `=2048` to fire as soon as it's correct).
fn dsa_min_prefill() -> usize {
    static N: std::sync::OnceLock<usize> = std::sync::OnceLock::new();
    *N.get_or_init(|| {
        std::env::var("COLI_DSA_MIN")
            .ok()
            .and_then(|v| v.parse().ok())
            .unwrap_or(4096)
    })
}

/// `COLI_DSA_CPU=1` forces the CPU `reconstruct_core` even when DSA is on, bypassing
/// the GPU sparse kernel — the correctness A/B (GPU-sparse must match CPU-sparse for
/// the same selection) and a fallback switch.
#[cfg(feature = "cuda")]
fn force_cpu_dsa() -> bool {
    static ON: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *ON.get_or_init(|| std::env::var("COLI_DSA_CPU").ok().as_deref() == Some("1"))
}

/// MLA attention over `S` new tokens `x[S, hidden]` beginning at `pos_base`,
/// writing `out[S, hidden]`. Uses the reconstruction core.
pub fn attention(
    cfg: &Config,
    l: &Layer,
    layer: usize,
    kv: &mut KvCache,
    x: &[f32],
    s_len: usize,
    pos_base: usize,
    out: &mut [f32],
) {
    attention_with(cfg, l, layer, kv, x, s_len, pos_base, out, AttnCore::Reconstruct, None);
}

/// Grouped-query attention (MiniMax-M3, `arch == MinimaxM3`) over `S` new tokens
/// `x[S, hidden]` starting at `pos_base`, writing `out[S, hidden]`. CPU reference:
/// projects q/k/v, applies per-head QK-norm and partial RoPE, appends the full K/V
/// to the cache, then runs causal scaled-dot-product attention with `n_heads` query
/// heads sharing `n_kv_heads` key/value heads (`group = n_heads / n_kv_heads`).
///
/// Head geometry reuses the Config fields: `qk_head` is the head dim, `qk_rope` the
/// rotary sub-dim (RoPE rotates only the first `qk_rope` of each head). NOTE: the
/// RoPE layout (interleaved here) and the exact QK-norm/RoPE order are to be
/// confirmed against the reference at end-to-end validation (task #56).
pub fn attention_gqa(
    cfg: &Config,
    l: &Layer,
    layer: usize,
    kv: &mut KvCache,
    x: &[f32],
    s_len: usize,
    pos_base: usize,
    out: &mut [f32],
) {
    let h = cfg.n_heads as usize;
    let kvh = cfg.n_kv_heads as usize;
    let hd = cfg.qk_head as usize; // head_dim
    let rot = cfg.qk_rope as usize; // rotary sub-dim
    let group = (h / kvh).max(1); // query heads per kv head
    let kv_dim = kvh * hd;
    let eps = cfg.eps;
    let theta = cfg.theta;
    let scale = cfg.attn_scale;
    let q_proj = l.q_proj.as_ref().expect("GQA layer missing q_proj");
    let k_proj = l.k_proj.as_ref().expect("GQA layer missing k_proj");
    let v_proj = l.v_proj.as_ref().expect("GQA layer missing v_proj");

    // ---- 1) project q, k, v (batched over all S rows) ----------------------
    let _tp = std::time::Instant::now();
    let mut q = vec![0f32; s_len * h * hd];
    let mut k = vec![0f32; s_len * kv_dim];
    let mut v = vec![0f32; s_len * kv_dim];
    matmul_qt(&mut q, x, q_proj, s_len);
    matmul_qt(&mut k, x, k_proj, s_len);
    matmul_qt(&mut v, x, v_proj, s_len);
    atime(&crate::forward::ATTN_PROJ_US, _tp);

    // ---- 2) per-head QK-norm + partial RoPE; append K/V to the cache -------
    let _tr = std::time::Instant::now();
    for s in 0..s_len {
        let pos = pos_base + s;
        for hh in 0..h {
            let qs = &mut q[s * h * hd + hh * hd..s * h * hd + hh * hd + hd];
            rmsnorm_inplace(qs, &l.q_norm, eps);
            rope_neox(&mut qs[..rot], pos, rot, theta);
        }
        for hh in 0..kvh {
            let ks = &mut k[s * kv_dim + hh * hd..s * kv_dim + hh * hd + hd];
            rmsnorm_inplace(ks, &l.k_norm, eps);
            rope_neox(&mut ks[..rot], pos, rot, theta);
        }
        kv.k_full_row_mut(layer, pos)
            .copy_from_slice(&k[s * kv_dim..(s + 1) * kv_dim]);
        kv.v_full_row_mut(layer, pos)
            .copy_from_slice(&v[s * kv_dim..(s + 1) * kv_dim]);
    }
    atime(&crate::forward::ATTN_ROPE_US, _tr);

    // ---- 3) causal GQA attention core -------------------------------------
    let _tc = std::time::Instant::now();
    let st0 = kv.kv_start[layer];
    let mut ctx = vec![0f32; s_len * h * hd];
    for s in 0..s_len {
        let pos = pos_base + s;
        let tk = pos + 1; // attend to cached positions [st0, pos]
        let nkeys = tk - st0;
        let krows = kv.k_full_rows(layer, st0, tk);
        let vrows = kv.v_full_rows(layer, st0, tk);
        for hh in 0..h {
            let kvhh = hh / group;
            let qvec = &q[s * h * hd + hh * hd..s * h * hd + hh * hd + hd];
            let mut scores = vec![0f32; nkeys];
            for (ti, sc) in scores.iter_mut().enumerate() {
                let base = ti * kv_dim + kvhh * hd;
                let krow = &krows[base..base + hd];
                let dot: f32 = qvec.iter().zip(krow).map(|(&a, &b)| a * b).sum();
                *sc = dot * scale;
            }
            softmax(&mut scores);
            let cvec = &mut ctx[s * h * hd + hh * hd..s * h * hd + hh * hd + hd];
            for (ti, &sc) in scores.iter().enumerate() {
                let base = ti * kv_dim + kvhh * hd;
                let vrow = &vrows[base..base + hd];
                for (c, &vv) in cvec.iter_mut().zip(vrow) {
                    *c += sc * vv;
                }
            }
        }
    }
    atime(&crate::forward::ATTN_CORE_US, _tc);

    // ---- 4) output projection: ctx[S, n_heads*head_dim] -> out[S, hidden] --
    let _to = std::time::Instant::now();
    matmul_qt(out, &ctx, &l.o, s_len);
    atime(&crate::forward::ATTN_OPROJ_US, _to);
}

/// As [`attention`], but selecting the core explicitly and optionally restricting each
/// query to a DSA sparse selection (`sel[s]` = the cached positions query `s` attends
/// to). `sel == None` (or an empty per-query list) is dense — full causal attention.
///
/// Returns the DSA selection this call *computed* — `Some` only on a FULL indexer
/// layer with DSA active, `None` otherwise — so the forward loop can carry it to the
/// following SHARED layers (which reuse it instead of falling back to dense).
#[allow(clippy::too_many_arguments)]
pub fn attention_with(
    cfg: &Config,
    l: &Layer,
    layer: usize,
    kv: &mut KvCache,
    x: &[f32],
    s_len: usize,
    pos_base: usize,
    out: &mut [f32],
    core: AttnCore,
    sel: Option<&[Vec<u32>]>,
) -> Option<Vec<Vec<u32>>> {
    attention_with_heads(cfg, l, layer, kv, x, s_len, pos_base, out, core, sel, (0, cfg.n_heads as usize))
}

/// As [`attention_with`], but computing only the head slice `heads = (h_start,
/// h_count)` in the attention core and its o-projection, leaving `out` a **partial**
/// sum (the contribution of heads `[h_start, h_start+h_count)`). Summing the partials
/// from a disjoint cover of `0..n_heads` reconstructs full attention — this is the
/// tensor-parallel split point: each node computes its head slice and the driver adds
/// the partials, exactly as [`crate::moe::moe_sharded`] folds expert partials.
///
/// Everything head-independent still runs in full regardless of the slice: the q/kv
/// projections, the RoPE, the KV-cache write (so the driver's cache stays complete for
/// later decode), and the DSA indexer. Only the per-head core + o-proj are restricted.
/// `heads == (0, n_heads)` is exactly [`attention_with`].
#[allow(clippy::too_many_arguments)]
pub fn attention_with_heads(
    cfg: &Config,
    l: &Layer,
    layer: usize,
    kv: &mut KvCache,
    x: &[f32],
    s_len: usize,
    pos_base: usize,
    out: &mut [f32],
    core: AttnCore,
    sel: Option<&[Vec<u32>]>,
    heads: (usize, usize),
) -> Option<Vec<Vec<u32>>> {
    let h = cfg.n_heads as usize;
    let (h0, hc) = heads;
    debug_assert!(h0 + hc <= h && hc > 0, "head slice ({h0},{hc}) out of range for {h} heads");
    // A partial head slice is only honored by the DSA-sparse GPU core and the CPU
    // reconstruct core; the dense GPU/absorb paths compute all H heads, so a partial
    // slice must not take them (guarded below with `full_heads`).
    let full_heads = h0 == 0 && hc == h;
    let qh = cfg.qk_head as usize;
    let qk_nope = cfg.qk_nope as usize;
    let r = cfg.qk_rope as usize;
    let vh = cfg.v_head as usize;
    let kvl = cfg.kv_lora as usize;
    let ql = cfg.q_lora as usize;
    let cw = kvl + r; // comp width
    let eps = cfg.eps;
    let theta = cfg.theta;

    // ---- 1) projections (batched over all S rows; exact kernel) ------------
    let _tp = std::time::Instant::now();
    let mut qr = vec![0f32; s_len * ql];
    matmul_qt(&mut qr, x, &l.q_a, s_len);
    for s in 0..s_len {
        rmsnorm_inplace(&mut qr[s * ql..(s + 1) * ql], &l.q_a_ln, eps);
    }
    let mut q = vec![0f32; s_len * h * qh];
    matmul_qt(&mut q, &qr, &l.q_b, s_len);
    let mut comp = vec![0f32; s_len * cw];
    matmul_qt(&mut comp, x, &l.kv_a, s_len);
    atime(&crate::forward::ATTN_PROJ_US, _tp);

    // ---- 2) RoPE the query rope halves; write the compressed cache ---------
    let _tr = std::time::Instant::now();
    for s in 0..s_len {
        let pos = pos_base + s;
        for hh in 0..h {
            let base = s * h * qh + hh * qh + qk_nope;
            rope_interleave(&mut q[base..base + r], pos, r, theta);
        }
        // normalized latent
        let latent_src_end = s * cw + kvl;
        {
            let ldst = kv.latent_row_mut(layer, pos);
            ldst.copy_from_slice(&comp[s * cw..latent_src_end]);
            rmsnorm_inplace(ldst, &l.kv_a_ln, eps);
        }
        // roped k_rot (shared across heads)
        {
            let rdst = kv.krot_row_mut(layer, pos);
            rdst.copy_from_slice(&comp[latent_src_end..latent_src_end + r]);
            rope_interleave(rdst, pos, r, theta);
        }
    }

    atime(&crate::forward::ATTN_ROPE_US, _tr);

    let st0 = kv.kv_start[layer];

    let _ti = std::time::Instant::now();
    let dsa_selection = indexer_select(cfg, l, layer, x, &qr, s_len, pos_base, st0, sel.is_some());
    atime(&crate::forward::ATTN_INDEX_US, _ti);
    let sel = sel.or(dsa_selection.as_deref());

    let mut ctx = vec![0f32; s_len * h * vh];

    let _tc = std::time::Instant::now();
    // GPU weight-absorption attention core for resident kv_b (falls back to CPU).
    // A DSA selection (long-context prefill) uses the sparse kernel; dense uses the
    // batch/decode kernels. Anything the GPU declines falls to the CPU cores below.
    #[cfg(feature = "cuda")]
    let ran_gpu = {
        let tk = pos_base + s_len;
        match sel {
            // DSA sparse prefill: reconstruct core restricted to the indexer selection.
            // DSA runs at st0 == 0 (so the selection's positions are latent-relative);
            // the Absorb decode core never carries a selection.
            Some(sels)
                if matches!(core, AttnCore::Reconstruct)
                    && st0 == 0
                    && !force_cpu_dsa()
                    && crate::gpu::available()
                    && l.kv_b.gpu_eligible =>
            {
                crate::gpu::try_attention_absorb_sparse(
                    &l.kv_b,
                    &mut ctx,
                    &q,
                    kv.latent_rows(layer, st0, tk),
                    kv.krot_rows(layer, st0, tk),
                    sels,
                    cfg.index_topk as usize,
                    h0,
                    hc,
                    s_len,
                    h,
                    qk_nope,
                    r,
                    vh,
                    kvl,
                    tk - st0,
                    cfg.attn_scale,
                )
            }
            // Selection active but GPU ineligible (or Absorb core) → CPU reconstruct.
            Some(_) => false,
            // Dense GPU paths compute all H heads, so only take them for the full slice;
            // a partial slice falls through to the head-range CPU reconstruct core.
            None if full_heads
                && s_len == 1
                && st0 == 0
                && crate::gpu::available()
                && l.kv_b.gpu_eligible =>
            {
                // Decode: persistent device KV — append the new row, read on device.
                match kv.sync_device(layer, pos_base, tk) {
                    Some((lat_dev, rope_dev)) => crate::gpu::try_attention_absorb_kvdev(
                        &l.kv_b, &mut ctx, &q, lat_dev, rope_dev, h, qk_nope, r, vh, kvl, tk, cfg.attn_scale,
                    ),
                    None => false,
                }
            }
            None if full_heads && crate::gpu::available() && l.kv_b.gpu_eligible => {
                // Prefill (S>1) or st0>0: one-time host upload of the cache slice.
                crate::gpu::try_attention_absorb(
                    &l.kv_b,
                    &mut ctx,
                    &q,
                    kv.latent_rows(layer, st0, tk),
                    kv.krot_rows(layer, st0, tk),
                    s_len,
                    h,
                    qk_nope,
                    r,
                    vh,
                    kvl,
                    tk - st0,
                    cfg.attn_scale,
                )
            }
            None => false,
        }
    };
    #[cfg(not(feature = "cuda"))]
    let ran_gpu = false;

    if !ran_gpu {
        match core {
            AttnCore::Reconstruct => {
                reconstruct_core(cfg, l, layer, kv, &q, s_len, pos_base, st0, &mut ctx, sel, h0, hc);
            }
            // Absorb is the S==1 decode core; DSA sparsifies the long-context prefill
            // (reconstruct), so a selection is not applied here. It has no head-slice
            // form (decode isn't tensor-parallel here) — a partial slice must use
            // Reconstruct.
            AttnCore::Absorb => {
                debug_assert!(full_heads, "Absorb core has no head-slice form; use Reconstruct");
                absorb_core(cfg, l, layer, kv, &q, s_len, pos_base, st0, &mut ctx);
            }
        }
    }
    atime(&crate::forward::ATTN_CORE_US, _tc);

    // ---- 4) output projection ----------------------------------------------
    let _to = std::time::Instant::now();
    matmul_qt(out, &ctx, &l.o, s_len);
    atime(&crate::forward::ATTN_OPROJ_US, _to);

    // Hand the caller the selection computed here (Some only on a FULL indexer layer
    // with DSA active) so it can be reused by the following SHARED layers instead of
    // them falling back to dense O(n²) attention.
    dsa_selection
}

/// The DSA lightning-indexer selection for this layer, or `None` when DSA is inactive
/// here. On a FULL indexer layer, once the context exceeds `index_topk`, it selects the
/// top-k keys per query so attention restricts to those. Gated to single-shot prefill
/// (`pos_base == 0`/`st0 == 0`): decode has no keys for prior positions and stays
/// dense. `have_sel` short-circuits when a selection was already supplied (SHARED-layer
/// reuse, or tests). Shared by [`attention_with_heads`] and [`dsa_selection_for`] so
/// both gate identically.
#[allow(clippy::too_many_arguments)]
fn indexer_select(
    cfg: &Config,
    l: &Layer,
    layer: usize,
    x: &[f32],
    qr: &[f32],
    s_len: usize,
    pos_base: usize,
    st0: usize,
    have_sel: bool,
) -> Option<Vec<Vec<u32>>> {
    if have_sel
        || st0 != 0
        || pos_base != 0
        || l.ix_wk.is_none()
        || !cfg.idx_type.get(layer).copied().unwrap_or(false)
        || pos_base + s_len <= cfg.index_topk as usize
        || pos_base + s_len < dsa_min_prefill()
        || !dsa_enabled()
    {
        return None;
    }
    let iw = crate::dsa::IndexerWeights {
        wk: l.ix_wk.as_ref().unwrap(),
        knorm_w: &l.ix_knorm_w,
        knorm_b: &l.ix_knorm_b,
        wq: l.ix_wq.as_ref().unwrap(),
        wp: l.ix_wp.as_ref().unwrap(),
    };
    Some(crate::dsa::indexer_forward(
        &iw,
        x,
        qr,
        s_len,
        cfg.index_nh as usize,
        cfg.index_hd as usize,
        cfg.index_topk as usize,
        cfg.qk_rope as usize,
        cfg.theta,
        pos_base,
    ))
}

/// Compute this layer's DSA selection standalone — the tensor-parallel driver needs it
/// *before* the attention core so it can ship the identical selection to every peer
/// (no node runs its own indexer, so nothing can diverge). Recomputes the cheap
/// normalized `q_a` projection the indexer consumes. Single-shot prefill only
/// (`pos_base == 0`, cache start 0); returns `None` on non-FULL/short/decode/DSA-off
/// layers, exactly as [`attention_with`] would.
pub fn dsa_selection_for(
    cfg: &Config,
    l: &Layer,
    layer: usize,
    x: &[f32],
    s_len: usize,
    pos_base: usize,
) -> Option<Vec<Vec<u32>>> {
    let ql = cfg.q_lora as usize;
    let mut qr = vec![0f32; s_len * ql];
    matmul_qt(&mut qr, x, &l.q_a, s_len);
    for s in 0..s_len {
        rmsnorm_inplace(&mut qr[s * ql..(s + 1) * ql], &l.q_a_ln, cfg.eps);
    }
    indexer_select(cfg, l, layer, x, &qr, s_len, pos_base, 0, false)
}

/// Contiguous, balanced head split for tensor-parallel attention: node `k` of
/// `num_nodes` computes `(h_start, h_count)` of `n_heads`. Remainder heads go to the
/// lowest-numbered nodes, so counts differ by at most one and the ranges exactly cover
/// `0..n_heads`. Every node derives the same split; the driver also sends each peer its
/// `(h_start, h_count)` explicitly.
pub fn head_slice(node: u32, num_nodes: u32, n_heads: usize) -> (usize, usize) {
    let n = (num_nodes as usize).max(1);
    let k = node as usize;
    let base = n_heads / n;
    let rem = n_heads % n;
    let start = k * base + k.min(rem);
    let count = base + if k < rem { 1 } else { 0 };
    (start, count)
}

/// Tensor-parallel MLA attention: split the heads across nodes so every box's GPU runs
/// part of the (dominant) attention core, then sum the o-projected partials — the
/// attention analogue of [`crate::moe::moe_sharded`]. This node computes its head slice
/// locally (which also writes its full, head-independent KV cache for later decode)
/// while each peer computes its slice over the shipped activations + the identical DSA
/// selection; partials fold in ascending node order.
///
/// Prefill only (`pos_base == 0`), Reconstruct/DSA core. `incoming_sel` is the carried
/// selection for SHARED layers (`None` on a FULL layer → computed here); the selection
/// this call establishes is returned for the caller to carry, as [`attention_with`]
/// does. On a single node this is never invoked (the caller gates on `num_nodes > 1`).
#[allow(clippy::too_many_arguments)]
pub fn attention_sharded<T: Transport + ?Sized>(
    cfg: &Config,
    l: &Layer,
    layer: usize,
    kv: &mut KvCache,
    x: &[f32],
    s_len: usize,
    pos_base: usize,
    out: &mut [f32],
    incoming_sel: Option<&[Vec<u32>]>,
    sharding: &ExpertSharding,
    transport: &T,
) -> io::Result<Option<Vec<Vec<u32>>>> {
    let d = cfg.hidden as usize;
    let h = cfg.n_heads as usize;
    let num_nodes = sharding.num_nodes();
    let me = transport.this_node();

    // One selection for the whole cluster this layer: carried (SHARED) or freshly
    // computed here (FULL). Shipped verbatim to peers so no node runs its own indexer.
    let computed = if incoming_sel.is_none() {
        dsa_selection_for(cfg, l, layer, x, s_len, pos_base)
    } else {
        None
    };
    let sel: Option<&[Vec<u32>]> = incoming_sel.or(computed.as_deref());
    // Ship a length-`s_len` selection ALWAYS: the real per-query selection when sparse,
    // or `s_len` empty per-query vecs when dense. This is the cluster's authoritative
    // decision — because it is `Some` and length `s_len`, every node's
    // `attention_with_heads` short-circuits its own indexer (`have_sel`) and treats
    // empty per-query entries as dense. Both the driver-local slice and the peers then
    // run the *same* sparse kernel (dense-mode when empty), so the per-head partials are
    // consistent and sum cleanly. A `None` would instead let a node re-run the indexer
    // against the wrong layer and diverge.
    let sel_full: Vec<Vec<u32>> = match sel {
        Some(s) => s.to_vec(),
        None => vec![Vec::new(); s_len],
    };

    let (h0, hc) = head_slice(me.0, num_nodes, h);
    for v in out.iter_mut() {
        *v = 0.0;
    }

    let mut partials: Vec<(u32, Vec<f32>)> = Vec::with_capacity(num_nodes as usize);
    let mut err: Option<io::Error> = None;
    std::thread::scope(|scope| {
        // Peers: each computes its own head slice over the same activations + selection.
        let handles: Vec<(u32, _)> = (0..num_nodes)
            .filter(|&n| NodeId(n) != me)
            .map(|n| {
                let (ph0, phc) = head_slice(n, num_nodes, h);
                let req = AttnRequest {
                    activations: x.to_vec(),
                    sel: sel_full.clone(),
                    n_tokens: s_len,
                    hidden: d,
                    pos_base: pos_base as u32,
                    h_start: ph0 as u32,
                    h_count: phc as u32,
                    layer: layer as u32,
                };
                let handle = scope.spawn(move || transport.exchange_attn(NodeId(n), &req));
                (n, handle)
            })
            .collect();

        // Local head slice computes while the peer requests are in flight. This also
        // writes the full (head-independent) KV cache so the driver can decode later.
        // Uses `sel_full` (not `sel`) so it takes the identical sparse-kernel path as
        // the peers.
        let mut local = vec![0f32; s_len * d];
        attention_with_heads(
            cfg, l, layer, kv, x, s_len, pos_base, &mut local, AttnCore::Reconstruct,
            Some(&sel_full), (h0, hc),
        );
        partials.push((me.0, local));

        for (n, handle) in handles {
            match handle.join() {
                Ok(Ok(resp)) if resp.outputs.len() == s_len * d => partials.push((n, resp.outputs)),
                Ok(Ok(resp)) => {
                    err.get_or_insert_with(|| {
                        io::Error::new(
                            io::ErrorKind::InvalidData,
                            format!(
                                "node {n}: attention expected {} outputs, got {}",
                                s_len * d,
                                resp.outputs.len()
                            ),
                        )
                    });
                }
                Ok(Err(e)) => {
                    err.get_or_insert_with(|| io::Error::new(io::ErrorKind::Other, e.to_string()));
                }
                Err(_) => {
                    err.get_or_insert_with(|| {
                        io::Error::new(
                            io::ErrorKind::Other,
                            format!("node {n}: attention exchange thread panicked"),
                        )
                    });
                }
            }
        }
    });
    if let Some(e) = err {
        return Err(e);
    }
    // Fold in ascending node order → deterministic accumulation across runs.
    partials.sort_by_key(|(n, _)| *n);
    for (_, p) in &partials {
        for (o, v) in out.iter_mut().zip(p.iter()) {
            *o += *v;
        }
    }
    Ok(computed)
}

/// Peer-side handler for an [`AttnRequest`]: compute this node's head slice
/// `[h0, h0+hc)` of MLA attention over layer input `x`, into `out` (`[s_len, hidden]`),
/// using the driver-supplied `sel` (never re-running the indexer). Stateless — builds a
/// fresh single-layer KV from `x` for single-shot prefill, so no persistent cache is
/// needed. `l` is the real layer's resident weights; the scratch KV is indexed as
/// layer 0. `sel` must be length `s_len` (the driver ships that invariant, empty
/// per-query = dense); anything else is treated as fully dense.
#[allow(clippy::too_many_arguments)]
pub fn compute_attention_partial(
    cfg: &Config,
    l: &Layer,
    x: &[f32],
    s_len: usize,
    pos_base: usize,
    h0: usize,
    hc: usize,
    sel: &[Vec<u32>],
    out: &mut [f32],
) {
    let kvl = cfg.kv_lora as usize;
    let r = cfg.qk_rope as usize;
    let mut kv = KvCache::new(1, kvl, r, s_len);
    // Always pass a length-`s_len` Some so `attention_with_heads` never re-runs the
    // indexer against the wrong (scratch layer 0) index — the driver's decision is
    // authoritative. A mis-sized ship degrades to dense rather than diverging.
    let dense;
    let sel_ref: &[Vec<u32>] = if sel.len() == s_len {
        sel
    } else {
        dense = vec![Vec::new(); s_len];
        &dense
    };
    attention_with_heads(
        cfg, l, 0, &mut kv, x, s_len, pos_base, out, AttnCore::Reconstruct, Some(sel_ref), (h0, hc),
    );
}

/// Reconstruction core: rebuild k_nope/value for all cached tokens via one kv_b
/// matmul, then causal attention. Port of `attention_rows` step 2/3.
#[allow(clippy::too_many_arguments)]
fn reconstruct_core(
    cfg: &Config,
    l: &Layer,
    layer: usize,
    kv: &KvCache,
    q: &[f32],
    s_len: usize,
    pos_base: usize,
    st0: usize,
    ctx: &mut [f32],
    sel: Option<&[Vec<u32>]>,
    h0: usize,
    hc: usize,
) {
    let h = cfg.n_heads as usize;
    let qh = cfg.qk_head as usize;
    let qk_nope = cfg.qk_nope as usize;
    let r = cfg.qk_rope as usize;
    let vh = cfg.v_head as usize;
    let scale = cfg.attn_scale;
    let head_kv = qk_nope + vh;
    let kvb_dim = h * head_kv;
    let tk = pos_base + s_len;
    let nrec = tk - st0;

    // kvb_all[t-st0] = kv_b(latent_t) -> [nrec, kvb_dim]
    let mut kvb_all = vec![0f32; nrec * kvb_dim];
    matmul_qt(&mut kvb_all, kv.latent_rows(layer, st0, tk), &l.kv_b, nrec);

    for s in 0..s_len {
        let pos = pos_base + s;
        let nt = pos + 1 - st0;
        // The cached positions (as `jj = t - st0`) this query attends to. DSA sparse
        // attention restricts to the indexer's selection; dense (None, or an empty
        // selection = the DSA no-op) attends to all — and the two must agree when the
        // selection is all positions (the `is_dense` invariant, tested below).
        let jjs: Vec<usize> = match sel {
            Some(sels) if !sels[s].is_empty() => {
                sels[s].iter().map(|&t| t as usize - st0).collect()
            }
            _ => (0..nt).collect(),
        };
        // Only this node's head slice; ctx columns for other heads stay zero (ctx is
        // pre-zeroed), so the o-projection over the full ctx yields this slice's partial.
        for hh in h0..h0 + hc {
            let qbase = s * h * qh + hh * qh;
            let (qnope, qrope) = q[qbase..qbase + qh].split_at(qk_nope);
            let mut sc = vec![0f32; jjs.len()];
            for (k, &jj) in jjs.iter().enumerate() {
                let t = st0 + jj;
                let kn_off = (t - st0) * kvb_dim + hh * head_kv;
                let kn = &kvb_all[kn_off..kn_off + qk_nope];
                let kr = kv.krot_row(layer, t);
                let mut a = 0f32;
                for i in 0..qk_nope {
                    a += qnope[i] * kn[i];
                }
                for d in 0..r {
                    a += qrope[d] * kr[d];
                }
                sc[k] = a * scale;
            }
            softmax(&mut sc);
            let cx = &mut ctx[(s * h + hh) * vh..(s * h + hh) * vh + vh];
            for (k, &a) in sc.iter().enumerate() {
                let t = st0 + jjs[k];
                let vv_off = (t - st0) * kvb_dim + hh * head_kv + qk_nope;
                let vv = &kvb_all[vv_off..vv_off + vh];
                for d in 0..vh {
                    cx[d] += a * vv[d];
                }
            }
        }
    }
}

/// Weight-absorption core: fold W_K into the query, apply W_V after softmax.
/// Port of `attention_rows`'s absorb branch.
#[allow(clippy::too_many_arguments)]
fn absorb_core(
    cfg: &Config,
    l: &Layer,
    layer: usize,
    kv: &KvCache,
    q: &[f32],
    s_len: usize,
    pos_base: usize,
    st0: usize,
    ctx: &mut [f32],
) {
    let h = cfg.n_heads as usize;
    let qh = cfg.qk_head as usize;
    let qk_nope = cfg.qk_nope as usize;
    let r = cfg.qk_rope as usize;
    let vh = cfg.v_head as usize;
    let kvl = cfg.kv_lora as usize;
    let scale = cfg.attn_scale;
    let head_kv = qk_nope + vh;

    for s in 0..s_len {
        let pos = pos_base + s;
        for hh in 0..h {
            let qbase = s * h * qh + hh * qh;
            let (qnope, qrope) = q[qbase..qbase + qh].split_at(qk_nope);
            let rbase = hh * head_kv;
            // qabs = W_K^h^T q_nope  (a [kv_lora] vector)
            let mut qabs = vec![0f32; kvl];
            for (d, &qn) in qnope.iter().enumerate() {
                qt_addrow(&l.kv_b, rbase + d, qn, &mut qabs);
            }
            let nt = pos + 1 - st0;
            let mut sc = vec![0f32; nt];
            for (jj, sc_jj) in sc.iter_mut().enumerate() {
                let t = st0 + jj;
                let lt = kv.latent_row(layer, t);
                let kr = kv.krot_row(layer, t);
                let mut a = 0f32;
                for i in 0..kvl {
                    a += qabs[i] * lt[i];
                }
                for d in 0..r {
                    a += qrope[d] * kr[d];
                }
                *sc_jj = a * scale;
            }
            softmax(&mut sc);
            // clat = Σ_t a_t L_t
            let mut clat = vec![0f32; kvl];
            for (jj, &a) in sc.iter().enumerate() {
                let lt = kv.latent_row(layer, st0 + jj);
                for i in 0..kvl {
                    clat[i] += a * lt[i];
                }
            }
            // ctx^h = W_V^h clat
            let cx = &mut ctx[(s * h + hh) * vh..(s * h + hh) * vh + vh];
            qt_matvec_rows(&l.kv_b, rbase + qk_nope, vh, &clat, cx);
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::quantize::qtensor_from_f32;
    use colibri_core::QTensor;

    // Deterministic small "random" weights.
    fn weights(o: usize, i: usize, seed: usize) -> QTensor {
        let w: Vec<f32> = (0..o * i)
            .map(|k| (((k * 7 + seed * 13 + 3) % 11) as f32 - 5.0) * 0.1)
            .collect();
        qtensor_from_f32(&w, o, i, 16) // f32 (exact)
    }

    fn vecf(n: usize, seed: usize) -> Vec<f32> {
        (0..n)
            .map(|k| (((k * 5 + seed * 3 + 1) % 7) as f32 - 3.0) * 0.1 + 1.0)
            .collect()
    }

    // A tiny but structurally faithful attention config.
    fn cfg() -> Config {
        let json = colibri_json::Json::parse(
            r#"{"hidden_size":6,"num_hidden_layers":1,"num_attention_heads":2,
                "n_routed_experts":4,"num_experts_per_tok":2,"moe_intermediate_size":4,
                "intermediate_size":6,"first_k_dense_replace":0,"q_lora_rank":4,
                "kv_lora_rank":4,"qk_nope_head_dim":3,"qk_rope_head_dim":2,"v_head_dim":3,
                "n_shared_experts":1,"vocab_size":10,"n_group":1,"topk_group":1,
                "rms_norm_eps":1e-5,"routed_scaling_factor":1.0,
                "rope_parameters":{"rope_theta":10000.0},"eos_token_id":[9],
                "index_topk":0,"index_n_heads":0,"index_head_dim":0}"#,
        )
        .unwrap();
        Config::from_json(&json).unwrap()
    }

    fn make_layer(c: &Config) -> Layer {
        let h = c.n_heads as usize;
        let d = c.hidden as usize;
        let mut l = Layer::default();
        l.q_a = weights(c.q_lora as usize, d, 1);
        l.q_a_ln = vec![1.0; c.q_lora as usize];
        l.q_b = weights(h * c.qk_head as usize, c.q_lora as usize, 2);
        l.kv_a = weights((c.kv_lora + c.qk_rope) as usize, d, 3);
        l.kv_a_ln = vec![1.0; c.kv_lora as usize];
        l.kv_b = weights(h * (c.qk_nope + c.v_head) as usize, c.kv_lora as usize, 4);
        l.o = weights(d, h * c.v_head as usize, 5);
        l
    }

    fn gqa_cfg() -> Config {
        let json = colibri_json::Json::parse(
            r#"{"model_type":"minimax_m3_vl","text_config":{
                "hidden_size":8,"intermediate_size":6,"num_hidden_layers":1,
                "num_attention_heads":4,"num_key_value_heads":2,"head_dim":4,
                "vocab_size":16,"max_position_embeddings":128,"rms_norm_eps":1e-6,
                "use_gemma_norm":true,"rope_theta":10000.0,"rotary_dim":2,
                "hidden_act":"swigluoai","use_qk_norm":true,"dense_intermediate_size":8,
                "shared_intermediate_size":6,"num_local_experts":4,"num_experts_per_tok":2,
                "n_shared_experts":1,"scoring_func":"sigmoid","use_routing_bias":true,
                "moe_layer_freq":[1],"swiglu_alpha":1.702,"swiglu_limit":7.0,
                "routed_scaling_factor":2.0,"eos_token_id":[15]}}"#,
        )
        .unwrap();
        Config::from_json(&json).unwrap()
    }

    fn make_gqa_layer(c: &Config) -> Layer {
        let h = c.n_heads as usize;
        let kvh = c.n_kv_heads as usize;
        let hd = c.qk_head as usize;
        let d = c.hidden as usize;
        let mut l = Layer::default();
        l.q_proj = Some(weights(h * hd, d, 1));
        l.k_proj = Some(weights(kvh * hd, d, 2));
        l.v_proj = Some(weights(kvh * hd, d, 3));
        l.o = weights(d, h * hd, 4);
        l.q_norm = vec![1.0; hd];
        l.k_norm = vec![1.0; hd];
        l
    }

    #[test]
    fn gqa_attention_is_causal() {
        let c = gqa_cfg();
        let l = make_gqa_layer(&c);
        let d = c.hidden as usize;
        let mk_kv = || {
            let mut kv = KvCache::new(1, c.kv_lora as usize, c.qk_rope as usize, 8);
            kv.enable_gqa(c.n_kv_heads as usize * c.qk_head as usize);
            kv
        };
        // Two-token prefill.
        let x2 = vecf(2 * d, 7);
        let mut kv2 = mk_kv();
        let mut out2 = vec![0f32; 2 * d];
        attention_gqa(&c, &l, 0, &mut kv2, &x2, 2, 0, &mut out2);
        // One-token prefill of just the first token, fresh cache.
        let mut kv1 = mk_kv();
        let mut out1 = vec![0f32; d];
        attention_gqa(&c, &l, 0, &mut kv1, &x2[..d], 1, 0, &mut out1);
        // Causality: position 0's output must not depend on position 1's presence.
        for j in 0..d {
            assert!(
                (out2[j] - out1[j]).abs() < 1e-5,
                "row0 differs at {j}: {} vs {}",
                out2[j],
                out1[j]
            );
        }
        assert!(out2.iter().all(|v| v.is_finite()));
    }

    // GPU vs CPU MLA absorb core at GLM dims (H=64, kv_lora=512) over a 2048-token
    // context. `cargo test -p colibri-engine --features cuda --release -- --ignored
    // --nocapture bench_attention`
    // The DSA correctness gate: the GPU sparse kernel must reproduce the CPU
    // reconstruct_core for the *same* per-query selection (they are the same math in
    // different fp order — absorb vs reconstruct). Compares `ctx` directly rather than
    // end-to-end tokens, which are too sensitive to fp order under heavy sparsity.
    // Run: `cargo test -p colibri-engine --features cuda --release -- --ignored
    // dsa_sparse_gpu_matches_cpu --nocapture`
    #[cfg(feature = "cuda")]
    #[test]
    #[ignore]
    fn dsa_sparse_gpu_matches_cpu_reconstruct() {
        if !crate::gpu::available() {
            eprintln!("skip: no CUDA device");
            return;
        }
        // GLM attention dims — real sizes so the kernel path is representative.
        let json = colibri_json::Json::parse(
            r#"{"hidden_size":6144,"num_hidden_layers":1,"num_attention_heads":64,
                "n_routed_experts":256,"num_experts_per_tok":8,"moe_intermediate_size":2048,
                "intermediate_size":12288,"first_k_dense_replace":0,"q_lora_rank":2048,
                "kv_lora_rank":512,"qk_nope_head_dim":128,"qk_rope_head_dim":64,"v_head_dim":128,
                "n_shared_experts":1,"vocab_size":2000,"n_group":1,"topk_group":1,
                "rms_norm_eps":1e-5,"routed_scaling_factor":1.0,
                "rope_parameters":{"rope_theta":10000.0},"eos_token_id":[1],
                "index_topk":4,"index_n_heads":0,"index_head_dim":0}"#,
        )
        .unwrap();
        let cfg = Config::from_json(&json).unwrap();
        let (h, qk_nope, r, vh, kvl) = (64usize, 128usize, 64usize, 128usize, 512usize);
        let kvb_dim = h * (qk_nope + vh);
        let wf: Vec<f32> = (0..kvb_dim * kvl).map(|k| ((k % 13) as f32 - 6.0) * 0.01).collect();
        let mut kv_b = qtensor_from_f32(&wf, kvb_dim, kvl, 8); // int8, like production
        kv_b.gpu_eligible = true;
        let mut l = Layer::default();
        l.kv_b = kv_b;

        // Single-shot prefill: s_len queries, context == s_len (all positions new).
        let s_len = 12usize;
        let t = s_len;
        let index_topk = 4usize;
        let mut kv = KvCache::new(1, kvl, r, t);
        for pos in 0..t {
            for (i, x) in kv.latent_row_mut(0, pos).iter_mut().enumerate() {
                *x = (((pos * 7 + i) % 17) as f32 - 8.0) * 0.02;
            }
            for (i, x) in kv.krot_row_mut(0, pos).iter_mut().enumerate() {
                *x = (((pos * 5 + i) % 11) as f32 - 5.0) * 0.02;
            }
        }
        let q: Vec<f32> =
            (0..s_len * h * (qk_nope + r)).map(|k| ((k % 7) as f32 - 3.0) * 0.01).collect();
        let latent = kv.latent_rows(0, 0, t).to_vec();
        let rope = kv.krot_rows(0, 0, t).to_vec();
        let scale = cfg.attn_scale;

        // A DSA-shaped selection: dense (empty) while nk <= index_topk, else the last
        // `index_topk` causal positions — a genuine strict subset for later queries.
        let sel: Vec<Vec<u32>> = (0..s_len)
            .map(|s| {
                let nk = s + 1;
                if nk <= index_topk {
                    Vec::new()
                } else {
                    ((nk - index_topk) as u32..nk as u32).collect()
                }
            })
            .collect();

        let mut ctx_gpu = vec![0f32; s_len * h * vh];
        let ok = crate::gpu::try_attention_absorb_sparse(
            &l.kv_b, &mut ctx_gpu, &q, &latent, &rope, &sel, index_topk, 0, h, s_len, h, qk_nope, r,
            vh, kvl, t, scale,
        );
        assert!(ok, "GPU sparse kernel must run when a device is present");

        let mut ctx_cpu = vec![0f32; s_len * h * vh];
        reconstruct_core(&cfg, &l, 0, &kv, &q, s_len, 0, 0, &mut ctx_cpu, Some(&sel), 0, h);

        // Tensor-parallel invariant: the GPU sparse kernel over two disjoint head
        // slices [0,h/2) + [h/2,h) must reproduce the full-head kernel exactly (each
        // slice zeroes the others' ctx columns), so summing head-parallel partials is
        // faithful. Same device, same math → bit-identical.
        let half = h / 2;
        let mut ctx_lo = vec![0f32; s_len * h * vh];
        let mut ctx_hi = vec![0f32; s_len * h * vh];
        assert!(crate::gpu::try_attention_absorb_sparse(
            &l.kv_b, &mut ctx_lo, &q, &latent, &rope, &sel, index_topk, 0, half, s_len, h, qk_nope,
            r, vh, kvl, t, scale,
        ));
        assert!(crate::gpu::try_attention_absorb_sparse(
            &l.kv_b, &mut ctx_hi, &q, &latent, &rope, &sel, index_topk, half, h - half, s_len, h,
            qk_nope, r, vh, kvl, t, scale,
        ));
        let split_err = (0..ctx_gpu.len())
            .map(|i| (ctx_gpu[i] - (ctx_lo[i] + ctx_hi[i])).abs())
            .fold(0.0f32, f32::max);
        eprintln!("head-split GPU sparse vs full: maxerr = {split_err:.2e}");
        // Exact: if either slice wrote outside its heads, the sum would double-count and
        // this would be non-zero.
        assert!(split_err == 0.0, "head-slice sum must equal full attention exactly; err={split_err:.3e}");

        let maxerr =
            ctx_gpu.iter().zip(&ctx_cpu).map(|(a, b)| (a - b).abs()).fold(0.0f32, f32::max);
        eprintln!("dsa sparse GPU vs CPU reconstruct: maxerr = {maxerr:.2e}");
        assert!(ctx_cpu.iter().any(|v| v.abs() > 1e-6), "output must be non-trivial");
        assert!(maxerr < 5e-3, "GPU sparse must match CPU reconstruct; maxerr={maxerr:.3e}");
    }

    #[cfg(feature = "cuda")]
    #[test]
    #[ignore]
    fn bench_attention_gpu_vs_cpu() {
        if !crate::gpu::available() {
            eprintln!("skip: no CUDA device");
            return;
        }
        let json = colibri_json::Json::parse(
            r#"{"hidden_size":6144,"num_hidden_layers":1,"num_attention_heads":64,
                "n_routed_experts":256,"num_experts_per_tok":8,"moe_intermediate_size":2048,
                "intermediate_size":12288,"first_k_dense_replace":0,"q_lora_rank":2048,
                "kv_lora_rank":512,"qk_nope_head_dim":128,"qk_rope_head_dim":64,"v_head_dim":128,
                "n_shared_experts":1,"vocab_size":2000,"n_group":1,"topk_group":1,
                "rms_norm_eps":1e-5,"routed_scaling_factor":1.0,
                "rope_parameters":{"rope_theta":10000.0},"eos_token_id":[1],
                "index_topk":0,"index_n_heads":0,"index_head_dim":0}"#,
        )
        .unwrap();
        let cfg = Config::from_json(&json).unwrap();
        let (h, qk_nope, r, vh, kvl) = (64usize, 128usize, 64usize, 128usize, 512usize);
        let kvb_dim = h * (qk_nope + vh);
        let wf: Vec<f32> = (0..kvb_dim * kvl).map(|k| ((k % 13) as f32 - 6.0) * 0.01).collect();
        let mut kv_b = qtensor_from_f32(&wf, kvb_dim, kvl, 4);
        kv_b.gpu_eligible = true;
        let mut l = Layer::default();
        l.kv_b = kv_b;

        let t = 4096usize;
        let mut kv = KvCache::new(1, kvl, r, t);
        for pos in 0..t {
            for x in kv.latent_row_mut(0, pos).iter_mut() {
                *x = 0.01;
            }
            for x in kv.krot_row_mut(0, pos).iter_mut() {
                *x = 0.01;
            }
        }
        let q: Vec<f32> = (0..h * (qk_nope + r)).map(|k| ((k % 7) as f32 - 3.0) * 0.01).collect();
        let latent = kv.latent_rows(0, 0, t).to_vec();
        let rope = kv.krot_rows(0, 0, t).to_vec();
        let scale = cfg.attn_scale;
        let mut cg = vec![0f32; h * vh];
        let mut cc = vec![0f32; h * vh];

        // correctness + warm
        crate::gpu::try_attention_absorb(&l.kv_b, &mut cg, &q, &latent, &rope, 1, h, qk_nope, r, vh, kvl, t, scale);
        absorb_core(&cfg, &l, 0, &kv, &q, 1, t - 1, 0, &mut cc);
        let maxerr = cg.iter().zip(&cc).map(|(a, b)| (a - b).abs()).fold(0.0f32, f32::max);

        // persistent device KV: sync once, then the kernel reads device memory
        let mut dev = crate::gpu::DeviceKv::new(1, t);
        let (lat_dev, rope_dev) = dev.sync(0, &latent, &rope, kvl, r, 0, t).unwrap();

        let iters: u64 = std::env::var("COLI_BENCH_ITERS")
            .ok()
            .and_then(|s| s.parse().ok())
            .unwrap_or(500);
        let tm = std::time::Instant::now();
        for _ in 0..iters {
            crate::gpu::try_attention_absorb(&l.kv_b, &mut cg, &q, &latent, &rope, 1, h, qk_nope, r, vh, kvl, t, scale);
        }
        let gpu_host = tm.elapsed().as_secs_f64();
        let tm = std::time::Instant::now();
        for _ in 0..iters {
            crate::gpu::try_attention_absorb_kvdev(&l.kv_b, &mut cg, &q, lat_dev, rope_dev, h, qk_nope, r, vh, kvl, t, scale);
        }
        let gpu_dev = tm.elapsed().as_secs_f64();
        let tm = std::time::Instant::now();
        for _ in 0..iters {
            absorb_core(&cfg, &l, 0, &kv, &q, 1, t - 1, 0, &mut cc);
        }
        let cpu = tm.elapsed().as_secs_f64();
        eprintln!(
            "attention absorb (H={h} T={t}) x{iters}: GPU-hostKV {:.0} us | GPU-deviceKV {:.0} us | CPU-NEON {:.0} us | deviceKV {:.2}x hostKV, {:.1}x CPU | max|Δ|={maxerr:.1e}",
            gpu_host / iters as f64 * 1e6,
            gpu_dev / iters as f64 * 1e6,
            cpu / iters as f64 * 1e6,
            gpu_host / gpu_dev,
            cpu / gpu_dev,
        );
    }

    #[test]
    fn reconstruct_and_absorb_agree() {
        let c = cfg();
        let l = make_layer(&c);
        let d = c.hidden as usize;
        let s_len = 3;
        let x = vecf(s_len * d, 9);

        let mut kv_a = KvCache::new(1, c.kv_lora as usize, c.qk_rope as usize, 16);
        let mut kv_b = KvCache::new(1, c.kv_lora as usize, c.qk_rope as usize, 16);
        let mut out_recon = vec![0f32; s_len * d];
        let mut out_absorb = vec![0f32; s_len * d];

        attention_with(&c, &l, 0, &mut kv_a, &x, s_len, 0, &mut out_recon, AttnCore::Reconstruct, None);
        attention_with(&c, &l, 0, &mut kv_b, &x, s_len, 0, &mut out_absorb, AttnCore::Absorb, None);

        for (a, b) in out_recon.iter().zip(&out_absorb) {
            assert!((a - b).abs() < 1e-4, "reconstruct {a} vs absorb {b}");
        }
        // sanity: not all zero
        assert!(out_recon.iter().any(|v| v.abs() > 1e-6));
    }

    #[test]
    fn dsa_select_all_equals_dense() {
        // THE DSA correctness gate (the C's DSA_FORCE): selecting *every* cached
        // position must reproduce the exact dense attention output — proving the sparse
        // core is a faithful restriction of full attention, not a different computation.
        let c = cfg();
        let l = make_layer(&c);
        let d = c.hidden as usize;
        let s_len = 4;
        let x = vecf(s_len * d, 9);

        let mut kv_dense = KvCache::new(1, c.kv_lora as usize, c.qk_rope as usize, 16);
        let mut kv_sparse = KvCache::new(1, c.kv_lora as usize, c.qk_rope as usize, 16);
        let mut out_dense = vec![0f32; s_len * d];
        let mut out_sparse = vec![0f32; s_len * d];

        // sel[s] = every causal position 0..=s (pos_base=0): exactly the dense set.
        let sel: Vec<Vec<u32>> = (0..s_len).map(|s| (0..=s as u32).collect()).collect();

        attention_with(&c, &l, 0, &mut kv_dense, &x, s_len, 0, &mut out_dense, AttnCore::Reconstruct, None);
        attention_with(&c, &l, 0, &mut kv_sparse, &x, s_len, 0, &mut out_sparse, AttnCore::Reconstruct, Some(&sel));

        for (a, b) in out_dense.iter().zip(&out_sparse) {
            assert!((a - b).abs() < 1e-6, "dense {a} vs select-all {b}");
        }
        assert!(out_dense.iter().any(|v| v.abs() > 1e-6));
    }

    #[test]
    fn dsa_subset_changes_output() {
        // A strict subset must differ from dense — otherwise the sparse path isn't
        // actually sparsifying. Query s attends only to position 0 here; for s>0 that
        // drops keys it would otherwise see, so the output must change.
        let c = cfg();
        let l = make_layer(&c);
        let d = c.hidden as usize;
        let s_len = 4;
        let x = vecf(s_len * d, 9);

        let mut kv_d = KvCache::new(1, c.kv_lora as usize, c.qk_rope as usize, 16);
        let mut kv_s = KvCache::new(1, c.kv_lora as usize, c.qk_rope as usize, 16);
        let mut out_d = vec![0f32; s_len * d];
        let mut out_s = vec![0f32; s_len * d];

        let sel: Vec<Vec<u32>> = (0..s_len).map(|_| vec![0u32]).collect(); // attend only to pos 0

        attention_with(&c, &l, 0, &mut kv_d, &x, s_len, 0, &mut out_d, AttnCore::Reconstruct, None);
        attention_with(&c, &l, 0, &mut kv_s, &x, s_len, 0, &mut out_s, AttnCore::Reconstruct, Some(&sel));

        let differ = out_d.iter().zip(&out_s).any(|(a, b)| (a - b).abs() > 1e-5);
        assert!(differ, "a strict subset selection must change the attention output");
    }

    #[test]
    fn dsa_full_layer_returns_selection_and_reuse_reproduces_it() {
        // The contract the SHARED-layer reuse relies on: a FULL indexer layer
        // (sel == None, indexer present, context > index_topk) COMPUTES and RETURNS
        // its selection; feeding that same selection back reproduces the identical
        // sparse output and does NOT recompute (returns None). That is exactly what
        // `layer_forward` does — carry a full layer's returned selection to the
        // shared layers after it.
        use crate::quantize::qtensor_from_f32;
        // DSA is on by default now, but the min-prefill gate (4096) would keep it dense
        // for this tiny `s_len=4` fixture, so drop the threshold to 2 (> index_topk).
        // Setting COLI_DSA explicitly too keeps the test independent of the default.
        // Only this test reaches `dsa_enabled()`/`dsa_min_prefill()` — the condition
        // checks `ix_wk.is_some()` first and no other test builds an indexer layer — so
        // their OnceLocks aren't cached wrong by a sibling test.
        std::env::set_var("COLI_DSA", "1");
        std::env::set_var("COLI_DSA_MIN", "2");
        let json = colibri_json::Json::parse(
            r#"{"hidden_size":6,"num_hidden_layers":1,"num_attention_heads":2,
                "n_routed_experts":4,"num_experts_per_tok":2,"moe_intermediate_size":4,
                "intermediate_size":6,"first_k_dense_replace":0,"q_lora_rank":4,
                "kv_lora_rank":4,"qk_nope_head_dim":3,"qk_rope_head_dim":2,"v_head_dim":3,
                "n_shared_experts":1,"vocab_size":10,"n_group":1,"topk_group":1,
                "rms_norm_eps":1e-5,"routed_scaling_factor":1.0,
                "rope_parameters":{"rope_theta":10000.0},"eos_token_id":[9],
                "index_topk":2,"index_n_heads":2,"index_head_dim":4,
                "indexer_types":["full"]}"#,
        )
        .unwrap();
        let c = Config::from_json(&json).unwrap();
        assert_eq!(c.idx_type, vec![true], "one FULL indexer layer");

        let mut l = make_layer(&c);
        let (hidden, ihd, nh, ql) =
            (c.hidden as usize, c.index_hd as usize, c.index_nh as usize, c.q_lora as usize);
        l.ix_wk = Some(qtensor_from_f32(&vecf(ihd * hidden, 1), ihd, hidden, 16));
        l.ix_wq = Some(qtensor_from_f32(&vecf(nh * ihd * ql, 2), nh * ihd, ql, 16));
        l.ix_wp = Some(qtensor_from_f32(&vecf(nh * hidden, 3), nh, hidden, 16));
        l.ix_knorm_w = vec![1.0; ihd];
        l.ix_knorm_b = vec![0.0; ihd];

        let d = hidden;
        let s_len = 4; // > index_topk (2) → DSA active
        let x = vecf(s_len * d, 9);

        // FULL layer: no incoming selection → it computes one and returns it.
        let mut kv1 = KvCache::new(1, c.kv_lora as usize, c.qk_rope as usize, 16);
        let mut o1 = vec![0f32; s_len * d];
        let sel = attention_with(&c, &l, 0, &mut kv1, &x, s_len, 0, &mut o1, AttnCore::Reconstruct, None)
            .expect("a FULL indexer layer must compute+return a selection past index_topk");
        // It is a genuine sparse restriction: the last query keeps at most index_topk keys.
        assert!(sel[s_len - 1].len() <= c.index_topk as usize && !sel[s_len - 1].is_empty());

        // SHARED layer: reuse the carried selection → must NOT recompute, and must
        // reproduce the full layer's sparse output byte-for-byte.
        let mut kv2 = KvCache::new(1, c.kv_lora as usize, c.qk_rope as usize, 16);
        let mut o2 = vec![0f32; s_len * d];
        let reused =
            attention_with(&c, &l, 0, &mut kv2, &x, s_len, 0, &mut o2, AttnCore::Reconstruct, Some(&sel));
        assert!(reused.is_none(), "a supplied selection must not be recomputed");
        for (a, b) in o1.iter().zip(&o2) {
            assert!((a - b).abs() < 1e-6, "reused selection must reproduce the sparse output: {a} vs {b}");
        }
    }

    #[test]
    fn head_slices_sum_to_full_cpu() {
        // The tensor-parallel invariant on the CPU reconstruct core: summing the
        // per-head-slice partials over a disjoint cover of 0..n_heads reproduces full
        // attention (o-proj is linear in ctx, and disjoint slices zero each other's ctx
        // columns). This is what `attention_sharded` relies on when the driver adds the
        // peers' head-slice partials.
        let c = cfg();
        let l = make_layer(&c);
        let d = c.hidden as usize;
        let hh = c.n_heads as usize; // 2 in this fixture
        let s_len = 3;
        let x = vecf(s_len * d, 9);

        let mut kv_full = KvCache::new(1, c.kv_lora as usize, c.qk_rope as usize, 16);
        let mut out_full = vec![0f32; s_len * d];
        attention_with_heads(
            &c, &l, 0, &mut kv_full, &x, s_len, 0, &mut out_full, AttnCore::Reconstruct, None,
            (0, hh),
        );

        // Two disjoint slices [0,1) + [1,2), each a fresh partial, summed.
        let mut out_sum = vec![0f32; s_len * d];
        for (h0, hc) in [(0usize, 1usize), (1usize, 1usize)] {
            let mut kv = KvCache::new(1, c.kv_lora as usize, c.qk_rope as usize, 16);
            let mut part = vec![0f32; s_len * d];
            attention_with_heads(
                &c, &l, 0, &mut kv, &x, s_len, 0, &mut part, AttnCore::Reconstruct, None, (h0, hc),
            );
            for (o, &p) in out_sum.iter_mut().zip(part.iter()) {
                *o += p;
            }
        }
        for (a, b) in out_full.iter().zip(&out_sum) {
            assert!((a - b).abs() < 1e-5, "head-slice partial sum {b} != full {a}");
        }
        assert!(out_full.iter().any(|v| v.abs() > 1e-6));
    }

    #[test]
    fn attention_sharded_two_nodes_equals_single_node() {
        // Tensor-parallel attention must reproduce single-node attention: node 0
        // computes head 0, node 1 computes head 1 (served over a real TCP loopback whose
        // handler runs `compute_attention_partial`), and the driver sums the o-projected
        // partials. Exercises the head split, the AttnRequest/Response wire path, the
        // shipped (dense here — this fixture has no indexer) selection, and the fold.
        use colibri_cluster::{serve_cluster, AttnResponse, ExpertRequest, ExpertResponse, TcpTransport};
        use std::collections::HashMap;
        use std::sync::Arc;

        let c = cfg(); // 2 heads, hidden 6
        let l = make_layer(&c);
        let d = c.hidden as usize;
        let s_len = 3;
        let x = vecf(s_len * d, 9);

        // Reference: single-node attention (both heads together).
        let mut kv_ref = KvCache::new(1, c.kv_lora as usize, c.qk_rope as usize, 16);
        let mut out_single = vec![0f32; s_len * d];
        attention_with(
            &c, &l, 0, &mut kv_ref, &x, s_len, 0, &mut out_single, AttnCore::Reconstruct, None,
        );

        let sharding = ExpertSharding::new(2, c.n_experts as u32);
        // Node 1's server. `make_layer` is deterministic, so this copy is identical to
        // the driver's `l` (Layer isn't Clone). Expert handler is unused here.
        let ch = c.clone();
        let lh = Arc::new(make_layer(&c));
        let addr = serve_cluster(
            "127.0.0.1:0".parse().unwrap(),
            sharding.fingerprint(),
            |req: &ExpertRequest| ExpertResponse {
                outputs: vec![0.0; req.n_tokens * req.hidden],
                n_tokens: req.n_tokens,
                hidden: req.hidden,
            },
            move |req: &AttnRequest| {
                let mut outputs = vec![0.0f32; req.n_tokens * req.hidden];
                compute_attention_partial(
                    &ch,
                    &lh,
                    &req.activations,
                    req.n_tokens,
                    req.pos_base as usize,
                    req.h_start as usize,
                    req.h_count as usize,
                    &req.sel,
                    &mut outputs,
                );
                AttnResponse { outputs, n_tokens: req.n_tokens, hidden: req.hidden }
            },
        )
        .unwrap();

        let mut peers = HashMap::new();
        peers.insert(NodeId(1), addr);
        let transport = TcpTransport::new(NodeId(0), peers, sharding.fingerprint());

        let mut kv_shard = KvCache::new(1, c.kv_lora as usize, c.qk_rope as usize, 16);
        let mut out_sharded = vec![0f32; s_len * d];
        attention_sharded(
            &c, &l, 0, &mut kv_shard, &x, s_len, 0, &mut out_sharded, None, &sharding, &transport,
        )
        .unwrap();

        for i in 0..s_len * d {
            assert!(
                (out_single[i] - out_sharded[i]).abs() < 1e-5,
                "mismatch at {i}: single {} vs sharded {}",
                out_single[i],
                out_sharded[i]
            );
        }
        assert!(out_single.iter().any(|v| v.abs() > 1e-6));
    }

    #[test]
    fn single_token_is_value_of_itself() {
        // With one token, softmax over one score is 1.0, so ctx = W_V(L_0) and
        // both cores must produce identical output (no averaging).
        let c = cfg();
        let l = make_layer(&c);
        let d = c.hidden as usize;
        let x = vecf(d, 4);

        let mut kv1 = KvCache::new(1, c.kv_lora as usize, c.qk_rope as usize, 4);
        let mut kv2 = KvCache::new(1, c.kv_lora as usize, c.qk_rope as usize, 4);
        let mut o1 = vec![0f32; d];
        let mut o2 = vec![0f32; d];
        attention_with(&c, &l, 0, &mut kv1, &x, 1, 0, &mut o1, AttnCore::Reconstruct, None);
        attention_with(&c, &l, 0, &mut kv2, &x, 1, 0, &mut o2, AttnCore::Absorb, None);
        for (a, b) in o1.iter().zip(&o2) {
            assert!((a - b).abs() < 1e-4);
        }
    }

    #[test]
    fn incremental_decode_matches_batched_prefill() {
        // Feeding 3 tokens at once (prefill) must match feeding them one at a
        // time (decode) — the KV-cache carries the context across calls.
        let c = cfg();
        let l = make_layer(&c);
        let d = c.hidden as usize;
        let x = vecf(3 * d, 9);

        let mut kv_batch = KvCache::new(1, c.kv_lora as usize, c.qk_rope as usize, 16);
        let mut out_batch = vec![0f32; 3 * d];
        attention(&c, &l, 0, &mut kv_batch, &x, 3, 0, &mut out_batch);

        let mut kv_step = KvCache::new(1, c.kv_lora as usize, c.qk_rope as usize, 16);
        let mut out_step = vec![0f32; 3 * d];
        for s in 0..3 {
            let mut o = vec![0f32; d];
            attention(&c, &l, 0, &mut kv_step, &x[s * d..(s + 1) * d], 1, s, &mut o);
            out_step[s * d..(s + 1) * d].copy_from_slice(&o);
        }
        for (a, b) in out_batch.iter().zip(&out_step) {
            assert!((a - b).abs() < 1e-4, "prefill {a} vs decode {b}");
        }
    }
}
