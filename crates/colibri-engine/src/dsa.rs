//! DSA — DeepSeek Sparse Attention, the GLM-5.2 "lightning indexer".
//!
//! Port of the `dsa_sel` selection block in the reference `glm.c`. At long context
//! the O(n²) attention is the wall (measured: prefill ~0.27 s/token, dominated by
//! attention as n grows). DSA cuts it to O(n·k): a lightweight indexer scores every
//! cached key against the query and the main attention only attends to the top
//! `index_topk` positions.
//!
//! # The indexer
//!
//! Per query position, with `nh` indexer heads of width `hd` and a single-headed
//! (MQA-style — cheap) indexer key per cached position:
//!
//! ```text
//! score(t) = (1/√nh) · Σ_h  w[h] · ReLU( (1/√hd) · (q_h · k_idx_t) )
//! ```
//!
//! `q_h` is indexer-query head `h`, `k_idx_t` is the single indexer key at position
//! `t` (shared across heads), `w[h]` is a per-head weight (from a tiny `x·ix_wp`
//! projection). The ReLU is applied to each head's scaled dot **before** weighting,
//! exactly as the C does — this is not a standard softmax attention.
//!
//! # Selection and the dense invariant
//!
//! `keep = min(nk, index_topk)`. When `nk <= index_topk` the indexer is a **no-op**:
//! every key is selected, and sparse attention over "all positions" must reproduce
//! the exact dense output. That invariant is the correctness gate the C calls
//! `DSA_FORCE`, and it is unit-tested below — it needs no model weights.

/// One query position's indexer score against one cached indexer key.
///
/// `q_heads` is `[nh * hd]` (row-major per head), `key` is `[hd]` (single head,
/// shared across all `nh`), `head_w` is `[nh]`. Mirrors the C inner loop precisely,
/// including ReLU-before-weight and the two scale factors.
pub fn indexer_score(q_heads: &[f32], key: &[f32], head_w: &[f32], nh: usize, hd: usize) -> f32 {
    debug_assert_eq!(q_heads.len(), nh * hd);
    debug_assert_eq!(key.len(), hd);
    debug_assert_eq!(head_w.len(), nh);
    let rs = 1.0f32 / (hd as f32).sqrt();
    let wsc = 1.0f32 / (nh as f32).sqrt();
    let mut a = 0.0f32;
    for h in 0..nh {
        let qh = &q_heads[h * hd..(h + 1) * hd];
        let mut d0 = 0.0f32;
        for i in 0..hd {
            d0 += qh[i] * key[i];
        }
        d0 *= rs;
        if d0 > 0.0 {
            a += head_w[h] * d0; // ReLU on the score, then weight
        }
    }
    a * wsc
}

/// Which cached positions the main attention should attend to for one query.
///
/// `scores[t]` is the indexer score for cached position `t` (t in `0..nk`).
/// Returns the selected position indices **in ascending position order** (attention
/// consumes them left-to-right). Semantics ported verbatim from `glm.c`:
///
/// - `keep = min(nk, index_topk)`.
/// - threshold = the `keep`-th largest score; take positions with `score > thr`
///   first, then `score == thr`, both in position order, until `keep` are chosen.
///
/// When `nk <= index_topk`, `keep == nk` and every position is returned — the
/// **dense no-op** that must equal full attention (see [`is_dense`]).
pub fn select_topk(scores: &[f32], index_topk: usize) -> Vec<u32> {
    let nk = scores.len();
    if nk == 0 {
        return Vec::new();
    }
    let keep = nk.min(index_topk);
    if keep == nk {
        // Dense no-op: all positions. Cheaper than sorting, and exact.
        return (0..nk as u32).collect();
    }
    // Threshold = keep-th largest. Sort a copy descending; ties resolved by taking
    // strictly-greater first, then equal-to-threshold, in position order — matches
    // the C's qsort-desc + two-pass scan and keeps selection deterministic.
    let mut sorted: Vec<f32> = scores.to_vec();
    sorted.sort_unstable_by(|a, b| b.partial_cmp(a).unwrap_or(std::cmp::Ordering::Equal));
    let thr = sorted[keep - 1];
    let mut sel: Vec<u32> = Vec::with_capacity(keep);
    for (t, &s) in scores.iter().enumerate() {
        if sel.len() >= keep {
            break;
        }
        if s > thr {
            sel.push(t as u32);
        }
    }
    for (t, &s) in scores.iter().enumerate() {
        if sel.len() >= keep {
            break;
        }
        if s == thr {
            sel.push(t as u32);
        }
    }
    // The C leaves these in `>thr` then `==thr` order; attention sums over the *set*
    // (softmax normalization + weighted sum are order-independent), so sort ascending
    // for a deterministic, left-to-right selection that matches the dense case's 0..nk.
    sel.sort_unstable();
    sel
}

/// Whether DSA is a no-op for this context length — every key selected, so the
/// engine must run the dense path (or sparse-over-all, which is identical).
#[inline]
pub fn is_dense(nk: usize, index_topk: usize) -> bool {
    nk <= index_topk
}

/// The per-layer lightning-indexer weights (a FULL indexer layer; SHARED layers reuse
/// the previous FULL layer's selection). Tensor names in the checkpoint:
/// `self_attn.indexer.{wk, k_norm, wq_b, weights_proj}`.
pub struct IndexerWeights<'a> {
    /// key projection `hidden -> index_hd` (`indexer.wk`)
    pub wk: &'a colibri_core::quant::QTensor,
    /// key LayerNorm weight + bias (`indexer.k_norm`), eps 1e-6
    pub knorm_w: &'a [f32],
    pub knorm_b: &'a [f32],
    /// query projection `q_lora -> nh*index_hd` (`indexer.wq_b`)
    pub wq: &'a colibri_core::quant::QTensor,
    /// per-head weight projection `hidden -> nh` (`indexer.weights_proj`)
    pub wp: &'a colibri_core::quant::QTensor,
}

/// Run the lightning indexer over a prefill batch and return, per query, the cached
/// positions the main attention should attend to (empty = dense no-op). Port of the
/// `idx_type[layer]` FULL block in `glm.c`, for `pos_base .. pos_base+s_len` with the
/// cache starting at 0 (the long-context prefill case DSA targets).
///
/// For each new token the indexer key is `rope(layernorm(x·wk), pos)` — RoPE on the
/// first `qk_rope` dims only, exactly as the C. For each query the indexer query is
/// `rope(q_lora·wq)` per head and the head weights are `x·wp`; positions are scored
/// with [`indexer_score`] and [`select_topk`] picks the top `index_topk`.
#[allow(clippy::too_many_arguments)]
pub fn indexer_forward(
    w: &IndexerWeights,
    x: &[f32],       // [s_len, hidden]
    q_lora: &[f32],  // [s_len, q_lora_dim]  (the q_a-normed query, `QR` in the C)
    s_len: usize,
    nh: usize,
    index_hd: usize,
    index_topk: usize,
    qk_rope: usize,
    theta: f32,
    pos_base: usize,
) -> Vec<Vec<u32>> {
    use crate::linear::matmul_qt;
    use crate::math::{layernorm, rope_interleave};

    let q_lora_dim = w.wq.i as usize;
    let hidden = w.wk.i as usize;
    let rope = qk_rope.min(index_hd);

    // 1) indexer keys for the new tokens: k[s] = rope(layernorm(x[s]·wk), pos).
    let mut keys = vec![0f32; s_len * index_hd];
    matmul_qt(&mut keys, x, w.wk, s_len);
    for s in 0..s_len {
        let pos = pos_base + s;
        let k = &mut keys[s * index_hd..(s + 1) * index_hd];
        layernorm(k, w.knorm_w, w.knorm_b, 1e-6);
        rope_interleave(&mut k[..rope], pos, rope, theta);
    }

    // 2) per query: indexer query + head weights, score every causal key, select.
    // This is the DSA hot loop — O(s_len · nk · nh · hd). It is embarrassingly
    // parallel (each query writes only its own `sel[s]` and reads shared keys/x/
    // q_lora), so split it across cores. Serial, it single-threads the whole indexer
    // while the rest of the engine is on the GPU, which made DSA net-*slower* than
    // dense despite the sparse-attention savings.
    let mut sel = vec![Vec::new(); s_len];
    let nthreads = std::thread::available_parallelism().map_or(1, |n| n.get());
    let chunk = s_len.div_ceil(nthreads).max(1);
    let keys = &keys;
    std::thread::scope(|scope| {
        for (ci, sel_chunk) in sel.chunks_mut(chunk).enumerate() {
            let base = ci * chunk;
            scope.spawn(move || {
                for (j, out) in sel_chunk.iter_mut().enumerate() {
                    let s = base + j;
                    let pos = pos_base + s;
                    let nk = pos + 1; // causal, cache starts at 0
                    if is_dense(nk, index_topk) {
                        continue; // no-op: attention attends to all (empty selection)
                    }
                    let mut qi = vec![0f32; nh * index_hd];
                    matmul_qt(&mut qi, &q_lora[s * q_lora_dim..(s + 1) * q_lora_dim], w.wq, 1);
                    for h in 0..nh {
                        rope_interleave(&mut qi[h * index_hd..h * index_hd + rope], pos, rope, theta);
                    }
                    let mut hw = vec![0f32; nh];
                    matmul_qt(&mut hw, &x[s * hidden..(s + 1) * hidden], w.wp, 1);

                    let mut scores = vec![0f32; nk];
                    for (t, sc) in scores.iter_mut().enumerate() {
                        let kt = &keys[t * index_hd..(t + 1) * index_hd];
                        *sc = indexer_score(&qi, kt, &hw, nh, index_hd);
                    }
                    *out = select_topk(&scores, index_topk);
                }
            });
        }
    });
    sel
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn score_relu_and_scaling_match_the_formula() {
        // nh=2, hd=2. head 0 aligned with key (positive dot), head 1 anti-aligned
        // (negative dot → ReLU zeroes it, so its weight must not contribute).
        let key = [1.0, 0.0];
        let q = [2.0, 0.0, /*h0*/ -3.0, 0.0 /*h1*/];
        let w = [1.0, 5.0];
        let (nh, hd) = (2, 2);
        let rs = 1.0 / (hd as f32).sqrt();
        let wsc = 1.0 / (nh as f32).sqrt();
        // h0: d0 = (2*1)*rs = 2*rs > 0 → contributes w0 * 2*rs
        // h1: d0 = (-3*1)*rs < 0 → ReLU zeroes it
        let expected = (1.0 * (2.0 * rs)) * wsc;
        let got = indexer_score(&q, &key, &w, nh, hd);
        assert!((got - expected).abs() < 1e-6, "got {got}, want {expected}");
    }

    #[test]
    fn select_dense_when_context_fits() {
        // nk <= index_topk → dense no-op: every position, in order. THE invariant.
        let scores = [0.9, 0.1, 0.5, 0.3];
        let sel = select_topk(&scores, 8);
        assert_eq!(sel, vec![0, 1, 2, 3], "must select all when nk <= index_topk");
        assert!(is_dense(scores.len(), 8));
        assert!(!is_dense(scores.len(), 2));
    }

    #[test]
    fn select_topk_picks_highest_in_position_order() {
        // scores by position: [0.9, 0.1, 0.5, 0.3, 0.8], index_topk=3.
        // top-3 values are 0.9, 0.8, 0.5 at positions 0, 4, 2. Returned in POSITION
        // order → [0, 2, 4].
        let scores = [0.9, 0.1, 0.5, 0.3, 0.8];
        assert_eq!(select_topk(&scores, 3), vec![0, 2, 4]);
    }

    #[test]
    fn select_topk_breaks_ties_by_position() {
        // threshold value appears at several positions; keep must be respected and
        // ties taken left-to-right. scores [0.5,0.9,0.5,0.5], k=2: top is 0.9 (pos 1),
        // then thr=0.5 → first tie at pos 0. → [0, 1] in position order.
        let scores = [0.5, 0.9, 0.5, 0.5];
        let sel = select_topk(&scores, 2);
        assert_eq!(sel.len(), 2);
        assert_eq!(sel, vec![0, 1]);
    }

    #[test]
    fn select_never_exceeds_keep() {
        // Many equal scores, k=3: exactly 3 selected, the first 3 positions.
        let scores = [1.0; 10];
        let sel = select_topk(&scores, 3);
        assert_eq!(sel, vec![0, 1, 2]);
    }

    fn vecf(n: usize, seed: u64) -> Vec<f32> {
        // deterministic pseudo-random in [-1, 1), no rng dependency
        (0..n)
            .map(|i| {
                let z = (i as u64).wrapping_mul(2654435761).wrapping_add(seed.wrapping_mul(40503));
                ((z % 2000) as f32 / 1000.0) - 1.0
            })
            .collect()
    }

    #[test]
    fn indexer_forward_selects_and_respects_dense() {
        use crate::quantize::qtensor_from_f32;
        let (hidden, index_hd, nh, q_lora_dim) = (8usize, 4usize, 2usize, 6usize);
        let (index_topk, qk_rope, s_len) = (2usize, 2usize, 4usize);
        // f32 (bits=16) synthetic weights → exact matmul.
        let wk = qtensor_from_f32(&vecf(index_hd * hidden, 1), index_hd, hidden, 16);
        let wq = qtensor_from_f32(&vecf(nh * index_hd * q_lora_dim, 2), nh * index_hd, q_lora_dim, 16);
        let wp = qtensor_from_f32(&vecf(nh * hidden, 3), nh, hidden, 16);
        let knw = vec![1.0f32; index_hd];
        let knb = vec![0.0f32; index_hd];
        let w = IndexerWeights { wk: &wk, knorm_w: &knw, knorm_b: &knb, wq: &wq, wp: &wp };
        let x = vecf(s_len * hidden, 7);
        let ql = vecf(s_len * q_lora_dim, 8);

        let sel = indexer_forward(&w, &x, &ql, s_len, nh, index_hd, index_topk, qk_rope, 10000.0, 0);

        // queries 0,1 have nk=1,2 <= index_topk=2 → dense no-op (empty selection).
        assert!(sel[0].is_empty() && sel[1].is_empty(), "context <= index_topk must be dense");
        // queries 2,3 have nk=3,4 > 2 → keep = min(nk, index_topk) = 2 positions.
        assert_eq!(sel[2].len(), 2);
        assert_eq!(sel[3].len(), 2);
        // every selected position is a valid causal index, ascending.
        for (s, sl) in sel.iter().enumerate() {
            assert!(sl.windows(2).all(|w| w[0] < w[1]), "selection must be ascending");
            for &t in sl {
                assert!((t as usize) <= s, "selected pos {t} not causal for query {s}");
            }
        }
    }
}
