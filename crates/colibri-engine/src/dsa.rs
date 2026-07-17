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
}
