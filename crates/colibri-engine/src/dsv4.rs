//! DeepSeek-V4-specific MoE pieces: the `sqrtsoftplus`/`noaux_tc` router and the clamped
//! SwiGLU used by its experts.
//!
//! Both are ported from the checkpoint's `inference/model.py` (`Gate.forward`,
//! `Expert.forward`) and pinned against vectors generated from that source.
//!
//! These live apart from the generic MoE code because V4 is the first arch here whose
//! scorer is neither `softmax` nor `sigmoid` — the two the shared `sigmoid_route` flag
//! selects between. Routing it through that flag would silently pick the wrong scorer, so
//! [`Arch::DeepseekV4`] leaves the flag false and calls these instead.

/// Numerically stable `softplus(z) = ln(1 + e^z)`.
///
/// Written in the `max(z,0) + ln1p(e^-|z|)` form because the naive version overflows for
/// large positive logits — and V4's scorer takes a square root of this, so an `inf` here
/// becomes an `inf` routing weight and a NaN expert output rather than a loud failure.
#[inline]
fn softplus(z: f32) -> f32 {
    z.max(0.0) + (-z.abs()).exp().ln_1p()
}

/// V4's expert score: `sqrt(softplus(logit))`.
#[inline]
pub fn sqrt_softplus(z: f32) -> f32 {
    softplus(z).sqrt()
}

/// Route one token: pick `topk` experts and return their (index, weight) pairs.
///
/// `logits` are the raw `gate.weight @ x` scores, `bias` the per-expert selection bias
/// (`noaux_tc`), empty for the hash-routed layers which supply indices directly.
///
/// **The bias shifts SELECTION only — never the weights.** The reference computes
/// `original_scores` before adding the bias and gathers the weights from those. Using the
/// biased scores for the weights too is a silent error: routing still "works", every shape
/// matches, and the model is simply wrong. That asymmetry is the whole point of `noaux_tc`,
/// and it is what the test pins.
///
/// Weights are then renormalised over the chosen experts and scaled by `route_scale`
/// (`routed_scaling_factor`, 1.5 for V4). Renormalisation is skipped for a softmax scorer
/// upstream; V4 is not one, so it always applies here.
pub fn route_topk(
    logits: &[f32],
    bias: &[f32],
    topk: usize,
    route_scale: f32,
    out: &mut Vec<(u32, f32)>,
) {
    out.clear();
    let n = logits.len();
    let topk = topk.min(n);
    if topk == 0 {
        return;
    }
    // Unbiased scores drive the WEIGHTS.
    let mut scores = Vec::with_capacity(n);
    for &z in logits {
        scores.push(sqrt_softplus(z));
    }
    // Biased scores drive the SELECTION.
    let mut order: Vec<u32> = (0..n as u32).collect();
    if bias.is_empty() {
        order.sort_unstable_by(|&a, &b| {
            scores[b as usize]
                .partial_cmp(&scores[a as usize])
                .unwrap_or(std::cmp::Ordering::Equal)
                .then(a.cmp(&b))
        });
    } else {
        debug_assert_eq!(bias.len(), n);
        order.sort_unstable_by(|&a, &b| {
            let (x, y) = (
                scores[a as usize] + bias[a as usize],
                scores[b as usize] + bias[b as usize],
            );
            y.partial_cmp(&x)
                .unwrap_or(std::cmp::Ordering::Equal)
                .then(a.cmp(&b))
        });
    }
    let mut sum = 0.0f32;
    for &e in order.iter().take(topk) {
        let w = scores[e as usize];
        sum += w;
        out.push((e, w));
    }
    let inv = if sum > 0.0 { route_scale / sum } else { 0.0 };
    for (_, w) in out.iter_mut() {
        *w *= inv;
    }
    out.sort_unstable_by_key(|&(e, _)| e);
}

/// Hash routing: the expert set comes from `tid2eid[token_id]`, not from the scores.
///
/// V4's first `num_hash_layers` (3) layers replace top-k SELECTION with a fixed
/// token-id -> expert-id table. **They still run the router matmul**, because it supplies
/// the WEIGHTS — `scores.gather(1, indices)` on the same `sqrt_softplus` scores every
/// other layer uses, then normalised and scaled by `route_scale`. Skipping the matmul on
/// these layers because "routing is a lookup" would leave the six chosen experts with no
/// weights at all.
///
/// These layers ship **no bias** (`bias = None`, not a zero bias). That is consistent:
/// the bias exists only to shift selection, and selection here is not a comparison.
///
/// Duplicate entries are merged by SUMMING their weights. Applying expert `e` twice with
/// weight `w` equals applying it once with `2w`, and the downstream union expects each
/// expert once.
pub fn route_hash(logits: &[f32], eids: &[u32], route_scale: f32, out: &mut Vec<(u32, f32)>) {
    out.clear();
    if eids.is_empty() {
        return;
    }
    let n = logits.len();
    let mut sum = 0.0f32;
    for &e in eids {
        let e = e as usize;
        debug_assert!(e < n, "tid2eid entry {e} past {n} experts");
        if e >= n {
            continue;
        }
        let w = sqrt_softplus(logits[e]);
        sum += w;
        match out.iter_mut().find(|(x, _)| *x == e as u32) {
            Some((_, acc)) => *acc += w,
            None => out.push((e as u32, w)),
        }
    }
    let inv = if sum > 0.0 { route_scale / sum } else { 0.0 };
    for (_, w) in out.iter_mut() {
        *w *= inv;
    }
    out.sort_unstable_by_key(|&(e, _)| e);
}

/// V4's clamped SwiGLU: `silu(min(gate, limit)) * clamp(up, -limit, limit)`.
///
/// **The clamp is asymmetric and that is not a typo.** The reference clamps `up` on BOTH
/// sides but `gate` only from above, because `gate` goes through `silu`, which already
/// bounds it below (`silu(z) -> 0` as `z -> -inf`) — a lower clamp there would change the
/// function, not protect it. Clamping both sides of `gate` is a plausible-looking edit that
/// silently alters every expert's output.
///
/// `limit <= 0` disables clamping, matching the reference's `if self.swiglu_limit > 0`.
pub fn swiglu_clamped(gate: &mut [f32], up: &[f32], limit: f32) {
    debug_assert_eq!(gate.len(), up.len());
    if limit > 0.0 {
        for (g, &u) in gate.iter_mut().zip(up) {
            let gc = g.min(limit);
            let uc = u.clamp(-limit, limit);
            *g = (gc / (1.0 + (-gc).exp())) * uc;
        }
    } else {
        for (g, &u) in gate.iter_mut().zip(up) {
            let gc = *g;
            *g = (gc / (1.0 + (-gc).exp())) * u;
        }
    }
}


/// Grouped O-LoRA output projection: `[n_heads*head_dim] -> [hidden]`.
///
/// V4 splits the attention output into `g` groups and sends each through its OWN slice of
/// `wo_a`, then concatenates and applies `wo_b`:
///
/// ```text
///   o   : [g, dg]           dg = n_heads*head_dim / g
///   wo_a: [g, rank, dg]     (stored flat as [g*rank, dg])
///   mid : [g, rank]         mid[gi][r] = sum_d o[gi][d] * wo_a[gi][r][d]
///   out : wo_b @ mid.flatten()          wo_b: [hidden, g*rank]
/// ```
///
/// The grouping is the part that is easy to get wrong: `wo_a` is a **block-diagonal**
/// operator, not a dense `[g*rank, g*dg]` one. Treating it as dense would multiply every
/// group by every slice — same output shape, silently different model, and no assertion
/// anywhere would fire. The test drives per-group-distinct inputs so a dense read produces
/// visibly different numbers.
pub fn o_lora_grouped(o: &[f32], wo_a: &[f32], g: usize, rank: usize, mid: &mut [f32]) {
    let dg = o.len() / g;
    debug_assert_eq!(o.len(), g * dg);
    debug_assert_eq!(wo_a.len(), g * rank * dg);
    debug_assert_eq!(mid.len(), g * rank);
    for gi in 0..g {
        let src = &o[gi * dg..(gi + 1) * dg];
        for r in 0..rank {
            let row = &wo_a[(gi * rank + r) * dg..(gi * rank + r + 1) * dg];
            let mut acc = 0.0f64;
            for (a, b) in row.iter().zip(src) {
                acc += (*a as f64) * (*b as f64);
            }
            mid[gi * rank + r] = acc as f32;
        }
    }
}


/// DeepSeek-V4 attention core: 64 query heads against ONE shared latent, with a per-head
/// attention sink.
///
/// `q` is `[S, H, hd]` and `kv` is `[T, hd]` — a single latent per position serving as both
/// K and V (`num_key_value_heads: 1`), so every head reads the same cache. Callers pass q
/// and kv already normed and rope-applied; the caller also applies the INVERSE rope to the
/// output (V is that same roped latent, so the rotation has to be undone).
///
/// **The sink contributes to the DENOMINATOR only** — `sum_exp += exp(sink[h] - max)` with
/// no matching term in the numerator, i.e. a learned "attend to nothing" mass per head.
/// Adding it to both, or treating it as an extra key with a value, is silent: outputs stay
/// finite and plausibly scaled, just uniformly too large. The test pins a large negative
/// sink against a large positive one so the denominators differ visibly.
///
/// DENSE over `[0, pos]` — a convenience wrapper over [`attention_dsv4_sparse`] that
/// builds the plain causal index list. Exact only while every query's window covers the
/// whole cache (`seqlen <= sliding_window`, no compressed entries); past that, callers
/// must build a real index list, because attending densely is WRONG, not merely slow.
#[allow(clippy::too_many_arguments)]
pub fn attention_dsv4(
    q: &[f32],
    kv: &[f32],
    attn_sink: &[f32],
    s: usize,
    h: usize,
    hd: usize,
    pos_base: usize,
    scale: f32,
    out: &mut [f32],
) {
    let topk = pos_base + s;
    let mut idxs = vec![-1i32; s * topk];
    for i in 0..s {
        for (j, slot) in idxs[i * topk..i * topk + pos_base + i + 1].iter_mut().enumerate() {
            *slot = j as i32;
        }
    }
    attention_dsv4_sparse(q, kv, attn_sink, s, h, hd, &idxs, topk, scale, out);
}

/// DeepSeek-V4 attention over an EXPLICIT per-query key-index list.
///
/// `idxs` is `[S, topk]` of indices into `kv`, with **`-1` meaning masked** — the same
/// contract as the reference `sparse_attn` kernel, whose gather does
/// `kv[idxs[i]] if idxs[i] != -1 else 0` and forces those scores to `-inf`.
///
/// Causality is carried by the index list rather than derived from a position, and that is
/// the whole point: V4's key set is **not a prefix**. Each query sees a sliding raw window
/// PLUS a set of compressed blocks that on 21 of 43 layers is chosen per-query by the
/// Indexer. There is no single `pos` that describes it. The previous signature took one,
/// which forced the caller to pretend the key set was contiguous; it computed
/// `pos_base - raw_from` on `usize`, which underflowed on any prompt longer than the
/// 128-token window and walked off the end of the cache.
///
/// Duplicate indices are permitted and meaningful: the reference deliberately lets a
/// compressed block that overlaps the raw window be attended BOTH ways.
#[allow(clippy::too_many_arguments)]
pub fn attention_dsv4_sparse(
    q: &[f32],
    kv: &[f32],
    attn_sink: &[f32],
    s: usize,
    h: usize,
    hd: usize,
    idxs: &[i32],
    topk: usize,
    scale: f32,
    out: &mut [f32],
) {
    debug_assert_eq!(q.len(), s * h * hd);
    debug_assert_eq!(out.len(), s * h * hd);
    debug_assert_eq!(attn_sink.len(), h);
    debug_assert_eq!(idxs.len(), s * topk);
    let rows = kv.len() / hd;
    let mut scores: Vec<f32> = Vec::new();
    for i in 0..s {
        let sel = &idxs[i * topk..(i + 1) * topk];
        for hh in 0..h {
            let qv = &q[(i * h + hh) * hd..(i * h + hh) * hd + hd];
            scores.clear();
            let mut m = f32::NEG_INFINITY;
            for &t in sel {
                if t < 0 {
                    scores.push(f32::NEG_INFINITY);
                    continue;
                }
                let j = t as usize;
                debug_assert!(j < rows, "key index {j} past the {rows}-row cache");
                let kr = &kv[j * hd..(j + 1) * hd];
                let mut acc = 0.0f64;
                for (a, b) in qv.iter().zip(kr) {
                    acc += (*a as f64) * (*b as f64);
                }
                let sc = (acc as f32) * scale;
                m = m.max(sc);
                scores.push(sc);
            }
            let mut den = 0.0f32;
            for v in scores.iter_mut() {
                // A masked slot contributes nothing: exp(-inf - m) == 0.
                *v = if v.is_finite() { (*v - m).exp() } else { 0.0 };
                den += *v;
            }
            // Sink: denominator only. Stabilised against the same max the scores used,
            // exactly as the reference kernel does.
            den += (attn_sink[hh] - m).exp();
            let dst = &mut out[(i * h + hh) * hd..(i * h + hh) * hd + hd];
            dst.fill(0.0);
            let inv = 1.0 / den;
            for (k, &w) in scores.iter().enumerate() {
                if w == 0.0 {
                    continue;
                }
                let w = w * inv;
                let vr = &kv[sel[k] as usize * hd..(sel[k] as usize + 1) * hd];
                for (o, &v) in dst.iter_mut().zip(vr) {
                    *o += w * v;
                }
            }
        }
    }
}


/// DeepSeek-V4 rotary embedding on the trailing `rd` dims, in place.
///
/// **Pairs are ADJACENT, not half-split.** The reference does
/// `view_as_complex(x.unflatten(-1, (-1, 2)))`, pairing `(x0,x1), (x2,x3), …`. Llama-style
/// code pairs `(x_i, x_{i+rd/2})` instead, and swapping the two conventions is silent —
/// same shapes, same magnitudes, every token subtly mis-rotated. The round-trip test below
/// would still pass under the wrong convention, which is why the forward result is pinned
/// against reference vectors too.
///
/// `inverse` conjugates the rotation (`-theta`). V4 needs it on the attention OUTPUT
/// because V is the same latent as K and already carries the forward rotation.
///
/// `cos`/`sin` are per position and hold `rd/2` entries. Only the LAST `rd` elements of
/// `x` are touched; the leading `head_dim - rd` "nope" dims are left alone.
pub fn rope_interleaved(x: &mut [f32], cos: &[f32], sin: &[f32], rd: usize, inverse: bool) {
    debug_assert!(rd % 2 == 0 && rd <= x.len());
    debug_assert_eq!(cos.len(), rd / 2);
    debug_assert_eq!(sin.len(), rd / 2);
    let off = x.len() - rd;
    for k in 0..rd / 2 {
        let (i, j) = (off + 2 * k, off + 2 * k + 1);
        let (a, b) = (x[i], x[j]);
        let (c, s) = (cos[k], sin[k]);
        if inverse {
            x[i] = a * c + b * s;
            x[j] = -a * s + b * c;
        } else {
            x[i] = a * c - b * s;
            x[j] = a * s + b * c;
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    const LOGITS: [f32; 8] = [
        0.3722732796,
        0.6776740599,
        0.7471270662,
        -1.9457971765,
        -1.9263764611,
        -0.5957540694,
        -0.0818209988,
        0.0474545494,
    ];
    const BIAS: [f32; 8] = [
        0.1068040283,
        -1.5889774618,
        -0.7203883114,
        -0.7264079265,
        -1.1121013328,
        1.0046830206,
        -0.2700934939,
        -0.2553420573,
    ];
    const WANT_SCORES: [f32; 8] = [
        0.9468411301,
        1.0432274548,
        1.0653265557,
        0.3654387977,
        0.3687737417,
        0.6625666248,
        0.808129496,
        0.8468505893,
    ];

    #[test]
    fn router_matches_the_reference() {
        for (z, w) in LOGITS.iter().zip(WANT_SCORES.iter()) {
            assert!(
                (sqrt_softplus(*z) - w).abs() < 1e-6,
                "sqrt_softplus({z}) = {} want {w}",
                sqrt_softplus(*z)
            );
        }
        let mut out = Vec::new();
        route_topk(&LOGITS, &BIAS, 3, 1.5, &mut out);
        let idx: Vec<u32> = out.iter().map(|&(e, _)| e).collect();
        assert_eq!(idx, vec![0, 5, 7], "selection must follow the BIASED scores");
        let want_w = [0.5782216266f32, 0.404619465, 0.5171589084];
        for ((_, g), w) in out.iter().zip(want_w.iter()) {
            assert!((g - w).abs() < 1e-5, "weight {g} want {w}");
        }
    }

    /// The bias must move the SELECTION and leave the WEIGHTS alone. Expert 5 has the
    /// lowest unbiased score of the three chosen (0.663) yet is selected on a +1.005 bias;
    /// if the weights were taken from the biased scores its weight would rank first instead
    /// of last. Nothing about shapes or sums would reveal that.
    #[test]
    fn bias_moves_selection_but_not_weights() {
        let mut out = Vec::new();
        route_topk(&LOGITS, &BIAS, 3, 1.5, &mut out);
        let w5 = out.iter().find(|&&(e, _)| e == 5).unwrap().1;
        let w0 = out.iter().find(|&&(e, _)| e == 0).unwrap().1;
        let w7 = out.iter().find(|&&(e, _)| e == 7).unwrap().1;
        assert!(w5 < w0 && w5 < w7, "expert 5 must carry the SMALLEST weight of the three");
        // And the weights sum to route_scale.
        let s: f32 = out.iter().map(|&(_, w)| w).sum();
        assert!((s - 1.5).abs() < 1e-5, "weights sum to {s}, want route_scale 1.5");
    }


    #[test]
    fn o_lora_is_block_diagonal_per_group() {
        const G: usize = 3;
        const DG: usize = 4;
        const R: usize = 2;
        let o: [f32; G * DG] = [
            0.3355941016, -0.2018069339, -0.3996449568, 0.4824536835, 0.502964959,
            -0.1215207255, -0.8406737872, 0.5007970179, -0.568683225, 0.3366070281,
            -0.340533081, 0.9958108747,
        ];
        let wo_a: [f32; G * R * DG] = [
            -0.0842778559, -0.054506896, 0.2843292964, -0.2830184939, -0.0037952994,
            -0.0335753325, 0.2136387231, -0.3279327701, 0.4416431718, -0.0533161334,
            0.7175545354, -0.1721734342, 0.219693547, 0.1894987069, 0.2305708847,
            0.0995794559, 0.0601189274, 0.1930313327, 0.423112906, -0.3680631313,
            -0.1372767793, 0.470703903, -0.0966227811, 0.0222486675,
        ];
        let want: [f32; G * R] = [
            -0.267457366, -0.2380899563, -0.4608431761, -0.0564956688, -0.4798181325,
            0.2915679619,
        ];
        let mut mid = [0f32; G * R];
        o_lora_grouped(&o, &wo_a, G, R, &mut mid);
        for (i, (g, w)) in mid.iter().zip(want.iter()).enumerate() {
            assert!((g - w).abs() < 1e-6, "mid[{i}] = {g} want {w}");
        }

        // Prove the PAIRING matters: run group 1's input through group 0's slice. If the
        // implementation ignored the grouping (or mismatched the strides) this is the kind
        // of value it would produce, and it must differ from the correct one.
        //
        // My first attempt at this check compared wo_a[..DG] against o[..DG], which IS
        // mid[0] by definition — it agreed trivially and asserted nothing. The test caught
        // that, which is the argument for making the contrast explicit rather than assumed.
        let mut crossed = 0.0f64;
        for (a, b) in wo_a[..DG].iter().zip(o[DG..2 * DG].iter()) {
            crossed += (*a as f64) * (*b as f64);
        }
        assert!(
            (crossed as f32 - mid[0]).abs() > 1e-3,
            "group 0's slice gives the same answer on group 1's input — grouping unproven"
        );
    }


    #[test]
    fn attention_sink_lands_in_the_denominator_only() {
        const S: usize = 2;
        const H: usize = 2;
        const HD: usize = 4;
        let q: [f32; S * H * HD] = [
            0.3379032508, 0.3834890916, -0.0756740387, -0.5830266066, -0.1004603573,
            0.4513336327, 0.5064108515, 0.3598505688, 0.7793086476, 0.3152932911,
            -1.877775699, -0.3249812003, -0.4180281703, 0.807281146, 1.0113295607,
            -0.3436872418,
        ];
        let kv: [f32; 3 * HD] = [
            0.1168053345, 0.350209567, -0.5709518924, 0.1920349968, 0.3827639102,
            -0.2535910361, -0.3726996383, 0.6406087806, -0.1964679144, 0.8522606005,
            -0.261153981, 0.3652516174,
        ];
        let sink = [0.3f32, -1.2];
        let want: [f32; S * H * HD] = [
            0.051212224, 0.1535461619, -0.2503286031, 0.0841959772, 0.0889944675,
            0.2668261175, -0.4350106083, 0.1463122583, 0.1648992436, 0.0607652527,
            -0.3423565076, 0.2746081392, 0.1913274224, 0.0741464273, -0.4011202593,
            0.3185897833,
        ];
        let mut out = vec![0f32; S * H * HD];
        let scale = (HD as f32).powf(-0.5);
        attention_dsv4(&q, &kv, &sink, S, H, HD, 0, scale, &mut out);
        for (i, (g, w)) in out.iter().zip(want.iter()).enumerate() {
            assert!((g - w).abs() < 1e-6, "out[{i}] = {g} want {w}");
        }

        // Head 0's sink is +0.3 and head 1's is -1.2, so head 0 sheds much more mass to the
        // sink. If the sink were ignored (or added to numerator and denominator alike) both
        // heads would keep full mass and the ratio below would be ~1.
        let mut no_sink = vec![0f32; S * H * HD];
        attention_dsv4(&q, &kv, &[-40.0, -40.0], S, H, HD, 0, scale, &mut no_sink);
        let m0 = out[0] / no_sink[0];
        let m1 = out[HD] / no_sink[HD];
        // RELATIONAL, not a magic threshold. Head 0's sink (+0.3) is larger than head 1's
        // (-1.2), so head 0 must shed strictly more mass; and any sink at all must shed
        // some. Picking absolute cutoffs here needs the denominator computed by hand — at
        // position 0 there is a single key, so even a -1.2 sink takes ~24% of the mass,
        // which is why an eyeballed ">0.95" was wrong while the implementation was right.
        assert!(m0 < m1, "bigger sink must shed more mass: head0 kept {m0}, head1 {m1}");
        assert!(m1 < 1.0, "a sink must shed SOME mass; head 1 kept {m1}");
        assert!(m0 > 0.0 && m1 > 0.0, "mass must stay positive: {m0}, {m1}");
    }


    #[test]
    fn rope_is_interleaved_and_invertible() {
        const RD: usize = 8;
        let x0: [f32; RD] = [
            -0.5678163306, -0.456691087, 0.3495151199, -0.9556618557, 0.0281217185,
            -0.0733127279, -0.7790557407, -0.4621380503,
        ];
        let cos = [0.9553364891f32, 0.4535961214, 0.7316888689, -0.7373937155];
        let sin = [0.2955202067f32, -0.8912073601, 0.68163876, 0.6754631806];
        let want_fwd: [f32; RD] = [
            -0.4074942154, -0.6040948591, -0.6931541768, -0.7449749584, 0.0705491453,
            -0.0344732536, 0.8866280446, -0.1854457745,
        ];
        let mut x = x0;
        rope_interleaved(&mut x, &cos, &sin, RD, false);
        for (i, (g, w)) in x.iter().zip(want_fwd.iter()).enumerate() {
            assert!((g - w).abs() < 1e-6, "fwd[{i}] = {g} want {w}");
        }
        // Inverse must return the original — and this alone does NOT prove the pairing is
        // right (a half-split rope also round-trips), which is why the forward values above
        // are pinned against the reference.
        rope_interleaved(&mut x, &cos, &sin, RD, true);
        for (i, (g, w)) in x.iter().zip(x0.iter()).enumerate() {
            assert!((g - w).abs() < 1e-5, "roundtrip[{i}] = {g} want {w}");
        }
    }

    /// Only the trailing `rd` dims rotate; the leading "nope" dims must be untouched.
    #[test]
    fn rope_leaves_the_nope_dims_alone() {
        let mut x = [1.0f32, 2.0, 3.0, 4.0, 5.0, 6.0];
        rope_interleaved(&mut x, &[0.0], &[1.0], 2, false);
        assert_eq!(&x[..4], &[1.0, 2.0, 3.0, 4.0], "nope dims were rotated");
        // (5,6) with cos=0,sin=1 -> (-6, 5)
        assert!((x[4] + 6.0).abs() < 1e-6 && (x[5] - 5.0).abs() < 1e-6);
    }

    #[test]
    fn clamped_swiglu_matches_the_reference() {
        let mut g = [-3.0f32, 0.5, 12.0, -20.0];
        let u = [2.0f32, -14.0, 11.0, 0.25];
        swiglu_clamped(&mut g, &u, 10.0);
        let want = [-0.2845552391f32, -3.112296656, 99.9954602131, -1.03e-8];
        for (i, (a, b)) in g.iter().zip(want.iter()).enumerate() {
            assert!((a - b).abs() < 1e-4, "swiglu[{i}] = {a} want {b}");
        }
    }

    /// `up` is clamped on both sides, `gate` only from above. Driving a very negative gate
    /// shows it: a symmetric clamp on `gate` would floor it at -limit and change the result.
    #[test]
    fn swiglu_clamp_is_asymmetric() {
        let (mut a, mut b) = ([-20.0f32], [-20.0f32]);
        swiglu_clamped(&mut a, &[1.0], 10.0);
        // What a (wrong) symmetric gate clamp would produce:
        let gc: f32 = -10.0;
        b[0] = (gc / (1.0 + (-gc).exp())) * 1.0;
        assert!(
            (a[0] - b[0]).abs() > 1e-4,
            "a symmetric gate clamp is indistinguishable here — the test proves nothing"
        );
        assert!(a[0].abs() < 1e-6, "silu(-20) should be ~0, got {}", a[0]);
    }
}

/// YaRN rotary tables for DeepSeek-V4: `cos`/`sin` for positions `0..seqlen`, each row
/// `rd/2` entries, laid out row-major.
///
/// Transcribed from `precompute_freqs_cis` in the reference `inference/model.py`, not
/// re-derived — YaRN has several published variants that disagree in the ramp. The
/// interpolation only applies when `original_seq_len > 0`; V4 sets 65536 with `factor` 16,
/// which is what stretches the trained window to 1M.
///
/// Note `find_correction_range` clamps against `dim - 1` while the ramp is built over
/// `dim/2` entries. That asymmetry is in the reference and is preserved deliberately —
/// "fixing" it changes every frequency, silently.
pub fn yarn_rope_tables(
    rd: usize,
    seqlen: usize,
    base: f32,
    factor: f32,
    original_seq_len: usize,
    beta_fast: f32,
    beta_slow: f32,
) -> (Vec<f32>, Vec<f32>) {
    let half = rd / 2;
    let mut freqs: Vec<f32> = (0..half)
        .map(|i| 1.0 / base.powf((2 * i) as f32 / rd as f32))
        .collect();

    if original_seq_len > 0 {
        let corr = |rot: f32| -> f32 {
            rd as f32 * ((original_seq_len as f32) / (rot * 2.0 * std::f32::consts::PI)).ln()
                / (2.0 * base.ln())
        };
        let low = corr(beta_fast).floor().max(0.0);
        let high = corr(beta_slow).ceil().min((rd - 1) as f32);
        let (lo, hi) = if low == high { (low, high + 0.001) } else { (low, high) };
        for (i, f) in freqs.iter_mut().enumerate() {
            let smooth = 1.0 - (((i as f32) - lo) / (hi - lo)).clamp(0.0, 1.0);
            *f = *f / factor * (1.0 - smooth) + *f * smooth;
        }
    }

    let mut cos = vec![0f32; seqlen * half];
    let mut sin = vec![0f32; seqlen * half];
    for t in 0..seqlen {
        for (i, &f) in freqs.iter().enumerate() {
            let a = t as f32 * f;
            cos[t * half + i] = a.cos();
            sin[t * half + i] = a.sin();
        }
    }
    (cos, sin)
}

/// Parameter-free per-head RMS applied to `q` after `wq_b`, in place.
///
/// The reference does `q *= rsqrt(q.square().mean(-1) + eps)` with **no learned weight** —
/// distinct from the `q_norm` RMSNorm applied a few lines earlier to the LoRA bottleneck,
/// which does have one. Two normalisations on the same tensor, one weighted and one not;
/// reusing `q_norm`'s weight here, or skipping this entirely, is silent either way.
pub fn per_head_rms(q: &mut [f32], head_dim: usize, eps: f32) {
    debug_assert_eq!(q.len() % head_dim, 0);
    for head in q.chunks_exact_mut(head_dim) {
        let mut ss = 0.0f64;
        for &v in head.iter() {
            ss += (v as f64) * (v as f64);
        }
        let r = (1.0 / ((ss / head_dim as f64) + eps as f64).sqrt()) as f32;
        for v in head.iter_mut() {
            *v *= r;
        }
    }
}

#[cfg(test)]
mod v4_rope_table_tests {
    use super::*;

    // YaRN must actually CHANGE the frequencies, and only the low ones. A table that
    // silently came out equal to plain RoPE would still round-trip, still look sane, and
    // still mis-place every long-context token — so pin the shape of the effect, not just
    // that it runs.
    #[test]
    fn yarn_interpolates_low_frequencies_only() {
        let (rd, base, factor, orig) = (64usize, 10000.0f32, 16.0f32, 65536usize);
        // A LARGE position. The first version of this test sampled position 1, where every
        // angle is ~0 and cos is ~1 in both arms: a real 5x frequency change showed up as a
        // 9e-9 difference and the test reported "YaRN changed nothing". The slow
        // frequencies are ~1e-4, so the position has to be O(1/freq) before the effect is
        // representable at all.
        let pos = 4096usize;
        let (c_plain, _) = yarn_rope_tables(rd, pos + 1, base, 1.0, 0, 32.0, 1.0);
        let (c_yarn, _) = yarn_rope_tables(rd, pos + 1, base, factor, orig, 32.0, 1.0);
        let half = rd / 2;
        let plain = &c_plain[pos * half..(pos + 1) * half];
        let yarn = &c_yarn[pos * half..(pos + 1) * half];
        // The FASTEST frequency (index 0) is left alone by the ramp.
        assert!((plain[0] - yarn[0]).abs() < 1e-6, "fastest freq must be untouched");
        // Some frequency must actually move, or `factor` did nothing.
        assert!(
            plain.iter().zip(yarn).any(|(a, b)| (a - b).abs() > 1e-4),
            "YaRN changed nothing — factor/original_seq_len are not reaching the ramp"
        );
    }
}

/// Scalar form of [`swiglu_clamped`] for the per-element CPU expert loop.
///
/// `up` is clamped on BOTH sides, `gate` only from above — the reference's asymmetry,
/// because silu already bounds the gate below. Result is `silu(gate) * up`, with no
/// `(up + 1)` and no `sigmoid(alpha*g)`: those belong to the oai variant, whose clamps are
/// identical and whose product is not.
#[inline]
pub fn swiglu_clamped_one(gate: f32, up: f32, limit: f32) -> f32 {
    let g = gate.min(limit);
    let u = up.clamp(-limit, limit);
    (g / (1.0 + (-g).exp())) * u
}

#[cfg(test)]
mod clamp_tests {
    use super::*;

    // The clamp must be ASYMMETRIC and must not be the oai product. Values are chosen to
    // exceed the limit in each direction so a symmetric clamp, a missing clamp, and an
    // `(up + 1)` product all fail — a test using small values would pass under all four.
    #[test]
    fn clamp_is_asymmetric_and_is_not_oai() {
        let lim = 10.0f32;
        // gate far below -limit must NOT be clamped up; silu(-50) ~ 0 so the product ~0.
        assert!(swiglu_clamped_one(-50.0, 1.0, lim).abs() < 1e-6);
        // gate far above +limit IS clamped: same result as feeding exactly the limit.
        assert_eq!(swiglu_clamped_one(50.0, 1.0, lim), swiglu_clamped_one(lim, 1.0, lim));
        // up is clamped BOTH ways.
        assert_eq!(swiglu_clamped_one(1.0, 50.0, lim), swiglu_clamped_one(1.0, lim, lim));
        assert_eq!(swiglu_clamped_one(1.0, -50.0, lim), swiglu_clamped_one(1.0, -lim, lim));
        // and it is silu(g)*u, NOT silu(g)*(u+1): with u = 0 the product must be 0.
        assert!(swiglu_clamped_one(2.0, 0.0, lim).abs() < 1e-6);
    }
}

/// The Compressor's overlap transform: `[nblk, ratio, 2*d]` -> `[nblk, 2*ratio, d]`.
///
/// Only used at `compress_ratio == 4`, where the Compressor runs OVERLAPPING windows so
/// block boundaries are smoother. The projections emit `2*d` per token: the first `d` dims
/// feed the *overlapping* window, the second `d` the *normal* one. Transcribed from
/// `Compressor.overlap_transform`:
///
/// ```text
///   out[i, ratio.. ] = t[i,   :, d.. ]      # normal   window, this block
///   out[i, ..ratio ] = t[i-1, :, ..d ]      # overlap  window, the PREVIOUS block
///   out[0, ..ratio ] = fill                 # block 0 has no predecessor
/// ```
///
/// The `i-1` is the whole point and is what makes it "overlapping". Using `i` there
/// produces the right shape, plausible values, and no overlap at all — which is why the
/// test below pins a distinguishable value per (block, slot) rather than checking dims.
/// `fill` is 0 for the values and -inf for the scores, so block 0's absent predecessor
/// contributes nothing after the softmax.
pub fn overlap_transform(t: &[f32], nblk: usize, ratio: usize, d: usize, fill: f32) -> Vec<f32> {
    debug_assert_eq!(t.len(), nblk * ratio * 2 * d);
    let mut out = vec![fill; nblk * 2 * ratio * d];
    for i in 0..nblk {
        for r in 0..ratio {
            // normal window: second half of this block's dims
            let src = (i * ratio + r) * 2 * d + d;
            let dst = (i * 2 * ratio + ratio + r) * d;
            out[dst..dst + d].copy_from_slice(&t[src..src + d]);
            // overlap window: first half of the PREVIOUS block's dims
            if i > 0 {
                let src = ((i - 1) * ratio + r) * 2 * d;
                let dst = (i * 2 * ratio + r) * d;
                out[dst..dst + d].copy_from_slice(&t[src..src + d]);
            }
        }
    }
    out
}

#[cfg(test)]
mod compressor_tests {
    use super::*;

    // Every (block, position, half) gets a unique value, so a transform that took the
    // CURRENT block instead of the previous one, or swapped the halves, or wrote the slot
    // ranges in the wrong order, all fail. A uniform-valued input would pass under all of
    // those while having exactly the right shape.
    #[test]
    fn overlap_transform_pulls_from_the_previous_block() {
        let (nblk, ratio, d) = (3usize, 4usize, 2usize);
        // value = 100*block + 10*pos + (0 for first half, 5 for second half) + dim
        let mut t = vec![0f32; nblk * ratio * 2 * d];
        for i in 0..nblk {
            for r in 0..ratio {
                for k in 0..d {
                    t[(i * ratio + r) * 2 * d + k] = (100 * i + 10 * r + k) as f32;
                    t[(i * ratio + r) * 2 * d + d + k] = (100 * i + 10 * r + 5 + k) as f32;
                }
            }
        }
        let o = overlap_transform(&t, nblk, ratio, d, f32::NEG_INFINITY);
        assert_eq!(o.len(), nblk * 2 * ratio * d);
        for i in 0..nblk {
            for r in 0..ratio {
                for k in 0..d {
                    // normal half: this block's SECOND half of dims
                    assert_eq!(
                        o[(i * 2 * ratio + ratio + r) * d + k],
                        (100 * i + 10 * r + 5 + k) as f32,
                        "normal slot, block {i} pos {r}"
                    );
                    // overlap half: the PREVIOUS block's FIRST half of dims
                    let got = o[(i * 2 * ratio + r) * d + k];
                    if i == 0 {
                        assert!(got.is_infinite() && got < 0.0, "block 0 overlap must be the fill");
                    } else {
                        assert_eq!(
                            got,
                            (100 * (i - 1) + 10 * r + k) as f32,
                            "overlap slot must come from block {}, not {i}", i - 1
                        );
                    }
                }
            }
        }
    }
}

/// Gated pooling: `out[b] = sum_w softmax(score[b, :, :])[w] * kv[b, w, :]`.
///
/// The softmax is over the WINDOW axis (`w`), independently per block — the reference's
/// `score.softmax(dim=2)` on `[b, s, window, d]`. Note it is per-DIMENSION too: each of
/// the `d` channels gets its own softmax across the window, because `score` is the full
/// width of `kv`, not a scalar per slot. Reducing to one weight per slot would be the
/// natural simplification and is wrong.
///
/// `-inf` scores (block 0's absent overlap predecessor) contribute exactly zero, which is
/// why the transform fills scores with `-inf` and values with `0`.
pub fn gated_pool(kv: &[f32], score: &[f32], nblk: usize, win: usize, d: usize, out: &mut [f32]) {
    debug_assert_eq!(kv.len(), nblk * win * d);
    debug_assert_eq!(score.len(), nblk * win * d);
    debug_assert_eq!(out.len(), nblk * d);
    for b in 0..nblk {
        for k in 0..d {
            let mut m = f32::NEG_INFINITY;
            for w in 0..win {
                m = m.max(score[(b * win + w) * d + k]);
            }
            // An all -inf column would make `exp(x - m)` NaN; it cannot occur for a real
            // block (the normal window is always populated), so treat it as zero weight
            // rather than propagating NaN into the KV cache.
            if !m.is_finite() {
                out[b * d + k] = 0.0;
                continue;
            }
            let mut den = 0.0f32;
            let mut acc = 0.0f32;
            for w in 0..win {
                let e = (score[(b * win + w) * d + k] - m).exp();
                den += e;
                acc += e * kv[(b * win + w) * d + k];
            }
            out[b * d + k] = acc / den;
        }
    }
}

#[cfg(test)]
mod pool_tests {
    use super::*;

    // Two properties a wrong reduction axis would break: (1) a -inf slot contributes
    // nothing, and (2) the softmax is PER-DIMENSION, so two channels with opposite score
    // orderings must select opposite window slots. A scalar-per-slot weighting would give
    // both channels the same winner and pass a single-channel test.
    #[test]
    fn pool_softmaxes_per_dimension_and_ignores_neg_inf() {
        let (nblk, win, d) = (1usize, 3usize, 2usize);
        // channel 0 favours slot 0, channel 1 favours slot 1; slot 2 is masked out.
        let kv = vec![
            10.0, 20.0, // slot 0
            30.0, 40.0, // slot 1
            50.0, 60.0, // slot 2 (masked)
        ];
        let score = vec![
            9.0, 0.0,
            0.0, 9.0,
            f32::NEG_INFINITY, f32::NEG_INFINITY,
        ];
        let mut out = vec![0f32; nblk * d];
        gated_pool(&kv, &score, nblk, win, d, &mut out);
        // channel 0 -> ~slot 0's value (10), channel 1 -> ~slot 1's value (40)
        assert!((out[0] - 10.0).abs() < 0.1, "ch0 got {}", out[0]);
        assert!((out[1] - 40.0).abs() < 0.1, "ch1 got {}", out[1]);
        // the masked slot's large values (50/60) must not leak in
        assert!(out[0] < 20.0 && out[1] < 50.0, "masked slot leaked: {out:?}");
    }
}

/// Per-layer Compressor state carried between forward calls.
///
/// Holds the tokens that have not yet formed a complete window. With overlap the buffer is
/// `2*ratio` slots wide: `[..ratio]` is the overlapping window's carry, `[ratio..]` the
/// normal window's. Scores start at `-inf` so an unfilled slot contributes nothing.
#[derive(Clone)]
pub struct CompressorState {
    pub kv: Vec<f32>,
    pub score: Vec<f32>,
    pub ratio: usize,
    pub coff: usize,
    pub d: usize,
}

impl CompressorState {
    pub fn new(ratio: usize, d: usize) -> Self {
        let coff = if ratio == 4 { 2 } else { 1 };
        let slots = coff * ratio;
        CompressorState {
            kv: vec![0.0; slots * coff * d],
            score: vec![f32::NEG_INFINITY; slots * coff * d],
            ratio,
            coff,
            d,
        }
    }
}

/// Compressor prefill (`start_pos == 0`): pool `[s, hidden]` into `ceil`-many compressed
/// KV rows of width `d`, and leave the unfinished tail in `st`.
///
/// `kvp`/`scorep` are the already-projected `wkv(x)`/`wgate(x)`, each `[s, coff*d]`.
/// Returns the compressed rows `[s/ratio, d]`, **pre-norm and pre-rope** — the caller
/// applies `comp_norm` and the Compressor's own rope (theta 160000, NOT the attention
/// theta) before writing them to the cache.
///
/// Transcribed from `Compressor.forward`'s `start_pos == 0` branch. The parts that are
/// easy to get wrong and silent if you do:
///
///   * The tail (`seqlen % ratio`) does NOT form a block — it goes to the state for the
///     next call. Rounding it up into a short block would produce a compressed row that
///     the reference never emits, shifting every later position.
///   * With overlap, the state ALSO keeps the last full window (`kv[cutoff-ratio..cutoff]`)
///     in slots `[..ratio]`, because the next block's overlap half comes from it.
///   * `ape` is added to the SCORE only, indexed by position-within-window.
pub fn compress_prefill(
    kvp: &[f32],
    scorep: &[f32],
    ape: &[f32],
    s: usize,
    ratio: usize,
    d: usize,
    st: &mut CompressorState,
) -> Vec<f32> {
    let coff = if ratio == 4 { 2 } else { 1 };
    let w = coff * d;
    debug_assert_eq!(kvp.len(), s * w);
    debug_assert_eq!(ape.len(), ratio * w);
    let overlap = coff == 2;
    let remainder = s % ratio;
    let cutoff = s - remainder;
    let offset = if overlap { ratio } else { 0 };

    // Carry: with overlap, the last COMPLETE window feeds the next block's overlap half.
    if overlap && cutoff >= ratio {
        for r in 0..ratio {
            let src = (cutoff - ratio + r) * w;
            st.kv[r * w..r * w + w].copy_from_slice(&kvp[src..src + w]);
            for k in 0..w {
                st.score[r * w + k] = scorep[src + k] + ape[r * w + k];
            }
        }
    }
    // Carry: the tail that does not fill a window.
    for r in 0..remainder {
        let src = (cutoff + r) * w;
        let dst = (offset + r) * w;
        st.kv[dst..dst + w].copy_from_slice(&kvp[src..src + w]);
        for k in 0..w {
            st.score[dst + k] = scorep[src + k] + ape[r * w + k];
        }
    }

    let nblk = cutoff / ratio;
    if nblk == 0 {
        return Vec::new();
    }
    // Blocks: [nblk, ratio, w], score gets `ape` per position-within-window.
    let mut kvb = vec![0f32; nblk * ratio * w];
    let mut scb = vec![0f32; nblk * ratio * w];
    for b in 0..nblk {
        for r in 0..ratio {
            let src = (b * ratio + r) * w;
            let dst = (b * ratio + r) * w;
            kvb[dst..dst + w].copy_from_slice(&kvp[src..src + w]);
            for k in 0..w {
                scb[dst + k] = scorep[src + k] + ape[r * w + k];
            }
        }
    }
    let (kvb, scb, win) = if overlap {
        (
            overlap_transform(&kvb, nblk, ratio, d, 0.0),
            overlap_transform(&scb, nblk, ratio, d, f32::NEG_INFINITY),
            2 * ratio,
        )
    } else {
        (kvb, scb, ratio)
    };
    let mut out = vec![0f32; nblk * d];
    gated_pool(&kvb, &scb, nblk, win, d, &mut out);
    out
}

#[cfg(test)]
mod prefill_tests {
    use super::*;

    // The tail must NOT become a block. With ratio 4 and s = 10, the reference emits 2
    // compressed rows (8 tokens) and carries 2 — an implementation that rounds up emits 3
    // and shifts every subsequent position, while still returning a well-shaped buffer.
    #[test]
    fn prefill_drops_the_tail_into_state_not_into_a_block() {
        let (ratio, d, s) = (4usize, 3usize, 10usize);
        let coff = 2;
        let w = coff * d;
        let kvp: Vec<f32> = (0..s * w).map(|k| (k as f32) * 0.01).collect();
        let scorep = vec![0f32; s * w];
        let ape = vec![0f32; ratio * w];
        let mut st = CompressorState::new(ratio, d);
        let out = compress_prefill(&kvp, &scorep, &ape, s, ratio, d, &mut st);
        assert_eq!(out.len(), (s / ratio) * d, "10 tokens at ratio 4 must give 2 rows");
        // the 2 carried tokens are in the state's NORMAL half (slots [ratio..])
        let tail0 = &st.kv[ratio * w..ratio * w + w];
        assert_eq!(tail0, &kvp[8 * w..9 * w], "first carried token must be token 8");
        // and the last complete window is kept for the next block's overlap half
        assert_eq!(&st.kv[0..w], &kvp[4 * w..5 * w], "overlap carry must be token 4");
    }

    // A sequence shorter than one window compresses to nothing at all.
    #[test]
    fn prefill_shorter_than_one_window_emits_no_blocks() {
        let (ratio, d, s) = (4usize, 3usize, 3usize);
        let w = 2 * d;
        let mut st = CompressorState::new(ratio, d);
        let out = compress_prefill(
            &vec![1.0; s * w], &vec![0.0; s * w], &vec![0.0; ratio * w], s, ratio, d, &mut st,
        );
        assert!(out.is_empty(), "3 tokens at ratio 4 cannot form a block");
    }
}

/// Compressor decode step: absorb one token, and every `ratio`-th step emit one compressed
/// row (pre-norm, pre-rope). Returns `None` on the steps that only accumulate.
///
/// Transcribed from `Compressor.forward`'s `start_pos > 0` branch. The overlap case is the
/// subtle one — it does NOT pool the state buffer as stored. It builds the window from
/// **different dim-halves of different state regions**:
///
/// ```text
///   window[..ratio] = state[..ratio ][.. d]   # previous window, FIRST half of dims
///   window[ratio..] = state[ratio.. ][d ..]   # current  window, SECOND half of dims
/// ```
///
/// then rotates `state[..ratio] = state[ratio..]` so this window becomes the next step's
/// predecessor. Pooling the buffer directly has the right shape and silently mixes the two
/// halves' semantics.
///
/// `should_compress` is `(pos + 1) % ratio == 0`, so the emitted row corresponds to the
/// window ENDING at `pos` — the caller ropes it at position `pos + 1 - ratio` and stores it
/// at cache index `pos / ratio`.
pub fn compress_decode(
    kvp: &[f32],
    scorep: &[f32],
    ape: &[f32],
    pos: usize,
    st: &mut CompressorState,
) -> Option<Vec<f32>> {
    let (ratio, coff, d) = (st.ratio, st.coff, st.d);
    let w = coff * d;
    debug_assert_eq!(kvp.len(), w);
    let overlap = coff == 2;
    let slot = pos % ratio;
    let base = if overlap { ratio + slot } else { slot };

    st.kv[base * w..base * w + w].copy_from_slice(kvp);
    for k in 0..w {
        st.score[base * w + k] = scorep[k] + ape[slot * w + k];
    }
    if (pos + 1) % ratio != 0 {
        return None;
    }

    let win = if overlap { 2 * ratio } else { ratio };
    let mut kvw = vec![0f32; win * d];
    let mut scw = vec![0f32; win * d];
    if overlap {
        for r in 0..ratio {
            // previous window, FIRST half of dims
            kvw[r * d..r * d + d].copy_from_slice(&st.kv[r * w..r * w + d]);
            scw[r * d..r * d + d].copy_from_slice(&st.score[r * w..r * w + d]);
            // current window, SECOND half of dims
            let src = (ratio + r) * w + d;
            kvw[(ratio + r) * d..(ratio + r) * d + d].copy_from_slice(&st.kv[src..src + d]);
            scw[(ratio + r) * d..(ratio + r) * d + d].copy_from_slice(&st.score[src..src + d]);
        }
    } else {
        kvw.copy_from_slice(&st.kv[..ratio * d]);
        scw.copy_from_slice(&st.score[..ratio * d]);
    }
    let mut out = vec![0f32; d];
    gated_pool(&kvw, &scw, 1, win, d, &mut out);

    if overlap {
        // This window becomes the next step's predecessor.
        let (a, b) = st.kv.split_at_mut(ratio * w);
        a.copy_from_slice(&b[..ratio * w]);
        let (a, b) = st.score.split_at_mut(ratio * w);
        a.copy_from_slice(&b[..ratio * w]);
    }
    Some(out)
}

#[cfg(test)]
mod decode_tests {
    use super::*;

    // Emission cadence: exactly one row every `ratio` steps, on the step where the window
    // closes. An off-by-one in `should_compress` still emits the right COUNT over a long
    // run, so the test checks WHICH steps fire, not how many.
    #[test]
    fn decode_emits_only_on_window_close() {
        let (ratio, d) = (4usize, 3usize);
        let w = 2 * d;
        let mut st = CompressorState::new(ratio, d);
        let ape = vec![0f32; ratio * w];
        let fired: Vec<usize> = (0..12)
            .filter(|&pos| {
                compress_decode(&vec![1.0; w], &vec![0.0; w], &ape, pos, &mut st).is_some()
            })
            .collect();
        assert_eq!(fired, vec![3, 7, 11], "must fire when (pos+1) % ratio == 0");
    }

    // The overlap window must draw its two halves from different state regions. Feed the
    // FIRST half of dims a distinguishable value and the second half another: the pooled
    // row must reflect the current window's SECOND half, not the buffer as stored.
    #[test]
    fn decode_overlap_takes_second_half_of_current_window() {
        let (ratio, d) = (4usize, 2usize);
        let w = 2 * d;
        let mut st = CompressorState::new(ratio, d);
        let ape = vec![0f32; ratio * w];
        // first half of dims = 1.0, second half = 9.0; scores equal so the pool averages.
        let mut kvp = vec![1.0f32; w];
        for k in d..w {
            kvp[k] = 9.0;
        }
        let mut out = None;
        for pos in 0..ratio {
            out = compress_decode(&kvp, &vec![0.0; w], &ape, pos, &mut st);
        }
        let out = out.expect("must emit on the closing step");
        // The previous window is unfilled (-inf scores) so only the current window's
        // SECOND half contributes: every channel must be 9.0, not 1.0 and not a blend.
        for (k, &v) in out.iter().enumerate() {
            assert!((v - 9.0).abs() < 1e-5, "channel {k} = {v}, expected the second dim-half (9.0)");
        }
    }
}

/// Walsh-Hadamard rotation over the trailing `n` elements, scaled by `n^-0.5`.
///
/// The reference's `rotate_activation`: `hadamard_transform(x, scale = x.size(-1) ** -0.5)`.
/// Its purpose is QAT-matching — it spreads information across dimensions so the following
/// fp4/fp8 quantisation sees a flatter distribution. Skipping it does not crash or look
/// wrong; it just scores against activations the model was not trained on, which is
/// precisely where the Indexer's top-k selection goes subtly bad.
///
/// Despite the docstring saying "randomized", the reference applies NO sign vector — it is
/// the plain transform. Adding one would be a reasonable-looking guess and wrong.
///
/// `n` must be a power of two (V4 uses `index_head_dim` 128).
pub fn hadamard_rotate(x: &mut [f32], n: usize) {
    debug_assert!(n.is_power_of_two(), "WHT needs a power-of-two length, got {n}");
    debug_assert_eq!(x.len() % n, 0);
    for row in x.chunks_exact_mut(n) {
        let mut h = 1;
        while h < n {
            let mut i = 0;
            while i < n {
                for j in i..i + h {
                    let (a, b) = (row[j], row[j + h]);
                    row[j] = a + b;
                    row[j + h] = a - b;
                }
                i += h << 1;
            }
            h <<= 1;
        }
        let s = (n as f32).powf(-0.5);
        for v in row.iter_mut() {
            *v *= s;
        }
    }
}

#[cfg(test)]
mod hadamard_tests {
    use super::*;

    // Involution alone is a WEAK check — the identity function is also its own inverse, so
    // a no-op implementation passes it. Pin the actual 2-point values and require real
    // mixing as well.
    #[test]
    fn hadamard_matches_known_values_and_actually_mixes() {
        let r2 = 2f32.powf(-0.5);
        // n = 2: [a, b] -> [(a+b)/sqrt2, (a-b)/sqrt2]
        let mut x = vec![3.0f32, 1.0];
        hadamard_rotate(&mut x, 2);
        assert!((x[0] - 4.0 * r2).abs() < 1e-6, "got {x:?}");
        assert!((x[1] - 2.0 * r2).abs() < 1e-6, "got {x:?}");

        // n = 4, one-hot input: every output must be 1/2 in magnitude — a no-op leaves
        // three zeros and fails, and a wrong scale fails on the magnitude.
        let mut y = vec![1.0f32, 0.0, 0.0, 0.0];
        hadamard_rotate(&mut y, 4);
        for (i, &v) in y.iter().enumerate() {
            assert!((v.abs() - 0.5).abs() < 1e-6, "element {i} = {v}, expected +-0.5");
        }
    }

    // Orthogonality: applying it twice returns the original. Checked SECOND, and only
    // after the value test above, because on its own it cannot distinguish the real
    // transform from doing nothing.
    #[test]
    fn hadamard_is_its_own_inverse() {
        let n = 128;
        let orig: Vec<f32> = (0..n).map(|k| ((k as f32) * 0.37).sin() * 3.0).collect();
        let mut x = orig.clone();
        hadamard_rotate(&mut x, n);
        // It must have CHANGED something first, or the round-trip proves nothing.
        assert!(x.iter().zip(&orig).any(|(a, b)| (a - b).abs() > 1e-3), "transform was a no-op");
        hadamard_rotate(&mut x, n);
        for (i, (a, b)) in x.iter().zip(&orig).enumerate() {
            assert!((a - b).abs() < 1e-4, "element {i}: {a} vs {b}");
        }
    }
}

/// Round to the nearest representable e4m3 value (FP8, 3 mantissa bits, max 448).
#[inline]
fn e4m3_round(v: f32) -> f32 {
    if v == 0.0 || !v.is_finite() {
        return v;
    }
    let a = v.abs();
    if a >= 448.0 {
        return 448.0f32.copysign(v);
    }
    // e4m3's smallest normal is 2^-6; below that it is subnormal with a fixed step.
    let e = a.log2().floor().max(-6.0);
    let step = (e - 3.0).exp2(); // 3 mantissa bits => 8 steps per binade
    // Ties to EVEN, matching the hardware cast (and `e2m1_round` below). `f32::round`
    // is ties-away-from-zero, which disagrees on exactly the midpoints — e.g. 12.5 steps
    // becomes 13 instead of 12.
    ((a / step).round_ties_even() * step).min(448.0).copysign(v)
}

/// In-place FP8 activation simulation: block-wise quantise then immediately dequantise.
///
/// The reference's `act_quant(..., inplace=True)` — "fused quant+dequant back to BF16". It
/// does NOT shrink anything; its whole purpose is to make inference see the same rounding
/// the model saw during QAT. Omitting it leaves activations *more* precise than training,
/// which sounds harmless and is not: V4 quantises the non-rope KV dims specifically, and
/// the rope dims specifically NOT, so the two halves are meant to carry different
/// precision.
///
/// `block` is the group size along the last dim (64 for V4's KV call).
///
/// The scale is a **power of two**, not `amax / 448`. The reference passes
/// `scale_fmt` into `act_quant`, which sets `round_scale = scale_fmt is not None`, and
/// `Transformer.__init__` resolves `scale_fmt` to `"ue8m0"` for this checkpoint — so the
/// `round_scale` branch is the live one and the scale goes through `fast_round_scale`.
/// The module-level default is `None`, which is what makes reading the call site alone
/// misleading. A plain `amax/448` scale is wrong by up to 2x per block.
pub fn act_quant_sim(x: &mut [f32], block: usize) {
    for grp in x.chunks_mut(block) {
        let amax = grp.iter().fold(0f32, |m, &v| m.max(v.abs())).max(1e-4);
        let s = round_pow2_scale(amax, 448.0);
        for v in grp.iter_mut() {
            *v = e4m3_round(*v / s) * s;
        }
    }
}

/// `2^ceil(log2(amax / fmt_max))` — the reference's `fast_round_scale`.
///
/// The E8M0 scale formats (`ue8m0`, and FP4's block scale) store only an exponent, so the
/// scale must be an exact power of two. The reference computes it by bit-twiddling the
/// float rather than calling `log2`; this is the same value, and the `mantissa != 0`
/// term is what makes it a CEILING — an exact power of two keeps its own exponent
/// instead of being bumped one binade.
///
/// Takes `fmt_max` and DIVIDES, where the reference multiplies by a precomputed `1/max`.
/// Division is correctly rounded, so `448/448` is exactly `1.0`; the reciprocal form can
/// land a hair above 1, and since the result is a ceiling on the exponent, one ulp there
/// doubles the scale for the whole block. Only bites when amax is exactly `fmt_max * 2^k`,
/// which is precisely the case the tests pin.
#[inline]
pub fn round_pow2_scale(amax: f32, fmt_max: f32) -> f32 {
    let t = amax / fmt_max;
    let bits = t.to_bits();
    let exp = ((bits >> 23) & 0xFF) as i32 - 127;
    let man = bits & 0x007F_FFFF;
    ((exp + i32::from(man != 0)) as f32).exp2()
}

/// Round to the nearest representable E2M1 (FP4) value, ties to even.
///
/// The whole format is 8 magnitudes: `0, 0.5, 1, 1.5, 2, 3, 4, 6`. Every tie lands on the
/// value with an even mantissa bit, which for this table is always the one further from
/// zero at the binade start and closer to zero elsewhere — so the rule cannot be replaced
/// by "round half up" or "round half away from zero" without changing 7 of the inputs.
#[inline]
fn e2m1_round(v: f32) -> f32 {
    const LEVELS: [f32; 8] = [0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0];
    // Mantissa-bit parity of each level, for ties-to-even.
    const EVEN: [bool; 8] = [true, false, true, false, true, false, true, false];
    let a = v.abs();
    if a >= 6.0 {
        return 6.0f32.copysign(v);
    }
    let mut best = 0usize;
    for k in 1..LEVELS.len() {
        let (dk, db) = ((a - LEVELS[k]).abs(), (a - LEVELS[best]).abs());
        // Strictly closer wins; an exact tie goes to whichever has an even mantissa.
        if dk < db || (dk == db && EVEN[k] && !EVEN[best]) {
            best = k;
        }
    }
    LEVELS[best].copysign(v)
}

/// In-place FP4 (E2M1) activation simulation — the reference's
/// `fp4_act_quant(x, fp4_block_size, inplace=True)`, block 32.
///
/// **`fp4_block_size = 32` is a module-level constant in `inference/model.py`, not a
/// config field** — `kernel.py`'s `fp4_act_quant` independently defaults to 32. It was
/// worth pinning rather than guessing: the block size sets which values share a scale, so
/// a wrong one leaves the Indexer scoring a plausible-looking but differently-distributed
/// query against its keys.
///
/// Unlike the FP8 path this quantises the WHOLE row including the rope dims, and it is
/// applied only where the reference applies it: the Indexer's query, and the Indexer's own
/// Compressor output (`rotate=True`).
pub fn fp4_act_quant_sim(x: &mut [f32], block: usize) {
    for grp in x.chunks_mut(block) {
        // The reference floors amax at 6*2^-126 — the smallest value for which the
        // power-of-two scale stays a normal float.
        let amax = grp.iter().fold(0f32, |m, &v| m.max(v.abs())).max(6.0 * (-126f32).exp2());
        let s = round_pow2_scale(amax, 6.0);
        for v in grp.iter_mut() {
            *v = e2m1_round((*v / s).clamp(-6.0, 6.0)) * s;
        }
    }
}

#[cfg(test)]
mod act_quant_tests {
    use super::*;

    // It must actually LOSE precision, and every value must stay within one quantisation
    // step. A no-op passes any "output is close to input" check, so assert both directions.
    //
    // NOTE: the block max does NOT survive exactly, and an earlier version of this test
    // asserted that it did. That assertion encoded a `amax/448` scale; the real scale is
    // `2^ceil(log2(amax/448))` (`scale_fmt="ue8m0"` => `round_scale=True`), so `amax` maps
    // to somewhere in (224, 448] rather than exactly onto 448 and picks up its own rounding.
    #[test]
    fn act_quant_rounds_and_stays_within_a_step() {
        // 1 + 1/64 needs 6 mantissa bits, so e4m3 cannot represent it and it must move.
        let mut x = vec![1.0 + 1.0 / 64.0, 100.0];
        let before = x.clone();
        act_quant_sim(&mut x, 2);
        assert!((x[0] - before[0]).abs() > 1e-4, "value needing 6 mantissa bits must round");
        // Scale for this block is 2^ceil(log2(100/448)) = 2^-2. In scaled units 100 becomes
        // 400, which sits in the 2^8 binade where e4m3's step is 2^(8-3) = 32 — so the step
        // back in original units is 32 * 2^-2 = 8. Everything must land within half of that.
        for (v, b) in x.iter().zip(&before) {
            assert!((v - b).abs() <= 4.0 + 1e-6, "{b} -> {v} moved more than half a step");
        }
    }

    // The scale is a power of two, and that is the whole substance of the `ue8m0` fix.
    // Pin it directly: a block whose amax is exactly a power of two must round-trip
    // representable values EXACTLY, which an `amax/448` scale would not.
    #[test]
    fn act_quant_scale_is_a_power_of_two() {
        assert_eq!(round_pow2_scale(448.0, 448.0), 1.0);
        assert_eq!(round_pow2_scale(896.0, 448.0), 2.0);
        // Not an exact power of two => ceiling, so 100/448 = 0.223 -> 2^-2.
        assert_eq!(round_pow2_scale(100.0, 448.0), 0.25);
        // With scale exactly 1, every e4m3-representable value is a fixed point.
        let mut x = vec![448.0f32, 2.0, 1.5];
        act_quant_sim(&mut x, 3);
        assert_eq!(x, vec![448.0, 2.0, 1.5]);
    }

    // Blocks are independent: a huge value in one block must not degrade another.
    #[test]
    fn act_quant_scales_per_block() {
        let mut x = vec![1.0, 1.0, 1000.0, 1000.0];
        act_quant_sim(&mut x, 2);
        assert!((x[0] - 1.0).abs() < 1e-3, "small block must keep its own scale: {}", x[0]);
    }

    #[test]
    fn act_quant_leaves_zero_blocks_alone() {
        let mut x = vec![0.0f32; 4];
        act_quant_sim(&mut x, 2);
        assert!(x.iter().all(|&v| v == 0.0), "an all-zero block must not produce NaN");
    }
}

#[cfg(test)]
mod hash_routing_tests {
    use super::*;

    // Selection is the TABLE, not the scores — the whole point. Give the table the two
    // WORST-scoring experts and check they are exactly what comes out; a `route_topk`
    // that ignored `eids` would return the two best and pass any weaker check.
    #[test]
    fn selection_comes_from_the_table_not_the_scores() {
        let logits = vec![5.0f32, -3.0, 4.0, -4.0];
        let mut out = Vec::new();
        route_hash(&logits, &[1, 3], 1.0, &mut out);
        assert_eq!(out.iter().map(|&(e, _)| e).collect::<Vec<_>>(), vec![1, 3]);
    }

    // The router matmul still supplies the WEIGHTS. They are the unbiased sqrt-softplus
    // scores at the chosen indices, normalised over the chosen set and scaled — so a
    // higher-scoring chosen expert must get more weight, and the total must be route_scale.
    #[test]
    fn weights_are_normalised_unbiased_scores_times_scale() {
        let logits = vec![5.0f32, -3.0, 4.0, -4.0];
        let mut out = Vec::new();
        route_hash(&logits, &[0, 2], 1.5, &mut out);
        let total: f32 = out.iter().map(|&(_, w)| w).sum();
        assert!((total - 1.5).abs() < 1e-5, "weights must sum to route_scale: {total}");
        let (w0, w2) = (out[0].1, out[1].1);
        let (s0, s2) = (sqrt_softplus(5.0), sqrt_softplus(4.0));
        assert!(w0 > w2, "expert 0 scores higher so must weigh more: {w0} vs {w2}");
        assert!((w0 / w2 - s0 / s2).abs() < 1e-4, "weights must keep the score ratio");
    }

    // A repeated expert is merged by SUMMING, so it ends up with the weight it would
    // have had from two entries. Emitting it twice would break the downstream union,
    // which assumes each expert appears once.
    #[test]
    fn duplicate_table_entries_merge_by_summing() {
        let logits = vec![1.0f32, 2.0, 3.0, 4.0];
        let mut out = Vec::new();
        route_hash(&logits, &[2, 2, 0], 1.0, &mut out);
        assert_eq!(out.len(), 2, "expert 2 must appear once: {out:?}");
        let w2 = out.iter().find(|&&(e, _)| e == 2).unwrap().1;
        let w0 = out.iter().find(|&&(e, _)| e == 0).unwrap().1;
        // 2*s2 vs s0, normalised over 2*s2 + s0
        let (s0, s2) = (sqrt_softplus(1.0), sqrt_softplus(3.0));
        let denom = 2.0 * s2 + s0;
        assert!((w2 - 2.0 * s2 / denom).abs() < 1e-5, "{w2}");
        assert!((w0 - s0 / denom).abs() < 1e-5, "{w0}");
    }

    // A layer with no table must produce nothing, so the caller falls back to score
    // routing rather than silently routing every token to expert 0.
    //
    // NOTE: an out-of-range expert id is NOT tested as "gracefully skipped" — it
    // `debug_assert`s. The loader validates every entry against `n_experts` when it
    // builds the table, so reaching `route_hash` with a bad id means the table is
    // corrupt, and quietly dropping an expert there would be a wrong model that runs.
    #[test]
    fn an_empty_table_selects_nothing() {
        let logits = vec![1.0f32, 2.0];
        let mut out = vec![(9u32, 9.0f32)];
        route_hash(&logits, &[], 1.0, &mut out);
        assert!(out.is_empty(), "an empty table must not leave stale picks: {out:?}");
    }
}

#[cfg(test)]
mod sparse_attention_tests {
    use super::*;

    // q = [1,0]; both keys score 1.0 against it but carry opposite values, so the weights
    // are visible directly in the output. A very negative sink keeps its mass out of the
    // denominator, which would otherwise scale every expectation below.
    fn fixture() -> (Vec<f32>, Vec<f32>, Vec<f32>) {
        let q = vec![1.0f32, 0.0];
        let kv = vec![1.0f32, 5.0, 1.0, -5.0, 1.0, 0.0];
        (q, kv, vec![-100.0f32])
    }

    // A masked slot must contribute NOTHING. Treating -1 as "key 0" or as an all-zero key
    // both keep the output finite and plausibly scaled — the zero key is the nastier one,
    // since it still adds exp(0) of mass to the denominator and shrinks every output.
    #[test]
    fn masked_slots_are_not_keys() {
        let (q, kv, sink) = fixture();
        let mut a = vec![0f32; 2];
        let mut b = vec![0f32; 2];
        attention_dsv4_sparse(&q, &kv, &sink, 1, 1, 2, &[0, 1, -1, -1], 4, 1.0, &mut a);
        attention_dsv4_sparse(&q, &kv, &sink, 1, 1, 2, &[0, 1], 2, 1.0, &mut b);
        for (x, y) in a.iter().zip(&b) {
            assert!((x - y).abs() < 1e-6, "padding changed the result: {a:?} vs {b:?}");
        }
        // and the value is the plain average of the two equally-scored keys
        assert!((a[0] - 1.0).abs() < 1e-6, "{a:?}");
        assert!(a[1].abs() < 1e-6, "{a:?}");
    }

    // Duplicates are MEANINGFUL, not a bug to dedupe: V4 lets a compressed block that
    // overlaps the raw window be attended both ways, so a repeated index must carry
    // repeated weight.
    #[test]
    fn duplicate_indices_count_twice() {
        let (q, kv, sink) = fixture();
        let mut out = vec![0f32; 2];
        attention_dsv4_sparse(&q, &kv, &sink, 1, 1, 2, &[0, 0, 1], 3, 1.0, &mut out);
        // (2*[1,5] + [1,-5]) / 3 = [1, 5/3]
        assert!((out[0] - 1.0).abs() < 1e-5, "{out:?}");
        assert!((out[1] - 5.0 / 3.0).abs() < 1e-5, "{out:?}");
    }

    // The key set is a SET, not a prefix: a query may skip a key in the middle. This is
    // what the Indexer does, and what the old position-derived signature could not express.
    #[test]
    fn a_gap_in_the_middle_is_honoured() {
        let (q, kv, sink) = fixture();
        let mut out = vec![0f32; 2];
        // keys 0 and 2 only — key 1 is never listed.
        attention_dsv4_sparse(&q, &kv, &sink, 1, 1, 2, &[0, 2], 2, 1.0, &mut out);
        // ([1,5] + [1,0]) / 2 = [1, 2.5]
        assert!((out[1] - 2.5).abs() < 1e-5, "key 1 leaked in: {out:?}");
    }

    // The dense wrapper must still be the plain causal case — this is what every other V4
    // test exercises, so a regression here would be broad but silent.
    #[test]
    fn dense_wrapper_is_causal() {
        let (q3, kv, sink) = {
            let (_, kv, sink) = fixture();
            // three queries, all equal, so only the causal span distinguishes them
            (vec![1.0f32, 0.0, 1.0, 0.0, 1.0, 0.0], kv, sink)
        };
        let mut out = vec![0f32; 6];
        attention_dsv4(&q3, &kv, &sink, 3, 1, 2, 0, 1.0, &mut out);
        // query 0 sees key 0 only => exactly v0
        assert!((out[0] - 1.0).abs() < 1e-5 && (out[1] - 5.0).abs() < 1e-5, "{out:?}");
        // query 1 sees keys 0,1 => [1, 0]
        assert!(out[3].abs() < 1e-5, "{out:?}");
        // query 2 sees keys 0,1,2 => [1, 0]
        assert!(out[5].abs() < 1e-5, "{out:?}");
    }
}

#[cfg(test)]
mod fp4_tests {
    use super::*;

    // The eight magnitudes ARE the format. Pin them: a table that is subtly wrong (say
    // 5.0 instead of 6.0 at the top, a uniform step, or a missing 1.5) still produces
    // finite, plausibly-scaled output, so nothing downstream would notice.
    #[test]
    fn e2m1_levels_are_exact_fixed_points() {
        for v in [0.0f32, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0] {
            assert_eq!(e2m1_round(v), v, "{v} must be representable");
            assert_eq!(e2m1_round(-v), -v, "-{v} must be representable");
        }
    }

    // Every tie in E2M1 resolves to the neighbour with an even mantissa bit. Checked
    // exhaustively over all seven midpoints, because "round half away from zero" agrees
    // on some of them and disagrees on others — a partial test would not separate them.
    #[test]
    fn e2m1_ties_go_to_even() {
        // (input, expected). Half-away-from-zero would give 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0.
        for (x, want) in [
            (0.25f32, 0.0f32),
            (0.75, 1.0),
            (1.25, 1.0),
            (1.75, 2.0),
            (2.5, 2.0),
            (3.5, 4.0),
            (5.0, 4.0),
        ] {
            assert_eq!(e2m1_round(x), want, "tie at {x}");
            assert_eq!(e2m1_round(-x), -want, "tie at -{x}");
        }
    }

    // A power-of-two scale means a block whose amax is 6*2^k round-trips its own maximum
    // exactly — and it must LOSE something in between, or it is a no-op.
    #[test]
    fn fp4_sim_is_lossy_but_keeps_a_scaled_max() {
        let mut x = vec![6.0f32, 3.0, 1.7, 0.1];
        let before = x.clone();
        fp4_act_quant_sim(&mut x, 4);
        // amax = 6 => scale 2^ceil(log2(1)) = 1, so the levels land on themselves.
        assert_eq!(x[0], 6.0);
        assert_eq!(x[1], 3.0);
        assert_eq!(x[2], 1.5, "1.7 must snap to the 1.5 level");
        assert!((x[3] - before[3]).abs() > 1e-6, "0.1 must move — the format has no such value");
    }

    // Blocks are independent, exactly as in the FP8 path.
    #[test]
    fn fp4_sim_scales_per_block() {
        let mut x = vec![6.0f32, 3.0, 600.0, 300.0];
        fp4_act_quant_sim(&mut x, 2);
        assert_eq!(x[0], 6.0, "small block keeps its own scale");
        // Second block: amax 600 => scale 2^ceil(log2(100)) = 2^7 = 128.
        // 600/128 = 4.6875 -> 4 (nearest level, 4.6875 is closer to 4 than to 6) -> 512.
        assert_eq!(x[2], 512.0);
    }

    // An all-zero block must not divide by zero. The reference floors amax at 6*2^-126
    // precisely so the power-of-two scale stays a normal float.
    #[test]
    fn fp4_sim_survives_a_zero_block() {
        let mut x = vec![0.0f32; 8];
        fp4_act_quant_sim(&mut x, 4);
        assert!(x.iter().all(|&v| v == 0.0), "an all-zero block must not produce NaN");
    }
}

#[cfg(test)]
mod reference_vector_tests {
    use super::*;

    // Vectors TRANSCRIBED from `inference/kernel.py` and `inference/model.py`, generated by
    // `scripts/dsv4_ref_vectors.py` (rerun it to regenerate). Transcribed rather than re-derived on purpose: a
    // re-derivation reproduces whatever misreading the Rust already contains, which is the
    // trap `hc.rs` records. Where the reference casts to a hardware float type the
    // generator uses torch's own cast, so the rounding is defined by the format, not by me.
    //
    // These close the gap between "implemented from reading the reference" and "agrees
    // with the reference" for the pieces added while wiring up the Indexer.

const ACT_IN: [f32; 64] = [0.0f32, 1.0848464f32, 2.0228639f32, 2.6870961f32, 2.9876425f32, 2.8838258f32, 2.3896964f32, 1.5721326f32, 0.5417887f32, -0.56188375f32, -1.5895085f32, -2.4020007f32, -2.8893929f32, -2.9857194f32, -2.6779428f32, -2.007719f32, -1.0657605f32, 0.020443805f32, 1.1038811f32, 2.0379143f32, 2.6961246f32, 2.9894269f32, 2.8781242f32, 2.377281f32, 1.5546844f32, 0.52166843f32, -0.58195275f32, -1.6068095f32, -2.4141936f32, -2.8948262f32, -2.9836576f32, -2.6686654f32, -1.9924818f32, -1.0466255f32, 0.040886663f32, 1.122865f32, 2.0528688f32, 2.7050281f32, 2.9910724f32, 2.8722894f32, 2.3647559f32, 1.5371637f32, 0.50152397f32, -0.60199475f32, -1.6240387f32, -2.4262724f32, -2.9001248f32, -2.9814577f32, -2.6592641f32, -1.9771497f32, -1.0274419f32, 0.061330482f32, 1.1417967f32, 2.0677309f32, 2.7138042f32, 2.992579f32, 2.8663206f32, 2.3521209f32, 1.5195694f32, 0.48135626f32, -0.62201154f32, -1.6411877f32, -2.4382415f32, -2.9052877f32];
const ACT_OUT: [f32; 64] = [0.0f32, 1.125f32, 2.0f32, 2.75f32, 3.0f32, 3.0f32, 2.5f32, 1.625f32, 0.5625f32, -0.5625f32, -1.625f32, -2.5f32, -3.0f32, -3.0f32, -2.75f32, -2.0f32, -1.125f32, 0.01953125f32, 1.125f32, 2.0f32, 2.75f32, 3.0f32, 3.0f32, 2.5f32, 1.5f32, 0.5f32, -0.5625f32, -1.625f32, -2.5f32, -3.0f32, -3.0f32, -2.75f32, -2.0f32, -1.0f32, 0.0390625f32, 1.125f32, 2.0f32, 2.75f32, 3.0f32, 2.75f32, 2.25f32, 1.5f32, 0.5f32, -0.625f32, -1.625f32, -2.5f32, -3.0f32, -3.0f32, -2.75f32, -2.0f32, -1.0f32, 0.0625f32, 1.125f32, 2.0f32, 2.75f32, 3.0f32, 2.75f32, 2.25f32, 1.5f32, 0.46875f32, -0.625f32, -1.625f32, -2.5f32, -3.0f32];
const FP4_IN: [f32; 32] = [5.0f32, 4.8901548f32, 4.5654449f32, 4.0401373f32, 3.3373141f32, 2.4878554f32, 1.5290846f32, 0.50312912f32, -0.5449335f32, -1.5690528f32, -2.5242302f32, -3.3684981f32, -4.0647602f32, -4.5824242f32, -4.8987446f32, -4.9998231f32, -4.8812189f32, -4.5481429f32, -4.0152292f32, -3.3058951f32, -2.4513049f32, -1.4890089f32, -0.46128857f32, 0.58669996f32, 1.60891f32, 2.5604274f32, 3.3994441f32, 4.0890942f32, 4.5990791f32, 4.9069881f32, 4.9992933f32, 4.8719387f32];
const FP4_OUT: [f32; 32] = [4.0f32, 4.0f32, 4.0f32, 4.0f32, 3.0f32, 2.0f32, 1.5f32, 0.5f32, -0.5f32, -1.5f32, -3.0f32, -3.0f32, -4.0f32, -4.0f32, -4.0f32, -4.0f32, -4.0f32, -4.0f32, -4.0f32, -3.0f32, -2.0f32, -1.5f32, -0.5f32, 0.5f32, 1.5f32, 3.0f32, 3.0f32, 4.0f32, 4.0f32, 4.0f32, 4.0f32, 4.0f32];
const HAD_IN: [f32; 16] = [0.0f32, 1.3036675f32, 1.9773035f32, 1.6953558f32, 0.59408289f32, -0.79429626f32, -1.7988106f32, -1.9340029f32, -1.1345373f32, 0.21322313f32, 1.457938f32, 1.9980659f32, 1.5725765f32, 0.38709828f32, -0.98545545f32, -1.8817619f32];
const HAD_OUT: [f32; 16] = [0.66761184f32, 0.17393696f32, 0.4032954f32, -0.21272229f32, 3.0878963f32, -1.628741f32, -3.7764506f32, -0.98390156f32, -0.14596176f32, 0.076988816f32, 0.17850864f32, 0.046508059f32, 1.3667803f32, 0.35609549f32, 0.82565463f32, -0.43549949f32];
const ROPE_C_COS: [f32; 32] = [0.75390226f32, 0.10103078f32, -0.98583692f32, -0.64834672f32, 0.0055487626f32, 0.47454271f32, 0.73836076f32, 0.87324423f32, 0.93937272f32, 0.97117621f32, 0.98633534f32, 0.99353057f32, 0.99693906f32, 0.9985522f32, 0.99931526f32, 0.99967623f32, 0.99987423f32, 0.9999522f32, 0.9999823f32, 0.99999368f32, 0.99999785f32, 0.99999934f32, 0.99999982f32, 0.99999994f32, 1.0f32, 1.0f32, 1.0f32, 1.0f32, 1.0f32, 1.0f32, 1.0f32, 1.0f32];
const ROPE_N_COS: [f32; 32] = [0.75390226f32, 0.51144928f32, -0.7004298f32, -0.98205763f32, -0.59943748f32, -0.089047149f32, 0.32025701f32, 0.59505278f32, 0.76484221f32, 0.86536109f32, 0.92351949f32, 0.95674759f32, 0.97559988f32, 0.98625427f32, 0.99226242f32, 0.99564636f32, 0.99755102f32, 0.9986226f32, 0.99922532f32, 0.99956435f32, 0.99975502f32, 0.99986225f32, 0.99992251f32, 0.99995643f32, 0.9999755f32, 0.99998623f32, 0.99999225f32, 0.99999565f32, 0.99999756f32, 0.99999863f32, 0.99999923f32, 0.99999958f32];
const IDX_Q: [f32; 32] = [-0.82013452f32, 0.39563093f32, 0.89890796f32, -1.388404f32, -0.16699602f32, 0.28514996f32, -0.64109153f32, -0.89365512f32, 0.92654306f32, -0.53551239f32, -1.1597207f32, -0.4601568f32, 0.70853907f32, 1.0127554f32, 0.23039687f32, 1.0901654f32, -1.5826646f32, -0.32456669f32, 1.9263673f32, -0.33001173f32, 0.19844405f32, 0.78207266f32, 1.0391077f32, -0.72451133f32, -0.20934042f32, -0.21534145f32, -1.8157296f32, -0.34524196f32, -2.0614779f32, 0.6741007f32, -1.3233458f32, -1.3597685f32];
const IDX_KV: [f32; 40] = [-0.083528236f32, -0.023477932f32, 0.17437536f32, 2.2983427f32, 0.95710206f32, -0.66186756f32, -0.82845193f32, -0.60568017f32, -1.401251f32, 1.2973359f32, 1.6409333f32, -1.0566841f32, -0.2615872f32, -0.25013262f32, 0.50112164f32, 0.26003638f32, -0.17819262f32, -0.25949886f32, -0.014488148f32, -0.38389099f32, -2.9661698f32, -1.060555f32, -0.30899638f32, 0.9342882f32, 1.5496191f32, 0.59890002f32, -0.63766766f32, -2.2858188f32, -0.36766419f32, -0.88218081f32, 0.5460121f32, 0.14851166f32, -0.75565356f32, 0.39174548f32, 0.74698102f32, 1.3797723f32, 1.2877198f32, 0.8684383f32, -1.3821985f32, -0.96322864f32];
const IDX_W: [f32; 4] = [0.10724146f32, 0.61251658f32, 0.32964543f32, -0.87627965f32];
const IDX_SCORE: [f32; 5] = [0.0f32, 2.2086484f32, -4.1848159f32, 0.70370895f32, 0.88177973f32];
const CMP_APE: [f32; 64] = [-0.51079756f32, 1.0282716f32, -0.35315159f32, 0.12299131f32, -0.1815543f32, -1.4972281f32, 0.14210454f32, -0.52428246f32, -0.2487371f32, -0.5252381f32, 2.8922136f32, -0.59471339f32, 1.3118331f32, 0.35217938f32, -1.3151243f32, -0.0079922052f32, 0.24788003f32, 1.5727036f32, -1.6394643f32, -1.592532f32, -0.15462512f32, -1.0964388f32, 1.3665975f32, 0.68928212f32, -0.39350492f32, 0.61710012f32, 0.75284058f32, 0.60225254f32, 2.0175424f32, -1.1686294f32, -1.3241941f32, 1.1267436f32, -0.22554065f32, 0.52176046f32, -2.0597744f32, 0.13833788f32, 0.49617523f32, -0.60527956f32, -0.80069393f32, 0.1586518f32, 0.71277058f32, -0.74382192f32, 0.28007761f32, 0.37349072f32, 2.995054f32, -0.25694311f32, -0.68379736f32, 1.0621186f32, -0.020517835f32, -0.63050759f32, 0.54850787f32, -1.5885055f32, 0.52806395f32, 0.89639783f32, 0.99747843f32, -0.21557206f32, 1.3265953f32, -0.10864087f32, -1.026504f32, 0.043577146f32, -0.97238588f32, 0.28029084f32, 0.56998521f32, 1.4841164f32];
const CMP_KV: [f32; 160] = [-1.455569f32, 0.55822611f32, -0.50624758f32, 0.46549124f32, -0.86042666f32, -0.85281724f32, 2.0163476f32, 0.16927698f32, -0.90297776f32, -1.7102059f32, -0.13622876f32, 1.4530207f32, 0.56196922f32, 1.1590515f32, -1.7209787f32, -0.05897475f32, 0.84549505f32, -0.13036077f32, -0.32316297f32, -0.71186703f32, -0.22731686f32, -2.5321956f32, 0.34146237f32, 1.1048303f32, 0.095739298f32, -0.029930454f32, 0.85525882f32, -0.024889624f32, 2.2696486f32, 0.86017042f32, -1.6498927f32, -0.043486282f32, 0.44980073f32, -0.55059594f32, -0.82031542f32, 1.126418f32, -1.2375157f32, -1.7875451f32, 1.0114359f32, -0.99823642f32, 2.1535051f32, 0.4631384f32, -1.6792134f32, 1.4899623f32, -0.93873143f32, 2.0292625f32, 0.72740942f32, 0.40350321f32, 0.76419163f32, -0.84028643f32, 0.34638101f32, -2.8590417f32, -1.7517779f32, -2.9518678f32, 0.074530721f32, 0.28184995f32, 0.38899875f32, -2.2678823f32, -0.69501954f32, 0.85430861f32, 0.17583968f32, 0.66416925f32, -1.1717383f32, 1.1146483f32, -0.95542157f32, -1.0054712f32, 1.2386687f32, 1.9056864f32, -0.37328732f32, 0.62881529f32, 1.3660841f32, -0.7475912f32, 0.63186312f32, 0.88945228f32, 0.9692952f32, 1.3373623f32, 0.066716485f32, -1.1984706f32, -0.062956452f32, -0.35285097f32, 0.33095807f32, -0.5749687f32, 0.41262743f32, -0.37868625f32, 1.2536883f32, -0.13334212f32, -1.3512917f32, -2.3166363f32, 0.48251361f32, 0.71190643f32, 1.2839335f32, 1.3342987f32, -0.39085653f32, -0.84368002f32, 0.25546083f32, 0.24559443f32, 1.6759048f32, 0.36471808f32, -0.14541329f32, -1.3702646f32, 0.12358887f32, -1.1937128f32, -0.029780904f32, -0.16966693f32, 1.3884029f32, -1.4214729f32, 1.0762222f32, -0.46897268f32, 1.0490566f32, -0.3588827f32, 0.62510043f32, -0.82303882f32, -2.152257f32, -0.016438264f32, -0.39449576f32, -0.78302509f32, -2.1304581f32, -2.7913682f32, 0.4841305f32, 1.0717751f32, 1.3165028f32, 0.66716677f32, -0.03442286f32, 0.97439408f32, -0.34046382f32, 0.85737211f32, 0.44083524f32, 2.073169f32, -1.3578546f32, -1.018029f32, 0.41470191f32, -0.86280823f32, -0.46847308f32, 0.27619344f32, -0.5579381f32, -0.46987242f32, 0.38596866f32, 0.38086393f32, -1.0696658f32, -0.99098831f32, 1.3204088f32, -0.69438642f32, 0.72719115f32, 0.21854676f32, 0.63754869f32, 0.068582557f32, -0.88554806f32, 0.35674274f32, -0.16567725f32, 1.2104261f32, 2.2048571f32, 0.49145934f32, -0.037666298f32, -1.7789043f32, -0.49455383f32, -1.7143315f32, 0.27666509f32, -2.1497917f32, 0.8340748f32, 0.078504451f32];
const CMP_SC: [f32; 160] = [-1.6789142f32, -0.47343758f32, 0.0089645628f32, 1.3364344f32, -0.81350803f32, -1.5927582f32, 0.13512461f32, -1.7722744f32, -1.6928688f32, -0.10530683f32, 0.49980941f32, -0.258652f32, 0.87794906f32, -0.095455527f32, 0.94668818f32, 1.0368659f32, -1.2688787f32, -2.183985f32, 0.50629455f32, -1.0247545f32, -0.60502404f32, 0.63024801f32, 1.0517759f32, 1.1720454f32, 1.5522583f32, 0.49486375f32, 0.2495428f32, -0.53132719f32, -0.6395691f32, -0.44820726f32, -0.2441045f32, 0.12814213f32, -1.3670595f32, 0.75069493f32, 0.52746183f32, -0.35470411f32, -0.60974216f32, 1.3632002f32, -0.39027071f32, 1.4879911f32, 0.26610461f32, 0.11000671f32, -1.4545977f32, 1.0615697f32, 0.33698356f32, -1.9703135f32, 1.1549238f32, 1.5272647f32, -0.75560641f32, 1.1663266f32, -0.12521933f32, -0.86499244f32, 1.3109136f32, -0.067146614f32, -1.1960717f32, -0.52043319f32, 0.27422333f32, -0.48958066f32, -0.13342392f32, -1.1337765f32, -0.46569216f32, -0.17483181f32, 1.1599597f32, -1.3986576f32, 0.98087752f32, -0.81718296f32, 1.2057636f32, -0.20678645f32, 0.0058343904f32, -0.15392596f32, -0.41975671f32, 0.91835612f32, 0.70683223f32, 0.15416446f32, -0.16072103f32, -0.21779875f32, -1.0126163f32, 1.8229949f32, 2.3349133f32, 1.3726791f32, 1.0858138f32, -1.8123809f32, -1.5646379f32, -0.41847914f32, -1.3343127f32, -0.30792278f32, -0.31654659f32, -0.83140254f32, -0.78996402f32, -0.17778517f32, -0.51719731f32, -0.61740494f32, -1.2219005f32, -0.081317604f32, -1.1256471f32, 0.23964168f32, 0.28047326f32, -0.65849209f32, 0.49548468f32, -0.075882442f32, -0.56549829f32, 1.3210151f32, -0.16519029f32, -0.38562736f32, -0.3686178f32, 0.11275813f32, -1.4778583f32, -0.41756064f32, 0.37212715f32, 0.1001941f32, 1.0585976f32, -0.94637871f32, 0.47574544f32, 0.28220457f32, 0.95677024f32, 0.87098157f32, -0.43680406f32, 0.10696021f32, -0.62483478f32, 0.14915578f32, 0.95442468f32, -0.51637238f32, 0.068999626f32, 2.2702446f32, 0.057648271f32, -1.2478452f32, -1.2302866f32, 0.83361083f32, 0.78773004f32, -1.0709596f32, 0.031559341f32, -0.99307173f32, 0.18971902f32, -0.47665688f32, -1.3015931f32, 0.46314922f32, 0.34223092f32, 0.39149544f32, -0.72505313f32, 1.681388f32, -1.2595315f32, 0.70357102f32, 1.7356442f32, -0.92421132f32, -1.17133f32, 0.056980297f32, -1.4893285f32, -1.1018589f32, -0.043698892f32, 0.13495684f32, -0.45158052f32, -0.080328494f32, -0.042121351f32, -1.9375231f32, 1.1246296f32, -1.2424834f32, -0.24834189f32, 0.49694249f32, -0.20252977f32, 1.7064201f32];
const CMP_PREFILL: [f32; 16] = [0.71567887f32, -0.43036425f32, -0.074255586f32, 1.1825151f32, -0.29695901f32, 0.96887422f32, -0.85703456f32, 0.30275223f32, 1.1655338f32, -0.10859892f32, 0.83212125f32, 0.78381443f32, 0.41380566f32, -1.4899371f32, 0.42415801f32, 0.69569933f32];
const CMP_DEC_KV: [f32; 128] = [1.0282f32, -0.95809096f32, -1.320243f32, -0.70802402f32, 0.92432612f32, 1.8846222f32, 0.13138571f32, -2.486047f32, 0.099453844f32, -1.0940026f32, 0.25858685f32, 0.56409609f32, 0.66766644f32, -1.2707447f32, 0.41276461f32, 0.40766305f32, -1.5800474f32, 0.007379659f32, -0.48618326f32, 0.061569415f32, 0.42975178f32, -0.60182828f32, 1.0823132f32, 1.332293f32, -0.55761886f32, 0.062355578f32, -0.03189731f32, 1.3617002f32, 0.22838138f32, -0.74205154f32, -0.40100458f32, -0.76068038f32, 1.0557278f32, 0.92102432f32, 0.14263381f32, -0.21668084f32, 0.0026214465f32, -0.12831862f32, 1.2598408f32, 1.5755346f32, 0.58220273f32, -1.0701493f32, -0.44822925f32, 1.1731933f32, -0.02110873f32, -0.38200077f32, 1.2794687f32, 1.05804f32, -0.8772164f32, -0.46458787f32, -0.01347102f32, -0.22508675f32, 1.368569f32, 0.20904671f32, -0.57002407f32, 0.71144187f32, -0.36446813f32, -0.53331804f32, 1.478876f32, 0.3513284f32, -0.19162913f32, -0.48211795f32, 0.055907801f32, 0.60378271f32, -0.88811821f32, 1.5584488f32, -0.089344129f32, 0.40422574f32, -0.78774309f32, -0.039726578f32, -0.30835116f32, 0.36971536f32, 0.26936704f32, -0.69963849f32, 0.36456829f32, 1.0583906f32, 1.7094656f32, -0.2206326f32, 0.10612396f32, -0.55122882f32, 1.2651844f32, -0.61754388f32, 0.20248654f32, -0.15518482f32, -0.089570694f32, -0.87542146f32, 0.68425745f32, -0.25022376f32, 0.40873951f32, -0.79569668f32, -0.41997245f32, 1.1479857f32, 0.54626048f32, -0.23456897f32, 0.51019007f32, 1.0036157f32, 0.023057438f32, -0.79882908f32, 2.1892719f32, -0.52159756f32, -1.5882901f32, 0.68467206f32, -0.8386873f32, 0.38905752f32, -0.040119141f32, -2.3443041f32, 1.4730508f32, -0.74085724f32, -1.206928f32, -0.23498f32, -0.51273614f32, 0.13559154f32, 1.250139f32, -1.1715357f32, -0.25378764f32, 1.395489f32, -0.4245401f32, -0.26050907f32, 0.31821752f32, 0.21247984f32, 0.95375144f32, -0.32798478f32, -1.3795455f32, -0.24266037f32, -1.38599f32, 0.070030473f32, -0.47380054f32, 0.14259203f32];
const CMP_DEC_SC: [f32; 128] = [0.79823416f32, -0.93080282f32, -0.71673328f32, -1.0696759f32, 0.096262746f32, 0.27633426f32, 1.2209487f32, 0.93442923f32, -0.99210596f32, -0.47616288f32, 0.70488632f32, 0.025717556f32, 1.3657246f32, 1.4473208f32, -0.46700665f32, 0.7171281f32, -0.28620645f32, 0.14870869f32, 1.0555689f32, 1.4698336f32, -0.93807244f32, 0.96038878f32, -0.68813974f32, -0.29267713f32, -0.43563277f32, 0.14553894f32, -0.49930865f32, -0.78457808f32, 1.2475897f32, 0.16672069f32, -0.33338231f32, -0.083774365f32, -0.82711279f32, -0.20469582f32, 0.72464144f32, 0.28750136f32, 0.77856326f32, -0.4386985f32, 0.17768846f32, -0.13496919f32, 0.54871768f32, 0.031928942f32, -0.93412066f32, 0.50538987f32, 0.99466485f32, 2.2254913f32, -2.6320543f32, 1.7921826f32, -1.4252253f32, 1.1850488f32, -0.81649822f32, 0.81906664f32, 1.0534604f32, -1.9213799f32, -0.85834581f32, -0.54518491f32, -0.75722945f32, -0.11812954f32, 1.2678713f32, 0.90029889f32, -0.025864407f32, -1.5149385f32, 0.87858546f32, 0.82239789f32, 0.1917953f32, -0.39037141f32, -1.1913899f32, 0.75353128f32, 0.35611269f32, 0.49744195f32, 1.7558457f32, -0.21568428f32, -1.0923637f32, 0.82938874f32, -2.1549311f32, 0.50670904f32, 0.5765329f32, 1.3942415f32, 0.90749967f32, -0.68698919f32, -0.62599415f32, 1.415998f32, -0.25759619f32, 0.0079628294f32, 0.25370547f32, -0.9849503f32, 1.6819547f32, -0.91569716f32, 1.2923079f32, -0.12945357f32, 1.5176507f32, -2.0061421f32, -0.44905174f32, 0.9760111f32, 2.1541495f32, 0.69007456f32, -1.4235982f32, -0.54585218f32, 0.058994465f32, -0.81819212f32, -0.02371067f32, -0.86964494f32, 1.2285522f32, 1.4708543f32, 0.30330139f32, -1.8498753f32, 0.97850209f32, -0.43430129f32, -2.4426892f32, 1.6465185f32, -0.55297083f32, 1.1851395f32, 0.77718985f32, -0.08558175f32, -1.3931558f32, -0.36579853f32, -0.56406546f32, 0.54658407f32, 0.048680633f32, 0.89318496f32, 1.7125527f32, -0.1594919f32, -1.0620359f32, -0.54301322f32, -1.1869813f32, -0.70194107f32, -1.5105433f32, -0.16732219f32];
const CMP_DEC_AT: [usize; 2] = [1, 5];
const CMP_DECODE: [f32; 16] = [-0.21438542f32, -0.2943075f32, -0.42643994f32, -0.30879754f32, 0.59517753f32, -1.3683102f32, -0.13959113f32, -0.047694378f32, 0.2751978f32, -0.36406153f32, 0.23002705f32, 0.49375531f32, 1.0555544f32, -0.31154799f32, 0.6553899f32, 0.41941389f32];

    fn close(a: &[f32], b: &[f32], tol: f32, what: &str) {
        assert_eq!(a.len(), b.len(), "{what}: length");
        for (i, (x, y)) in a.iter().zip(b).enumerate() {
            assert!((x - y).abs() <= tol, "{what}[{i}]: {x} vs reference {y}");
        }
    }

    // FP8 activation simulation, block 64, ue8m0 power-of-two scale. Exact: both sides
    // land on representable e4m3 values times the same power of two.
    #[test]
    fn act_quant_matches_the_reference() {
        let mut x = ACT_IN;
        act_quant_sim(&mut x, 64);
        close(&x, &ACT_OUT, 0.0, "act_quant_sim");
    }

    // FP4 (E2M1) simulation, block 32 — `fp4_block_size`, the value this was blocked on.
    #[test]
    fn fp4_act_quant_matches_the_reference() {
        let mut y = FP4_IN;
        fp4_act_quant_sim(&mut y, 32);
        close(&y, &FP4_OUT, 0.0, "fp4_act_quant_sim");
    }

    // Hadamard rotation. Catches a transposed/sequency-ordered transform, which is
    // otherwise invisible: the norm is identical under any ordering.
    #[test]
    fn hadamard_matches_the_reference() {
        let mut h = HAD_IN;
        hadamard_rotate(&mut h, 16);
        close(&h, &HAD_OUT, 1e-5, "rotate_activation");
    }

    // The two rope tables. A compressor layer uses compress_rope_theta WITH YaRN; a layer
    // without one uses rope_theta and NO YaRN. Position 7 of each, so the angles are far
    // enough from zero to distinguish the tables (position 1 would not — see the YaRN
    // test that was itself the bug).
    #[test]
    fn both_rope_tables_match_the_reference() {
        let (c_cos, _) = yarn_rope_tables(64, 8, 160000.0, 16.0, 65536, 32.0, 1.0);
        close(&c_cos[7 * 32..8 * 32], &ROPE_C_COS, 1e-6, "compress rope (YaRN on)");
        let (n_cos, _) = yarn_rope_tables(64, 8, 10000.0, 16.0, 0, 32.0, 1.0);
        close(&n_cos[7 * 32..8 * 32], &ROPE_N_COS, 1e-6, "base rope (YaRN off)");
        // and they must actually DIFFER, or the test proves nothing about the split
        assert!(
            ROPE_C_COS.iter().zip(&ROPE_N_COS).any(|(a, b)| (a - b).abs() > 1e-3),
            "the two rope tables are identical — the per-layer split is meaningless"
        );
    }

    // Indexer scoring: relu FIRST, then the per-head weight, then the sum over HEADS.
    // Reordering those (weight before relu, or summing over t) keeps the shape and gives
    // a different ranking, which is exactly the kind of error top-k hides.
    #[test]
    fn indexer_scoring_matches_the_reference() {
        let (nh, hd, nt) = (4usize, 8usize, 5usize);
        let mut got = vec![0f32; nt];
        for (t, g) in got.iter_mut().enumerate() {
            let kr = &IDX_KV[t * hd..(t + 1) * hd];
            let mut acc = 0f32;
            for h in 0..nh {
                let qv = &IDX_Q[h * hd..(h + 1) * hd];
                let d: f32 = qv.iter().zip(kr).map(|(a, b)| a * b).sum();
                acc += d.max(0.0) * IDX_W[h];
            }
            *g = acc;
        }
        close(&got, &IDX_SCORE, 1e-4, "index_score");
    }

    // The Compressor's pooling, prefill AND the carry into decode. This is the piece the
    // whole long-context path rests on and it had only algorithm-level tests: at ratio 4
    // the windows OVERLAP, the softmax is per-DIMENSION over 2*ratio entries, and prefill
    // leaves a remainder in the carry state that later decode steps consume. A wrong carry
    // is slightly-off context, never a crash.
    #[test]
    fn compressor_prefill_matches_the_reference() {
        let (r, d, sq) = (4usize, 8usize, 10usize);
        let mut st = CompressorState::new(r, d);
        let got = compress_prefill(&CMP_KV, &CMP_SC, &CMP_APE, sq, r, d, &mut st);
        close(&got, &CMP_PREFILL, 1e-5, "compress_prefill");
    }

    // Decode must emit on exactly the steps the reference does (`(start_pos+1) % ratio`)
    // and with the same values, from state the prefill left behind. Running the prefill
    // first is the point — a fresh state would hide a carry bug completely.
    #[test]
    fn compressor_decode_matches_the_reference_after_a_prefill() {
        let (r, d, sq, w) = (4usize, 8usize, 10usize, 16usize);
        let mut st = CompressorState::new(r, d);
        let _ = compress_prefill(&CMP_KV, &CMP_SC, &CMP_APE, sq, r, d, &mut st);
        let (mut at, mut out) = (Vec::new(), Vec::new());
        for i in 0..8 {
            let kv = &CMP_DEC_KV[i * w..(i + 1) * w];
            let sc = &CMP_DEC_SC[i * w..(i + 1) * w];
            if let Some(row) = compress_decode(kv, sc, &CMP_APE, sq + i, &mut st) {
                at.push(i);
                out.extend_from_slice(&row);
            }
        }
        assert_eq!(at, CMP_DEC_AT, "emitted on different steps than the reference");
        close(&out, &CMP_DECODE, 1e-5, "compress_decode");
    }
}
