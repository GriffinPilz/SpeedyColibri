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
