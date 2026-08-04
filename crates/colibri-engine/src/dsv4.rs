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
/// DENSE over `[0, pos]`. That is exact only while every query's window covers the whole
/// cache — `seqlen <= sliding_window` (128) with no compressed entries. Beyond that V4
/// attends to a window plus compressed positions, and running dense is WRONG, not merely
/// slow: it attends to positions the model never does. Callers must enforce that.
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
    debug_assert_eq!(q.len(), s * h * hd);
    debug_assert_eq!(out.len(), s * h * hd);
    debug_assert_eq!(attn_sink.len(), h);
    let mut scores: Vec<f32> = Vec::new();
    for i in 0..s {
        let pos = pos_base + i;
        let n = pos + 1;
        debug_assert!(kv.len() >= n * hd, "kv cache shorter than the causal span");
        for hh in 0..h {
            let qv = &q[(i * h + hh) * hd..(i * h + hh) * hd + hd];
            scores.clear();
            let mut m = f32::NEG_INFINITY;
            for j in 0..n {
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
                *v = (*v - m).exp();
                den += *v;
            }
            // Sink: denominator only. Stabilised against the same max the scores used,
            // exactly as the reference kernel does.
            den += (attn_sink[hh] - m).exp();
            let dst = &mut out[(i * h + hh) * hd..(i * h + hh) * hd + hd];
            dst.fill(0.0);
            let inv = 1.0 / den;
            for (j, &w) in scores.iter().enumerate() {
                let w = w * inv;
                let vr = &kv[j * hd..(j + 1) * hd];
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
