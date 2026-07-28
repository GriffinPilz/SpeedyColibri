//! Kimi Delta Attention — the linear mixer on 69 of Kimi-K3's 93 layers.
//!
//! A delta rule: instead of a context-growing KV cache, each head carries a fixed
//! `[K, V]` association matrix `S` between steps, so memory is O(1) in context.
//! Per token, per head:
//!
//! ```text
//! S  <- S * exp(g)            (row-wise: g is per key-dim)
//! S  <- S + beta * k (x) (v - kᵀS)
//! o   = qᵀS
//! ```
//!
//! Ported against `fla` (fla-org/flash-linear-attention) — `fla/ops/kda/naive.py`
//! for the recurrence, `fla/ops/kda/gate.py` for the decay, and
//! `fla/modules/fused_norm_gate.py` for the output norm. K3's own
//! `modeling_kimi_linear.py` only forwards to those kernels, so it does not pin the
//! semantics on its own.
//!
//! # Two places K3 departs from the reference, both silent if got wrong
//!
//! **`A_log` is per key-dim, not per head.** `fla`'s gate does
//! `A_log.view(H, 1)` and its Triton kernel indexes `A_log + i_h` — per head. That
//! matches Kimi-Linear-48B, which ships `A_log [1,1,32,1]` for 32 heads. K3 ships
//! `A_log [128]` with `num_heads = 96`, `head_dim = 128` — verified on the original
//! `moonshotai/Kimi-K3`, so it is not a repack artifact. 128 cannot `view(96, 1)`,
//! so the only shape-consistent reading is a broadcast across *heads*, per key-dim.
//! Porting `fla`'s gate unmodified would broadcast the wrong way and produce
//! plausible-but-wrong output rather than an error.
//!
//! **The gate is the lower-bounded form.** K3 sets `gate_lower_bound = -5.0`, which
//! makes `USE_LOWER_BOUND` true, selecting
//! `g = lower_bound * sigmoid(exp(A_log) * (g_raw + dt_bias))` — NOT the
//! `-exp(A_log) * softplus(...)` form that `naive_kda_gate` shows first. The bound
//! keeps `g` in `(-5, 0)`, so the per-step decay `exp(g)` stays in `(0.0067, 1)`.

use crate::linear::matmul_qt;
use crate::mamba2::causal_conv1d_silu;
use crate::math::sigmoid;
use crate::model::{KvCache, Layer};
use colibri_core::Config;

/// Geometry of one KDA layer, all derived from config so a shape mistake surfaces
/// as an assert here rather than as silently wrong numerics deeper in.
pub struct KdaDims {
    /// attention heads
    pub h: usize,
    /// per-head key/value dim (K == V on K3)
    pub dk: usize,
    /// `h * dk`, the projection width
    pub c: usize,
    /// short-conv kernel width
    pub k: usize,
}

impl KdaDims {
    pub fn new(cfg: &Config) -> KdaDims {
        let (h, dk) = (cfg.kda_n_heads as usize, cfg.kda_head_dim as usize);
        KdaDims { h, dk, c: h * dk, k: cfg.kda_d_conv as usize }
    }
}

/// `g = lower_bound * sigmoid(exp(A_log) * (g_raw + dt_bias))`, in place.
///
/// `g_raw` is `[s, h, dk]` flattened; `dt_bias` is `[h * dk]`; `a_log` is **`[dk]`**
/// and broadcasts across heads (see the module note — this is where K3 differs from
/// `fla`, which indexes it per head).
pub fn kda_gate(g: &mut [f32], a_log: &[f32], dt_bias: &[f32], lower_bound: f32, d: &KdaDims) {
    assert_eq!(a_log.len(), d.dk, "K3 A_log is per key-dim, expected {}", d.dk);
    assert_eq!(dt_bias.len(), d.c);
    // exp(A_log) is loop-invariant across tokens and heads; hoist it.
    let ea: Vec<f32> = a_log.iter().map(|v| v.exp()).collect();
    for row in g.chunks_mut(d.c) {
        for hh in 0..d.h {
            for (i, &a) in ea.iter().enumerate() {
                let j = hh * d.dk + i;
                row[j] = lower_bound * sigmoid(a * (row[j] + dt_bias[j]));
            }
        }
    }
}

/// L2-normalise each `[dk]` head vector of a `[s, h, dk]` buffer, in place.
/// `use_qk_l2norm_in_kernel=True` in K3's call, applied to q and k.
pub fn l2norm_heads(x: &mut [f32], d: &KdaDims, eps: f32) {
    for row in x.chunks_mut(d.c) {
        for hh in 0..d.h {
            let v = &mut row[hh * d.dk..(hh + 1) * d.dk];
            let ss: f32 = v.iter().map(|a| a * a).sum();
            let inv = 1.0 / (ss.sqrt() + eps);
            for a in v.iter_mut() {
                *a *= inv;
            }
        }
    }
}

/// `out = rmsnorm(y) * weight * sigmoid(gate)`, per head.
///
/// This is `fla`'s `FusedRMSNormGated(activation='sigmoid')`: normalise, THEN scale
/// by weight, THEN gate. Deliberately not `mamba2::gated_rmsnorm`, which gates
/// *before* normalising and uses silu — same name, different function.
pub fn gated_rmsnorm_sigmoid(
    y: &[f32],
    gate: &[f32],
    weight: &[f32],
    d: &KdaDims,
    eps: f32,
    out: &mut [f32],
) {
    assert_eq!(weight.len(), d.dk);
    for ((yr, gr), or) in y.chunks(d.c).zip(gate.chunks(d.c)).zip(out.chunks_mut(d.c)) {
        for hh in 0..d.h {
            let (a, b) = (hh * d.dk, (hh + 1) * d.dk);
            let ss: f32 = yr[a..b].iter().map(|v| v * v).sum();
            let inv = 1.0 / (ss / d.dk as f32 + eps).sqrt();
            for i in 0..d.dk {
                or[a + i] = yr[a + i] * inv * weight[i] * sigmoid(gr[a + i]);
            }
        }
    }
}

/// The delta-rule recurrence over `s` tokens, advancing `state` in place.
///
/// `state` is `[h, dk, dk]` (K-major: `state[((hh*dk)+kk)*dk + vv]`). `q`/`k`/`v` and
/// `g` are `[s, h, dk]`; `beta` is `[s, h]` **pre-sigmoid** (applied here, matching
/// `use_beta_sigmoid_in_kernel`). Writes `[s, h, dk]` into `o`.
///
/// Order matters: the state is updated with the current token BEFORE the output is
/// read, so `o` reflects the just-written association (`fla/ops/kda/naive.py`).
#[allow(clippy::too_many_arguments)] // q/k/v/g/beta/dims/state/out: all load-bearing
pub fn kda_recurrent(
    q: &[f32],
    k: &[f32],
    v: &[f32],
    g: &[f32],
    beta: &[f32],
    d: &KdaDims,
    state: &mut [f32],
    o: &mut [f32],
) {
    assert_eq!(state.len(), d.h * d.dk * d.dk);
    let s = q.len() / d.c;
    let mut kts = vec![0.0f32; d.dk]; // kᵀS scratch, [V]
    for t in 0..s {
        let row = t * d.c;
        for hh in 0..d.h {
            let base = hh * d.dk;
            let sh = &mut state[hh * d.dk * d.dk..(hh + 1) * d.dk * d.dk];
            let (qh, kh, vh, gh) = (
                &q[row + base..row + base + d.dk],
                &k[row + base..row + base + d.dk],
                &v[row + base..row + base + d.dk],
                &g[row + base..row + base + d.dk],
            );
            // 1) decay: scale row kk of S by exp(g[kk]).
            for kk in 0..d.dk {
                let e = gh[kk].exp();
                for vv in 0..d.dk {
                    sh[kk * d.dk + vv] *= e;
                }
            }
            // 2) delta: u = v - kᵀS, then S += beta * k (x) u.
            for x in kts.iter_mut() {
                *x = 0.0;
            }
            for kk in 0..d.dk {
                let kv = kh[kk];
                if kv != 0.0 {
                    for vv in 0..d.dk {
                        kts[vv] += kv * sh[kk * d.dk + vv];
                    }
                }
            }
            let b = sigmoid(beta[t * d.h + hh]);
            for kk in 0..d.dk {
                let bk = b * kh[kk];
                if bk != 0.0 {
                    for vv in 0..d.dk {
                        sh[kk * d.dk + vv] += bk * (vh[vv] - kts[vv]);
                    }
                }
            }
            // 3) read out: o = qᵀS (against the UPDATED state).
            let oh = &mut o[row + base..row + base + d.dk];
            for x in oh.iter_mut() {
                *x = 0.0;
            }
            for kk in 0..d.dk {
                let qv = qh[kk];
                if qv != 0.0 {
                    for vv in 0..d.dk {
                        oh[vv] += qv * sh[kk * d.dk + vv];
                    }
                }
            }
        }
    }
}

/// Run one KDA layer over `x[s * hidden]`, writing `out[s * hidden]`.
///
/// `x` is already `in_ln`-normalised by the layer driver; `out` is the mixer result
/// before the residual add. Advances this layer's conv history and association state
/// in `kv`.
pub fn kda_mixer(
    cfg: &Config,
    l: &Layer,
    kv: &mut KvCache,
    layer: usize,
    x: &[f32],
    s: usize,
    out: &mut [f32],
) {
    let d = KdaDims::new(cfg);

    // Projections. q/k/v then the two gates.
    let proj = |w: &Option<colibri_core::QTensor>, o: usize| -> Vec<f32> {
        let mut y = vec![0.0f32; s * o];
        matmul_qt(&mut y, x, w.as_ref().expect("kda projection"), s);
        y
    };
    let mut q = proj(&l.q_proj, d.c);
    let mut k = proj(&l.k_proj, d.c);
    let mut v = proj(&l.v_proj, d.c);

    // Short causal conv + SiLU on each of q/k/v, carrying `k-1` tokens of history so a
    // decode step convolves against its predecessors instead of zeros.
    //
    // The history is of the PRE-conv projections, not the conv output: the kernel's
    // taps read the projection sequence. Storing the post-conv values instead would
    // feed each step's own SiLU'd output back in as if it were the previous token.
    let pad = d.k - 1;
    let hist = kv.kda_conv_take(layer, d.c, d.k);
    let mut carries = vec![0.0f32; 3 * pad * d.c];
    for (idx, (buf, w)) in
        [(&mut q, &l.kda_conv_q), (&mut k, &l.kda_conv_k), (&mut v, &l.kda_conv_v)]
            .into_iter()
            .enumerate()
    {
        let prev = &hist[idx * pad * d.c..(idx + 1) * pad * d.c];
        // `prev ++ buf`, convolved, then the padding rows dropped.
        let mut full = Vec::with_capacity((pad + s) * d.c);
        full.extend_from_slice(prev);
        full.extend_from_slice(buf);
        let out_full = causal_conv1d_silu(&full, w, &[], pad + s, d.c, d.k);
        *buf = out_full[pad * d.c..].to_vec();
        // Next step's history is the last `pad` rows of the PRE-conv sequence, which
        // for s < pad correctly keeps part of the old carry.
        carries[idx * pad * d.c..(idx + 1) * pad * d.c]
            .copy_from_slice(&full[full.len() - pad * d.c..]);
    }
    kv.kda_conv_store(layer, &carries);

    // q/k are L2-normalised per head, then q is scaled by 1/sqrt(dk).
    l2norm_heads(&mut q, &d, 1e-6);
    l2norm_heads(&mut k, &d, 1e-6);
    let scale = (d.dk as f32).powf(-0.5);
    for a in q.iter_mut() {
        *a *= scale;
    }

    // Decay gate: g_raw = f_b(f_a(x)), then the lower-bounded sigmoid form.
    let r = l.kda_f_a.as_ref().expect("f_a").o as usize;
    let mut low = vec![0.0f32; s * r];
    matmul_qt(&mut low, x, l.kda_f_a.as_ref().unwrap(), s);
    let mut g = vec![0.0f32; s * d.c];
    matmul_qt(&mut g, &low, l.kda_f_b.as_ref().expect("f_b"), s);
    kda_gate(&mut g, &l.kda_a_log, &l.kda_dt_bias, K3_GATE_LOWER_BOUND, &d);

    // beta is raw here; `kda_recurrent` applies the sigmoid.
    let mut beta = vec![0.0f32; s * d.h];
    matmul_qt(&mut beta, x, l.kda_b_proj.as_ref().expect("b_proj"), s);

    let mut o = vec![0.0f32; s * d.c];
    let state = kv.kda_state_mut(layer);
    kda_recurrent(&q, &k, &v, &g, &beta, &d, state, &mut o);

    // Output gate (full-rank g_proj) into the gated RMSNorm, then o_proj.
    let mut gate = vec![0.0f32; s * d.c];
    matmul_qt(&mut gate, x, l.attn_gate.as_ref().expect("g_proj"), s);
    let mut normed = vec![0.0f32; s * d.c];
    gated_rmsnorm_sigmoid(&o, &gate, &l.kda_o_norm, &d, cfg.eps, &mut normed);
    matmul_qt(out, &normed, &l.o, s);
}

/// K3's `linear_attn_config.gate_lower_bound`. Hard-coded rather than read from
/// config because it selects the gate FORM (`USE_LOWER_BOUND`), not just a constant.
const K3_GATE_LOWER_BOUND: f32 = -5.0;


#[cfg(test)]
mod tests {
    use super::*;

    const H: usize = 2;
    const DK: usize = 3;
    const S: usize = 4;

    fn dims() -> KdaDims {
        KdaDims { h: H, dk: DK, c: H * DK, k: 4 }
    }
    // Deterministic inputs, generated by the same closed forms the Python reference
    // used, so the two agree without pasting 24 literals for each buffer.
    fn qv() -> Vec<f32> { (0..S * H * DK).map(|i| ((i * 7 % 13) as f32) / 13.0 - 0.5).collect() }
    fn kv_() -> Vec<f32> { (0..S * H * DK).map(|i| ((i * 5 % 11) as f32) / 11.0 - 0.5).collect() }
    fn vv() -> Vec<f32> { (0..S * H * DK).map(|i| ((i * 3 % 17) as f32) / 17.0 - 0.5).collect() }
    fn gv() -> Vec<f32> { (0..S * H * DK).map(|i| -0.1 - 0.05 * ((i * 2 % 5) as f32)).collect() }
    fn bv() -> Vec<f32> { (0..S * H).map(|i| 0.3 * ((i % 4) as f32 - 1.5)).collect() }

    /// The delta rule, checked value-by-value against `fla/ops/kda/naive.py`
    /// (`naive_recurrent_kda`) re-run on these exact inputs. This pins the three things
    /// that are easy to get subtly wrong and impossible to spot downstream: that the
    /// decay scales S row-wise by key-dim, that the update subtracts kᵀS *before*
    /// writing, and that the output reads the UPDATED state.
    ///
    /// The constants below are only worth anything if they really came from the
    /// reference rather than from an earlier run of this same code, so that is
    /// re-checkable: `python3 scripts/kda_ref_check.py` reproduces all 27 of them from an
    /// independent transcription of the fla source (verified 2026-07-28, max diff
    /// < 2e-6). Re-run it if you change the inputs or the expectations.
    #[test]
    fn recurrence_matches_fla_naive_reference() {
        let d = dims();
        let scale = (DK as f32).powf(-0.5);
        let q: Vec<f32> = qv().iter().map(|a| a * scale).collect();
        let (k, v, g, beta) = (kv_(), vv(), gv(), bv());
        let mut state = vec![0.0f32; H * DK * DK];
        let mut o = vec![0.0f32; S * H * DK];
        kda_recurrent(&q, &k, &v, &g, &beta, &d, &mut state, &mut o);

        #[rustfmt::skip]
        let want: [f32; 24] = [
            -0.008449558, -0.005467361, -0.002485164, -0.001332029, -0.009324205, -0.01731638,
             0.02029327,   0.01201599,   0.003738716, -0.01205965,  -0.03982272,  -0.06758579,
            -0.003106576, -0.003155352, -0.003204128, -0.0005213997, 0.008427472,  0.08134121,
             0.01579968,   0.005727862, -0.004343956, -0.01423591,  -0.02219233,   0.06456562,
        ];
        for (i, (got, exp)) in o.iter().zip(want.iter()).enumerate() {
            assert!((got - exp).abs() < 2e-6, "o[{i}]: got {got}, want {exp}");
        }
        // The carried state must match too — a recurrence can produce the right first
        // outputs and still drift if the state update is wrong.
        for (i, exp) in [0.08012541f32, 0.04224312, 0.004360832].iter().enumerate() {
            assert!((state[i] - exp).abs() < 2e-6, "state[0,0,{i}]: {} vs {exp}", state[i]);
        }
    }

    /// Running one token at a time through a carried state must equal running the whole
    /// sequence at once — the property decode depends on.
    #[test]
    fn stepwise_equals_batched() {
        let d = dims();
        let (q, k, v, g, beta) = (qv(), kv_(), vv(), gv(), bv());
        let mut s_all = vec![0.0f32; H * DK * DK];
        let mut o_all = vec![0.0f32; S * H * DK];
        kda_recurrent(&q, &k, &v, &g, &beta, &d, &mut s_all, &mut o_all);

        let mut s_step = vec![0.0f32; H * DK * DK];
        let mut o_step = vec![0.0f32; S * H * DK];
        for t in 0..S {
            let (a, b) = (t * d.c, (t + 1) * d.c);
            let mut o1 = vec![0.0f32; d.c];
            kda_recurrent(&q[a..b], &k[a..b], &v[a..b], &g[a..b],
                          &beta[t * H..(t + 1) * H], &d, &mut s_step, &mut o1);
            o_step[a..b].copy_from_slice(&o1);
        }
        for i in 0..o_all.len() {
            assert!((o_all[i] - o_step[i]).abs() < 1e-6, "token-at-a-time diverged at {i}");
        }
    }

    /// `A_log` broadcasts across HEADS (one entry per key-dim), which is where K3
    /// departs from `fla`/Kimi-Linear. If it were read per head instead, two heads
    /// would get different factors at the same key-dim — this asserts they do not.
    #[test]
    fn a_log_broadcasts_across_heads_not_dims() {
        let d = dims();
        let a_log = vec![0.25f32, -0.5, 1.0]; // one per key-dim
        let dt_bias = vec![0.0f32; H * DK];
        // Same raw gate value everywhere, so any difference comes only from A_log.
        let mut g = vec![0.7f32; H * DK];
        kda_gate(&mut g, &a_log, &dt_bias, -5.0, &d);
        for i in 0..DK {
            assert!((g[i] - g[DK + i]).abs() < 1e-7,
                    "key-dim {i} differs between heads: {} vs {}", g[i], g[DK + i]);
            let want = -5.0 * sigmoid(a_log[i].exp() * 0.7);
            assert!((g[i] - want).abs() < 1e-6, "dim {i}: {} vs {want}", g[i]);
        }
        // Distinct key-dims must NOT collapse to the same value.
        assert!((g[0] - g[1]).abs() > 1e-3, "A_log is not varying across key-dims");
        // The lower bound keeps the decay usable: exp(g) in (exp(-5), 1).
        for v in &g {
            assert!(*v > -5.0 && *v < 0.0, "gate {v} outside (-5, 0)");
        }
    }

    /// `out = rmsnorm(y) * weight * sigmoid(gate)` — normalise, then weight, then gate.
    /// Mamba's `gated_rmsnorm` gates BEFORE normalising and uses silu; mixing them up
    /// changes the result without changing any shape.
    #[test]
    fn output_norm_is_normalise_then_weight_then_sigmoid_gate() {
        let d = KdaDims { h: 1, dk: 4, c: 4, k: 4 };
        let y = vec![1.0f32, -2.0, 3.0, -4.0];
        let gate = vec![0.5f32; 4];
        let w = vec![2.0f32, 1.0, 0.5, 1.0];
        let mut out = vec![0.0f32; 4];
        gated_rmsnorm_sigmoid(&y, &gate, &w, &d, 1e-6, &mut out);
        let ms = y.iter().map(|v| v * v).sum::<f32>() / 4.0;
        let inv = 1.0 / (ms + 1e-6).sqrt();
        for i in 0..4 {
            let want = y[i] * inv * w[i] * sigmoid(0.5);
            assert!((out[i] - want).abs() < 1e-6, "{i}: {} vs {want}", out[i]);
        }
    }
}
