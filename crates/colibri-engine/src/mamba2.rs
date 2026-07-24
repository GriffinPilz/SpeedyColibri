//! Mamba2 selective-scan (state-space) mixer — the non-attention core of
//! Nemotron-H. Ported from `NemotronHMamba2Mixer.torch_forward` in the reference
//! `modeling_nemotron_h.py`.
//!
//! A Mamba2 layer replaces attention with a linear-time selective scan. Per layer,
//! given the block-normed input `x: [S, hidden]`, the forward is:
//!
//! ```text
//!   proj      = x @ in_proj.T                       # [S, 2*d_inner + 2*G*N + n_heads]
//!   gate, hBC, dt = split(proj, [d_inner, conv_dim, n_heads])   # d_mlp == 0 here
//!   hBC       = silu(causal_conv1d(hBC))            # depthwise, width d_conv
//!   h, B, C   = split(hBC, [d_inner, G*N, G*N])
//!   y         = selective_scan(h, B, C, dt; A, D, dt_bias)      # the recurrence
//!   y         = gated_rmsnorm(y, gate)             # per-group RMSNorm, silu gate
//!   out       = y @ out_proj.T                      # [S, hidden]
//! ```
//!
//! This module implements the three primitives that have no analog elsewhere in the
//! engine — [`causal_conv1d_silu`], [`selective_scan`], and [`gated_rmsnorm`] — on
//! plain `f32` slices. The two matmuls (`in_proj`/`out_proj`) reuse the existing
//! quantized-matmul path when this is wired into the model (a later phase).
//!
//! ## Selective-scan recurrence (per head `h`, head-dim `p`, state `n`)
//! `A[h] = -exp(A_log[h])` (input-independent); `B`/`C`/`dt` are input-dependent
//! (this is what makes the scan *selective*). B/C are shared across the
//! `heads_per_group = n_heads / n_groups` heads in a group. For each token `t`:
//! ```text
//!   dt_h      = softplus(dt[t,h] + dt_bias[h]); clamp to [dt_min, ∞)
//!   dA_h      = exp(dt_h * A[h])                     # scalar decay per head
//!   ssm[h,p,n] = ssm[h,p,n] * dA_h + dt_h * B[t,g,n] * h[t,h,p]
//!   y[t,h,p]  = Σ_n ssm[h,p,n] * C[t,g,n] + h[t,h,p] * D[h]
//! ```
//! Prefill is just this recurrence run over the sequence (`ssm` carried); decode is
//! one step of it against the persisted `ssm` state. A chunked/parallel prefill is a
//! later GPU-perf optimization — the sequential form here is the correctness ground
//! truth and matches the reference's naive path exactly.

/// Numerically-stable softplus: `ln(1 + e^x)`.
#[inline]
fn softplus(x: f32) -> f32 {
    // For large x, ln(1+e^x) ≈ x (avoids overflow); for small x use log1p(exp).
    if x > 20.0 {
        x
    } else {
        (x.exp()).ln_1p()
    }
}

/// SiLU / swish: `x * sigmoid(x)`.
#[inline]
fn silu(x: f32) -> f32 {
    x / (1.0 + (-x).exp())
}

/// Depthwise **causal** 1-D convolution over a sequence, followed by SiLU — the
/// `silu(conv1d(hBC))` step. `input` is `[seq, channels]` (row-major, one row per
/// token); `weight` is `[channels, k]` (per-channel kernel, `conv1d.weight`
/// squeezed from `[channels,1,k]`); `bias` is `[channels]` (or empty for no bias).
///
/// Causal padding: output at token `t`, channel `c` is
/// `Σ_{j=0..k} weight[c,j] * input[t - (k-1) + j, c]` (out-of-range = 0), i.e. the
/// kernel's last tap aligns with the current token. Returns `[seq, channels]`.
pub fn causal_conv1d_silu(
    input: &[f32],
    weight: &[f32],
    bias: &[f32],
    seq: usize,
    channels: usize,
    k: usize,
) -> Vec<f32> {
    assert_eq!(input.len(), seq * channels);
    assert_eq!(weight.len(), channels * k);
    assert!(bias.is_empty() || bias.len() == channels);
    let mut out = vec![0.0f32; seq * channels];
    for t in 0..seq {
        for c in 0..channels {
            let mut acc = if bias.is_empty() { 0.0 } else { bias[c] };
            for j in 0..k {
                // tap j reads token (t - (k-1) + j); skip if before the start.
                let off = (k - 1) - j;
                if t >= off {
                    acc += weight[c * k + j] * input[(t - off) * channels + c];
                }
            }
            out[t * channels + c] = silu(acc);
        }
    }
    out
}

/// Persistent per-layer Mamba2 recurrent state: `ssm[h, p, n]` flattened
/// row-major (`n_heads * head_dim * d_state`). Fresh state is all zeros.
#[derive(Clone)]
pub struct SsmState {
    pub data: Vec<f32>,
    pub n_heads: usize,
    pub head_dim: usize,
    pub d_state: usize,
}

impl SsmState {
    pub fn zeros(n_heads: usize, head_dim: usize, d_state: usize) -> Self {
        SsmState { data: vec![0.0f32; n_heads * head_dim * d_state], n_heads, head_dim, d_state }
    }
}

/// Geometry of a Mamba2 mixer, shared by prefill and decode.
#[derive(Clone, Copy)]
pub struct MambaDims {
    pub n_heads: usize,
    pub head_dim: usize,
    pub d_state: usize,
    pub n_groups: usize,
    /// Lower clamp on the discretization step `dt` (`time_step_min`, e.g. 0.001).
    pub dt_min: f32,
}

impl MambaDims {
    #[inline]
    fn d_inner(&self) -> usize {
        self.n_heads * self.head_dim
    }
    #[inline]
    fn heads_per_group(&self) -> usize {
        self.n_heads / self.n_groups
    }
}

/// Run the selective-scan recurrence over `seq` tokens, updating `state` in place
/// and returning the scan output `y: [seq, d_inner]` (row-major).
///
/// Inputs (all row-major, one row per token):
/// - `hidden`: `[seq, d_inner]` — the post-conv `h` split.
/// - `b`, `c`: `[seq, n_groups * d_state]` — input-dependent B/C, per group.
/// - `dt`: `[seq, n_heads]` — raw (pre-softplus) step.
/// - `a_log`, `d`, `dt_bias`: `[n_heads]` — `A = -exp(a_log)`, skip `D`, step bias.
///
/// With `seq == 1` and a persisted `state` this is the decode step; with `seq > 1`
/// and a zeroed `state` it is prefill. Either way `state` holds the final SSM state.
pub fn selective_scan(
    dims: MambaDims,
    state: &mut SsmState,
    hidden: &[f32],
    b: &[f32],
    c: &[f32],
    dt: &[f32],
    a_log: &[f32],
    d: &[f32],
    dt_bias: &[f32],
    seq: usize,
) -> Vec<f32> {
    let (nh, p, n, g) = (dims.n_heads, dims.head_dim, dims.d_state, dims.n_groups);
    let hpg = dims.heads_per_group();
    let d_inner = dims.d_inner();
    assert_eq!(hidden.len(), seq * d_inner);
    assert_eq!(b.len(), seq * g * n);
    assert_eq!(c.len(), seq * g * n);
    assert_eq!(dt.len(), seq * nh);
    assert_eq!(state.data.len(), nh * p * n);

    // A is input-independent: A[h] = -exp(a_log[h]).
    let a: Vec<f32> = a_log.iter().map(|&v| -(v.exp())).collect();
    let mut y = vec![0.0f32; seq * d_inner];

    for t in 0..seq {
        for h in 0..nh {
            let grp = h / hpg;
            let dt_h = softplus(dt[t * nh + h] + dt_bias[h]).max(dims.dt_min);
            let da_h = (dt_h * a[h]).exp();
            let d_h = d[h];
            let b_row = &b[t * g * n + grp * n..t * g * n + grp * n + n];
            let c_row = &c[t * g * n + grp * n..t * g * n + grp * n + n];
            for pp in 0..p {
                let x_hp = hidden[t * d_inner + h * p + pp];
                let base = h * p * n + pp * n;
                let ss = &mut state.data[base..base + n];
                let mut acc = 0.0f32;
                for nn in 0..n {
                    // ssm = ssm*dA + dt*B*x  ;  y += ssm*C
                    ss[nn] = ss[nn] * da_h + dt_h * b_row[nn] * x_hp;
                    acc += ss[nn] * c_row[nn];
                }
                y[t * d_inner + h * p + pp] = acc + x_hp * d_h;
            }
        }
    }
    y
}

/// Gated per-group RMSNorm — the Mamba `MambaRMSNormGated` with
/// `norm_before_gate = false`: `out = rmsnorm(y ⊙ silu(gate)) ⊙ weight`, where the
/// RMS is computed independently over each of `n_groups` contiguous groups of
/// `d_inner / n_groups` channels. `y`, `gate`, `out` are `[seq, d_inner]`;
/// `weight` is `[d_inner]`.
pub fn gated_rmsnorm(
    y: &[f32],
    gate: &[f32],
    weight: &[f32],
    seq: usize,
    d_inner: usize,
    n_groups: usize,
    eps: f32,
) -> Vec<f32> {
    assert_eq!(y.len(), seq * d_inner);
    assert_eq!(gate.len(), seq * d_inner);
    assert_eq!(weight.len(), d_inner);
    assert_eq!(d_inner % n_groups, 0);
    let gsz = d_inner / n_groups;
    let mut out = vec![0.0f32; seq * d_inner];
    for t in 0..seq {
        let row = t * d_inner;
        for grp in 0..n_groups {
            let off = row + grp * gsz;
            // gate then per-group RMS.
            let mut ss = 0.0f32;
            let mut gated: Vec<f32> = Vec::with_capacity(gsz);
            for i in 0..gsz {
                let v = y[off + i] * silu(gate[off + i]);
                gated.push(v);
                ss += v * v;
            }
            let inv = 1.0 / (ss / gsz as f32 + eps).sqrt();
            for i in 0..gsz {
                out[off + i] = gated[i] * inv * weight[grp * gsz + i];
            }
        }
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Hand-computed selective-scan recurrence (n_heads=1, head_dim=1, d_state=2,
    /// n_groups=1, seq=2). A = -1, D = 0.5, dt_bias = 0, dt raw = 0 →
    /// softplus(0) = ln2 ≈ 0.6931, dA = exp(-ln2) = 0.5.
    ///   t0: x=1, B=[1,0], C=[1,1]  → ssm=[0.6931, 0], y = 0.6931 + 1*0.5 = 1.1931
    ///   t1: x=1, B=[0,1], C=[1,1]  → ssm=[0.34657, 0.6931], y = 1.03972 + 0.5 = 1.53972
    #[test]
    fn scan_matches_hand_computed() {
        let dims = MambaDims { n_heads: 1, head_dim: 1, d_state: 2, n_groups: 1, dt_min: 0.0 };
        let mut st = SsmState::zeros(1, 1, 2);
        let hidden = vec![1.0, 1.0]; // [seq=2, d_inner=1]
        let b = vec![1.0, 0.0, 0.0, 1.0]; // [2, g*n=2]
        let c = vec![1.0, 1.0, 1.0, 1.0];
        let dt = vec![0.0, 0.0]; // [2, n_heads=1]
        let a_log = vec![0.0]; // A = -1
        let d = vec![0.5];
        let dt_bias = vec![0.0];
        let y = selective_scan(dims, &mut st, &hidden, &b, &c, &dt, &a_log, &d, &dt_bias, 2);
        assert!((y[0] - 1.1931).abs() < 1e-3, "y0={}", y[0]);
        assert!((y[1] - 1.53972).abs() < 1e-3, "y1={}", y[1]);
        // final ssm state carried: [0.34657, 0.6931]
        assert!((st.data[0] - 0.34657).abs() < 1e-3);
        assert!((st.data[1] - 0.6931).abs() < 1e-3);
    }

    /// Prefill (seq=2 in one call) must equal decode (two seq=1 calls carrying state).
    #[test]
    fn prefill_equals_stepwise_decode() {
        let dims = MambaDims { n_heads: 2, head_dim: 3, d_state: 4, n_groups: 1, dt_min: 0.001 };
        let d_inner = 6;
        let (g, n, nh) = (1, 4, 2);
        // deterministic pseudo-random-ish inputs
        let f = |i: usize| ((i as f32 * 0.37).sin() * 0.5) + 0.1;
        let hidden: Vec<f32> = (0..2 * d_inner).map(f).collect();
        let b: Vec<f32> = (0..2 * g * n).map(|i| f(i + 7)).collect();
        let c: Vec<f32> = (0..2 * g * n).map(|i| f(i + 13)).collect();
        let dt: Vec<f32> = (0..2 * nh).map(|i| f(i + 21)).collect();
        let a_log: Vec<f32> = (0..nh).map(|i| f(i + 31)).collect();
        let d: Vec<f32> = (0..nh).map(|i| f(i + 41)).collect();
        let dt_bias: Vec<f32> = (0..nh).map(|i| f(i + 51)).collect();

        let mut st_full = SsmState::zeros(nh, 3, n);
        let y_full =
            selective_scan(dims, &mut st_full, &hidden, &b, &c, &dt, &a_log, &d, &dt_bias, 2);

        let mut st_step = SsmState::zeros(nh, 3, n);
        let y0 = selective_scan(
            dims, &mut st_step, &hidden[..d_inner], &b[..g * n], &c[..g * n], &dt[..nh],
            &a_log, &d, &dt_bias, 1,
        );
        let y1 = selective_scan(
            dims, &mut st_step, &hidden[d_inner..], &b[g * n..], &c[g * n..], &dt[nh..],
            &a_log, &d, &dt_bias, 1,
        );
        for i in 0..d_inner {
            assert!((y_full[i] - y0[i]).abs() < 1e-6, "t0[{i}] {} vs {}", y_full[i], y0[i]);
            assert!(
                (y_full[d_inner + i] - y1[i]).abs() < 1e-6,
                "t1[{i}] {} vs {}",
                y_full[d_inner + i],
                y1[i]
            );
        }
        for i in 0..st_full.data.len() {
            assert!((st_full.data[i] - st_step.data[i]).abs() < 1e-6);
        }
    }

    /// B in the same group is shared across heads; different groups pick different B.
    #[test]
    fn groups_route_b_c_by_head() {
        // 2 heads, 1 group → both heads see the same B/C slice; result symmetric.
        let dims = MambaDims { n_heads: 2, head_dim: 1, d_state: 1, n_groups: 1, dt_min: 0.0 };
        let mut st = SsmState::zeros(2, 1, 1);
        let hidden = vec![1.0, 1.0]; // [seq1, d_inner=2] head0=1, head1=1
        let b = vec![2.0]; // [1, g*n=1]
        let c = vec![1.0];
        let dt = vec![0.0, 0.0];
        let a_log = vec![0.0, 0.0];
        let d = vec![0.0, 0.0];
        let dt_bias = vec![0.0, 0.0];
        let y = selective_scan(dims, &mut st, &hidden, &b, &c, &dt, &a_log, &d, &dt_bias, 1);
        assert!((y[0] - y[1]).abs() < 1e-9, "same group → same output");
    }

    #[test]
    fn causal_conv_is_causal() {
        // channels=1, k=2, weight=[0,1] (identity of current token), no bias, silu applied.
        let out = causal_conv1d_silu(&[1.0, 2.0, 3.0], &[0.0, 1.0], &[], 3, 1, 2);
        for (i, &x) in [1.0f32, 2.0, 3.0].iter().enumerate() {
            assert!((out[i] - silu(x)).abs() < 1e-6);
        }
    }
}
