//! Normalization, RoPE, and elementwise activations — ports of `rmsnorm`,
//! `layernorm`, `rope_interleave`, `sigmoidf`, and `siluf` from `c/glm.c`.
//!
//! Mean/variance reductions accumulate in `f64` to match the C `double`
//! accumulators; the final scale and elementwise math stay `f32`.

/// RMSNorm: `out[i] = x[i] * w[i] / sqrt(mean(x^2) + eps)`. Port of `rmsnorm`.
pub fn rmsnorm(out: &mut [f32], x: &[f32], w: &[f32], eps: f32) {
    let d = x.len();
    debug_assert_eq!(out.len(), d);
    debug_assert_eq!(w.len(), d);
    let mut ms = 0f64;
    for &v in x {
        ms += v as f64 * v as f64;
    }
    let r = 1.0 / ((ms / d as f64) as f32 + eps).sqrt();
    for k in 0..d {
        out[k] = x[k] * r * w[k];
    }
}

/// In-place RMSNorm: `x[i] = x[i] * w[i] / sqrt(mean(x^2) + eps)`.
pub fn rmsnorm_inplace(x: &mut [f32], w: &[f32], eps: f32) {
    let d = x.len();
    debug_assert_eq!(w.len(), d);
    let mut ms = 0f64;
    for &v in x.iter() {
        ms += v as f64 * v as f64;
    }
    let r = 1.0 / ((ms / d as f64) as f32 + eps).sqrt();
    for k in 0..d {
        x[k] = x[k] * r * w[k];
    }
}

/// Plain softmax in place (max-subtract for stability). Port of the `softmax`
/// used by attention scores in `c/glm.c`.
pub fn softmax(x: &mut [f32]) {
    if x.is_empty() {
        return;
    }
    let m = x.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
    let mut s = 0f32;
    for v in x.iter_mut() {
        *v = (*v - m).exp();
        s += *v;
    }
    if s > 0.0 {
        for v in x.iter_mut() {
            *v /= s;
        }
    }
}

/// Classic LayerNorm (mean+variance, weight+bias), in place. Used by the DSA
/// indexer's `k_norm`. Port of `layernorm`.
pub fn layernorm(v: &mut [f32], w: &[f32], b: &[f32], eps: f32) {
    let n = v.len();
    debug_assert_eq!(w.len(), n);
    debug_assert_eq!(b.len(), n);
    let mut mu = 0f64;
    for &x in v.iter() {
        mu += x as f64;
    }
    mu /= n as f64;
    let mut var = 0f64;
    for &x in v.iter() {
        let d = x as f64 - mu;
        var += d * d;
    }
    var /= n as f64;
    let r = 1.0 / (var as f32 + eps).sqrt();
    for k in 0..n {
        v[k] = ((v[k] as f64 - mu) as f32) * r * w[k] + b[k];
    }
}

/// Interleaved partial RoPE on a `qk_rope`-length vector at position `pos`.
/// Port of `rope_interleave`.
///
/// Consumes interleaved pairs `(v[2j], v[2j+1])` and writes the rotated result
/// into the split halves `v[j]` and `v[half+j]` (GLM's layout). `theta` is the
/// rope base.
pub fn rope_interleave(v: &mut [f32], pos: usize, qk_rope: usize, theta: f32) {
    let half = qk_rope / 2;
    debug_assert!(v.len() >= qk_rope);
    // snapshot the input, since outputs overwrite positions we still read
    let mut input = [0f32; 512];
    debug_assert!(qk_rope <= input.len(), "qk_rope exceeds rope scratch");
    input[..qk_rope].copy_from_slice(&v[..qk_rope]);
    for j in 0..half {
        let inv = theta.powf(-2.0 * j as f32 / qk_rope as f32);
        let ang = pos as f32 * inv;
        let (sn, cs) = ang.sin_cos();
        let a = input[2 * j];
        let b = input[2 * j + 1];
        v[j] = a * cs - b * sn;
        v[half + j] = b * cs + a * sn;
    }
}

/// NeoX "rotate-half" partial RoPE on a `dim`-length vector at position `pos`
/// (the HF `rotate_half` convention used by MiniMax-M3, GPT-NeoX, Llama, etc.).
///
/// Pairs dimension `j` with `j + dim/2` (contiguous halves), NOT the interleaved
/// `(2j, 2j+1)` pairs of [`rope_interleave`]. `q_embed[j] = q[j]·cos − q[half+j]·sin`,
/// `q_embed[half+j] = q[half+j]·cos + q[j]·sin`, with `freq_j = theta^(-2j/dim)`.
pub fn rope_neox(v: &mut [f32], pos: usize, dim: usize, theta: f32) {
    let half = dim / 2;
    debug_assert!(v.len() >= dim);
    for j in 0..half {
        let inv = theta.powf(-2.0 * j as f32 / dim as f32);
        let ang = pos as f32 * inv;
        let (sn, cs) = ang.sin_cos();
        let a = v[j];
        let b = v[half + j];
        v[j] = a * cs - b * sn;
        v[half + j] = b * cs + a * sn;
    }
}

/// Sigmoid. Port of `sigmoidf`.
#[inline]
pub fn sigmoid(x: f32) -> f32 {
    1.0 / (1.0 + (-x).exp())
}

/// SiLU / swish: `x * sigmoid(x)`. Port of `siluf`.
#[inline]
pub fn silu(x: f32) -> f32 {
    x / (1.0 + (-x).exp())
}

/// Clamped OpenAI-SwiGLU (`swigluoai`, MiniMax-M3 / gpt-oss). Combines the gate
/// and up projections of one element:
///   `gate` is clamped to `max = limit`; `up` is clamped to `[-limit, limit]`;
///   the gated term is `gate · sigmoid(alpha · gate)` (SiLU with an `alpha`
///   pre-scale), and the result is `(up + 1) · gated`.
/// Reduces to standard SwiGLU as `alpha → 1`, `limit → ∞`, minus the `up + 1`
/// shift. NOTE: verify the exact formulation against the reference at end-to-end
/// validation (task #56) before trusting generations.
#[inline]
pub fn swiglu_oai(gate: f32, up: f32, alpha: f32, limit: f32) -> f32 {
    let g = gate.min(limit); // clamp upper only
    let u = up.clamp(-limit, limit);
    let gated = g / (1.0 + (-alpha * g).exp()); // g * sigmoid(alpha * g)
    gated * (u + 1.0)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn rmsnorm_gives_unit_rms_with_unit_weight() {
        let x = [3.0f32, 4.0];
        let w = [1.0f32, 1.0];
        let mut out = [0f32; 2];
        rmsnorm(&mut out, &x, &w, 0.0);
        let rms = ((out[0] * out[0] + out[1] * out[1]) / 2.0).sqrt();
        assert!((rms - 1.0).abs() < 1e-5, "rms was {rms}");
        // direction preserved
        assert!((out[1] / out[0] - 4.0 / 3.0).abs() < 1e-5);
    }

    #[test]
    fn layernorm_zero_mean_unit_var() {
        let mut v = [1.0f32, 2.0, 3.0];
        let w = [1.0f32; 3];
        let b = [0.0f32; 3];
        layernorm(&mut v, &w, &b, 0.0);
        let mean: f32 = v.iter().sum::<f32>() / 3.0;
        assert!(mean.abs() < 1e-5);
        let var: f32 = v.iter().map(|x| (x - mean).powi(2)).sum::<f32>() / 3.0;
        assert!((var - 1.0).abs() < 1e-4, "var {var}");
        assert!(v[0] < 0.0 && v[2] > 0.0);
    }

    #[test]
    fn rope_at_pos_zero_is_identity() {
        // pos 0 -> all angles 0 -> cos=1,sin=0, but note the layout still
        // de-interleaves (v[2j],v[2j+1]) -> (v[j], v[half+j]).
        let mut v = [1.0f32, 2.0, 3.0, 4.0]; // qk_rope=4, half=2
        rope_interleave(&mut v, 0, 4, 10000.0);
        // j=0: a=1,b=2 -> v[0]=1, v[2]=2 ; j=1: a=3,b=4 -> v[1]=3, v[3]=4
        assert_eq!(v, [1.0, 3.0, 2.0, 4.0]);
    }

    #[test]
    fn rope_rotation_preserves_pair_norm() {
        let mut v = [1.0f32, 0.0, 0.0, 1.0];
        let before0 = v[0] * v[0] + v[1] * v[1];
        rope_interleave(&mut v, 5, 4, 10000.0);
        // pair j=0 maps to (v[0], v[2]); its norm is preserved by the rotation
        let after0 = v[0] * v[0] + v[2] * v[2];
        assert!((before0 - after0).abs() < 1e-5);
    }

    #[test]
    fn silu_and_sigmoid() {
        assert!((sigmoid(0.0) - 0.5).abs() < 1e-6);
        assert!((silu(0.0)).abs() < 1e-6);
        assert!((silu(1.0) - 1.0 * sigmoid(1.0)).abs() < 1e-6);
    }

    #[test]
    fn swiglu_oai_shape() {
        let (alpha, limit) = (1.702f32, 7.0f32);
        // At gate=0: gated term is 0, so the whole product is 0 regardless of up.
        assert!(swiglu_oai(0.0, 3.0, alpha, limit).abs() < 1e-6);
        // Unclamped mid-range matches the definition exactly.
        let g = 1.5f32;
        let u = 2.0f32;
        let expect = (g / (1.0 + (-alpha * g).exp())) * (u + 1.0);
        assert!((swiglu_oai(g, u, alpha, limit) - expect).abs() < 1e-6);
        // Clamping: gate clamps at max=limit, up clamps to [-limit, limit].
        let big = swiglu_oai(100.0, 100.0, alpha, limit);
        let clamped = (limit / (1.0 + (-alpha * limit).exp())) * (limit + 1.0);
        assert!((big - clamped).abs() < 1e-4, "got {big}, want {clamped}");
        // up clamps on the negative side too.
        let neg = swiglu_oai(2.0, -50.0, alpha, limit);
        let g2 = 2.0f32;
        let expect_neg = (g2 / (1.0 + (-alpha * g2).exp())) * (-limit + 1.0);
        assert!((neg - expect_neg).abs() < 1e-5);
    }
}
