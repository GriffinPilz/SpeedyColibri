//! Token sampling — temperature + nucleus (top-p).
//!
//! Port of the sampling path in `c/glm.c`. The engine's defaults are tuned for
//! int4 reality (temp 0.7 / top-p 0.90) rather than the official 1.0 / 0.95,
//! which samples quantization noise from the tail.
//!
//! `argmax` (greedy) is fully implemented and is what `DRAFT=0` byte-exact runs
//! use. Nucleus sampling takes an injected uniform draw so it is deterministic
//! and testable; the engine supplies a real RNG.

/// Sampling configuration.
#[derive(Debug, Clone, Copy)]
pub struct SampleConfig {
    pub temperature: f32,
    pub top_p: f32,
}

impl Default for SampleConfig {
    fn default() -> Self {
        // colibrì's int4-tuned defaults.
        SampleConfig {
            temperature: 0.7,
            top_p: 0.90,
        }
    }
}

/// Greedy: the index of the maximum logit. Ties go to the lowest index (matching
/// the C `>` comparison scan).
pub fn argmax(logits: &[f32]) -> usize {
    let mut best = 0usize;
    let mut bv = f32::NEG_INFINITY;
    for (i, &v) in logits.iter().enumerate() {
        if v > bv {
            bv = v;
            best = i;
        }
    }
    best
}

/// Softmax in place with temperature (numerically stable). `temperature <= 0`
/// collapses to a one-hot at the argmax.
pub fn softmax_temp(logits: &mut [f32], temperature: f32) {
    if temperature <= 0.0 {
        let a = argmax(logits);
        for (i, v) in logits.iter_mut().enumerate() {
            *v = if i == a { 1.0 } else { 0.0 };
        }
        return;
    }
    let inv_t = 1.0 / temperature;
    let max = logits.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
    let mut sum = 0.0f32;
    for v in logits.iter_mut() {
        *v = ((*v - max) * inv_t).exp();
        sum += *v;
    }
    if sum > 0.0 {
        for v in logits.iter_mut() {
            *v /= sum;
        }
    }
}

/// Nucleus (top-p) sample from raw logits.
///
/// `uniform` is a draw in `[0, 1)`. Applies temperature, sorts by probability,
/// keeps the smallest prefix whose cumulative mass ≥ `top_p`, renormalizes, and
/// selects by inverse-CDF. Returns a token index.
pub fn sample_top_p(logits: &[f32], cfg: SampleConfig, uniform: f32) -> usize {
    if cfg.temperature <= 0.0 {
        return argmax(logits);
    }
    let mut probs: Vec<f32> = logits.to_vec();
    softmax_temp(&mut probs, cfg.temperature);

    // indices sorted by descending probability
    let mut order: Vec<usize> = (0..probs.len()).collect();
    order.sort_by(|&a, &b| probs[b].partial_cmp(&probs[a]).unwrap_or(std::cmp::Ordering::Equal));

    // smallest nucleus with cumulative mass >= top_p
    let mut cum = 0.0f32;
    let mut cutoff = order.len();
    for (rank, &idx) in order.iter().enumerate() {
        cum += probs[idx];
        if cum >= cfg.top_p {
            cutoff = rank + 1;
            break;
        }
    }
    let nucleus = &order[..cutoff];
    let mass: f32 = nucleus.iter().map(|&i| probs[i]).sum();

    // inverse-CDF selection within the (renormalized) nucleus
    let target = uniform.clamp(0.0, 1.0 - f32::EPSILON) * mass;
    let mut acc = 0.0f32;
    for &idx in nucleus {
        acc += probs[idx];
        if acc > target {
            return idx;
        }
    }
    *nucleus.last().unwrap_or(&argmax(logits))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn argmax_picks_max_lowest_index_on_tie() {
        assert_eq!(argmax(&[0.1, 0.9, 0.3]), 1);
        assert_eq!(argmax(&[1.0, 1.0, 0.5]), 0); // tie -> lowest index
    }

    #[test]
    fn softmax_sums_to_one() {
        let mut l = [1.0, 2.0, 3.0];
        softmax_temp(&mut l, 1.0);
        let s: f32 = l.iter().sum();
        assert!((s - 1.0).abs() < 1e-5);
        assert!(l[2] > l[1] && l[1] > l[0]);
    }

    #[test]
    fn zero_temperature_is_greedy() {
        let cfg = SampleConfig {
            temperature: 0.0,
            top_p: 0.9,
        };
        assert_eq!(sample_top_p(&[0.2, 5.0, 1.0], cfg, 0.999), 1);
    }

    #[test]
    fn top_p_restricts_to_nucleus() {
        // One dominant logit: nucleus is just that token, any uniform picks it.
        let cfg = SampleConfig {
            temperature: 1.0,
            top_p: 0.9,
        };
        let logits = [10.0, 0.0, 0.0, 0.0];
        assert_eq!(sample_top_p(&logits, cfg, 0.0), 0);
        assert_eq!(sample_top_p(&logits, cfg, 0.5), 0);
        assert_eq!(sample_top_p(&logits, cfg, 0.999), 0);
    }

    #[test]
    fn top_p_inverse_cdf_boundaries() {
        // Two equal tokens, top_p = 1.0 -> both in nucleus, each 0.5 mass.
        let cfg = SampleConfig {
            temperature: 1.0,
            top_p: 1.0,
        };
        let logits = [0.0f32, 0.0];
        assert_eq!(sample_top_p(&logits, cfg, 0.0), 0); // first half
        assert_eq!(sample_top_p(&logits, cfg, 0.75), 1); // second half
    }
}
