//! The assembled forward pass and greedy decode loop — port of `layer_forward`
//! / `layers_forward` / `forward_all` / `generate` in `c/glm.c`.
//!
//! Per layer (CPU path): `in_ln` RMSNorm → MLA attention → residual add →
//! `post_ln` RMSNorm → MoE (or dense MLP for the first `first_k_dense_replace`
//! layers) → residual add. Then a final RMSNorm and the `lm_head` produce
//! logits, and greedy decoding feeds the argmax back in one token at a time.

use crate::attention::attention;
use crate::linear::{embed_row, matmul_qt};
use crate::math::rmsnorm;
use crate::model::{KvCache, Model};
use crate::moe::{dense_mlp, moe, ExpertProvider};
use crate::sampling::argmax;
use std::io;
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::OnceLock;

/// COLI_PROFILE=1 accumulates per-section wall time (microseconds) across the
/// forward pass so `generate_greedy` can print a breakdown. Off by default.
pub(crate) fn profile_on() -> bool {
    static ON: OnceLock<bool> = OnceLock::new();
    *ON.get_or_init(|| std::env::var("COLI_PROFILE").ok().as_deref() == Some("1"))
}
static ATTN_US: AtomicU64 = AtomicU64::new(0);
static MOE_US: AtomicU64 = AtomicU64::new(0);
static DENSE_US: AtomicU64 = AtomicU64::new(0);
static EMBED_US: AtomicU64 = AtomicU64::new(0);
/// Time spent fetching experts through the provider (disk→RAM on a cache miss).
/// A sub-total of `MOE_US`. Incremented from `moe`.
pub(crate) static LOAD_US: AtomicU64 = AtomicU64::new(0);

/// Time `f` into `acc` when profiling is enabled (else just run it).
#[inline]
fn timed<T>(acc: &AtomicU64, f: impl FnOnce() -> T) -> T {
    if !profile_on() {
        return f();
    }
    let t = std::time::Instant::now();
    let r = f();
    acc.fetch_add(t.elapsed().as_micros() as u64, Ordering::Relaxed);
    r
}

/// Run the transformer stack over `ids` (positions `pos_base..pos_base+S`),
/// updating `kv` and writing the final hidden states `[S * hidden]` to
/// `hidden_out`. Port of embed + `layers_forward`.
pub fn forward<P: ExpertProvider>(
    model: &Model,
    kv: &mut KvCache,
    provider: &P,
    ids: &[i32],
    pos_base: usize,
    hidden_out: &mut [f32],
) -> io::Result<()> {
    let cfg = &model.cfg;
    let d = cfg.hidden as usize;
    let s = ids.len();
    assert_eq!(hidden_out.len(), s * d);

    // token embeddings
    let mut x = vec![0f32; s * d];
    timed(&EMBED_US, || {
        for (i, &tok) in ids.iter().enumerate() {
            embed_row(&model.embed, tok as usize, &mut x[i * d..(i + 1) * d]);
        }
    });

    let mut nrm = vec![0f32; s * d];
    let mut tmp = vec![0f32; s * d];
    for (li, l) in model.layers.iter().enumerate() {
        // in_ln -> attention -> residual
        for si in 0..s {
            rmsnorm(&mut nrm[si * d..(si + 1) * d], &x[si * d..(si + 1) * d], &l.in_ln, cfg.eps);
        }
        timed(&ATTN_US, || attention(cfg, l, li, kv, &nrm, s, pos_base, &mut tmp));
        for j in 0..s * d {
            x[j] += tmp[j];
        }
        // post_ln -> MoE/dense -> residual
        for si in 0..s {
            rmsnorm(&mut nrm[si * d..(si + 1) * d], &x[si * d..(si + 1) * d], &l.post_ln, cfg.eps);
        }
        if l.sparse {
            timed(&MOE_US, || moe(cfg, l, li, &nrm, s, &mut tmp, true, provider))?;
        } else {
            timed(&DENSE_US, || dense_mlp(l, &nrm, s, &mut tmp));
        }
        for j in 0..s * d {
            x[j] += tmp[j];
        }
    }

    hidden_out.copy_from_slice(&x);
    Ok(())
}

/// Logits for a single hidden-state row: final RMSNorm then `lm_head`. Port of
/// the tail of `forward_all`.
pub fn logits(model: &Model, hidden_row: &[f32]) -> Vec<f32> {
    let d = model.cfg.hidden as usize;
    let v = model.cfg.vocab as usize;
    let mut row = vec![0f32; d];
    rmsnorm(&mut row, hidden_row, &model.final_norm, model.cfg.eps);
    let mut lo = vec![0f32; v];
    matmul_qt(&mut lo, &row, &model.lm_head, 1);
    lo
}

/// Greedy generation: prefill the prompt, then decode up to `n_new` tokens by
/// argmax, feeding each back through the cache. Stops early on a config stop
/// token. Port of `generate` (greedy path, no speculation). Returns the full
/// sequence (prompt + continuation).
pub fn generate_greedy<P: ExpertProvider>(
    model: &Model,
    kv: &mut KvCache,
    provider: &P,
    prompt: &[i32],
    n_new: usize,
) -> io::Result<Vec<i32>> {
    let d = model.cfg.hidden as usize;
    assert!(!prompt.is_empty(), "prompt must be non-empty");
    assert!(
        kv.max_t >= prompt.len() + n_new,
        "Kv cache too small: max_t={} needs >= {}",
        kv.max_t,
        prompt.len() + n_new
    );

    // COLI_TIMING=1 prints per-step wall time (prefill + each decode token) to
    // stderr and a steady-state decode tok/s summary. Off by default so the
    // C-vs-Rust validation harness output stays clean.
    let timing = std::env::var("COLI_TIMING").ok().as_deref() == Some("1");

    // prefill
    let s = prompt.len();
    let mut hidden = vec![0f32; s * d];
    let t_pre = std::time::Instant::now();
    forward(model, kv, provider, prompt, 0, &mut hidden)?;
    if timing {
        let ms = t_pre.elapsed().as_secs_f64() * 1e3;
        eprintln!("[timing] prefill {s} tok: {ms:.1} ms ({:.1} tok/s)", s as f64 / (ms / 1e3));
    }
    let mut out = prompt.to_vec();
    let mut logits_us = 0u64;
    let mut logit = {
        let t = std::time::Instant::now();
        let l = logits(model, &hidden[(s - 1) * d..s * d]);
        logits_us += t.elapsed().as_micros() as u64;
        l
    };
    let mut pos = s;

    let mut decode_ms: Vec<f64> = Vec::with_capacity(n_new);
    for _ in 0..n_new {
        let next = argmax(&logit) as i32;
        out.push(next);
        if model.cfg.stop_ids.contains(&next) {
            break;
        }
        let mut h = vec![0f32; d];
        let t = std::time::Instant::now();
        forward(model, kv, provider, &[next], pos, &mut h)?;
        let ms = t.elapsed().as_secs_f64() * 1e3;
        if timing {
            eprintln!("[timing] decode tok {}: {ms:.1} ms ({:.2} tok/s)", pos - s, 1e3 / ms);
        }
        decode_ms.push(ms);
        let tl = std::time::Instant::now();
        logit = logits(model, &h);
        logits_us += tl.elapsed().as_micros() as u64;
        pos += 1;
    }
    if timing && !decode_ms.is_empty() {
        // Steady state: drop the first half (cold expert-cache misses) and
        // average the rest.
        let warm = &decode_ms[decode_ms.len() / 2..];
        let mean = warm.iter().sum::<f64>() / warm.len() as f64;
        let min = warm.iter().cloned().fold(f64::INFINITY, f64::min);
        eprintln!(
            "[timing] decode steady-state (last {} of {} tok): mean {mean:.1} ms ({:.2} tok/s), best {min:.1} ms ({:.2} tok/s)",
            warm.len(),
            decode_ms.len(),
            1e3 / mean,
            1e3 / min,
        );
    }
    if profile_on() {
        // Totals across prefill + all decode steps (microseconds -> ms).
        let ms = |a: &AtomicU64| a.load(Ordering::Relaxed) as f64 / 1e3;
        eprintln!(
            "[profile] totals: attn {:.0} ms | moe {:.0} ms (of which expert-load {:.0} ms) | dense {:.0} ms | embed {:.0} ms | logits {:.0} ms",
            ms(&ATTN_US),
            ms(&MOE_US),
            ms(&LOAD_US),
            ms(&DENSE_US),
            ms(&EMBED_US),
            logits_us as f64 / 1e3,
        );
    }
    Ok(out)
}
