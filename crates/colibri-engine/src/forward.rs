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
use crate::model::{KvCache, Layer, Model};
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

/// Tokens to speculatively draft per forward via the MTP head: `DRAFT=n`.
///
/// **Defaults to 0 (off)** — same as the C's `g_draft`, where speculation is
/// opt-in because the win is workload- and acceptance-dependent. Capped at 63
/// (the C's `draft[64]`).
pub(crate) fn draft_budget() -> usize {
    static N: OnceLock<usize> = OnceLock::new();
    *N.get_or_init(|| {
        std::env::var("DRAFT").ok().and_then(|v| v.parse::<usize>().ok()).unwrap_or(0).min(63)
    })
}

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

/// Run ONE layer over `x[S * hidden]` in place (positions
/// `pos_base..pos_base+S`), updating `kv[li]`. Port of `layer_forward`.
///
/// `nrm`/`tmp` are caller-owned scratch, each `[S * hidden]`, so a hot loop can
/// reuse them. Shared by the main stack and by the MTP head, which runs its own
/// block at `li = n_layers`.
pub fn layer_forward<P: ExpertProvider>(
    model: &Model,
    kv: &mut KvCache,
    provider: &P,
    l: &Layer,
    li: usize,
    x: &mut [f32],
    s: usize,
    pos_base: usize,
    nrm: &mut [f32],
    tmp: &mut [f32],
) -> io::Result<()> {
    let cfg = &model.cfg;
    let d = cfg.hidden as usize;
    // in_ln -> attention -> residual
    for si in 0..s {
        rmsnorm(&mut nrm[si * d..(si + 1) * d], &x[si * d..(si + 1) * d], &l.in_ln, cfg.eps);
    }
    timed(&ATTN_US, || attention(cfg, l, li, kv, nrm, s, pos_base, tmp));
    for j in 0..s * d {
        x[j] += tmp[j];
    }
    // post_ln -> MoE/dense -> residual
    for si in 0..s {
        rmsnorm(&mut nrm[si * d..(si + 1) * d], &x[si * d..(si + 1) * d], &l.post_ln, cfg.eps);
    }
    if l.sparse {
        timed(&MOE_US, || moe(cfg, l, li, nrm, s, tmp, true, provider))?;
    } else {
        timed(&DENSE_US, || dense_mlp(l, nrm, s, tmp));
    }
    for j in 0..s * d {
        x[j] += tmp[j];
    }
    Ok(())
}

/// Run the transformer stack over `ids` (positions `pos_base..pos_base+S`),
/// updating `kv` and writing the final hidden states `[S * hidden]` to
/// `hidden_out`. Port of embed + `layers_forward`.
///
/// `hidden_out` is the **raw** hidden state — before `model.norm`. That is what
/// [`logits`] and the MTP head both expect as input (each applies `final_norm`
/// itself).
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
    for li in 0..model.layers.len() {
        layer_forward(
            model,
            kv,
            provider,
            &model.layers[li],
            li,
            &mut x,
            s,
            pos_base,
            &mut nrm,
            &mut tmp,
        )?;
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
    let mut out = prompt.to_vec();
    generate_stream(model, kv, provider, prompt, n_new, |tok| {
        out.push(tok);
        true
    })?;
    Ok(out)
}

/// Streaming greedy generation: like [`generate_greedy`], but invokes `on_token`
/// with each newly decoded token id as it is produced (before the next forward
/// step), so a caller can stream output live. Returning `false` from `on_token`
/// stops generation early — used by the server to abort when a client
/// disconnects. A config stop token is delivered to `on_token` and then ends the
/// run. `generate_greedy` is a thin wrapper that collects the tokens.
pub fn generate_stream<P, F>(
    model: &Model,
    kv: &mut KvCache,
    provider: &P,
    prompt: &[i32],
    n_new: usize,
    on_token: F,
) -> io::Result<()>
where
    P: ExpertProvider,
    F: FnMut(i32) -> bool,
{
    let budget = if model.has_mtp { draft_budget() } else { 0 };
    generate_stream_drafting(model, kv, provider, prompt, n_new, budget, on_token)?;
    Ok(())
}

/// What a decode run did. `forwards < emitted` is exactly the speculation win;
/// `drafts_accepted / drafts_proposed` is the acceptance rate that decides
/// whether the head is earning its keep.
#[derive(Debug, Default, Clone, Copy, PartialEq, Eq)]
pub struct DecodeStats {
    pub emitted: usize,
    /// main-model forwards actually run
    pub forwards: u64,
    pub drafts_proposed: u64,
    pub drafts_accepted: u64,
}

/// [`generate_stream`] with an explicit speculation budget: `budget` tokens are
/// drafted by the MTP head per forward and verified against the main model.
/// `generate_stream` supplies `DRAFT` for it.
///
/// `budget == 0` disables speculation, and the loop then reduces exactly to the
/// plain one-token-per-forward path — which is the property the
/// "speculation does not change output" test relies on. Exposed separately
/// because `DRAFT` is read once per process and so cannot be varied in-process.
pub fn generate_stream_drafting<P, F>(
    model: &Model,
    kv: &mut KvCache,
    provider: &P,
    prompt: &[i32],
    n_new: usize,
    budget: usize,
    mut on_token: F,
) -> io::Result<DecodeStats>
where
    P: ExpertProvider,
    F: FnMut(i32) -> bool,
{
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
    let mut logits_us = 0u64;
    let mut logit = {
        let t = std::time::Instant::now();
        let l = logits(model, &hidden[(s - 1) * d..s * d]);
        logits_us += t.elapsed().as_micros() as u64;
        l
    };
    let mut pos = s;
    // Raw hidden at the position that produced `logit` — the MTP head's input.
    let mut hlast = hidden[(s - 1) * d..s * d].to_vec();

    // Speculation budget for this run. Starts at `DRAFT` (0 = off, matching the
    // C's `g_draft`) and is zeroed by the auto-off guard below. With `g == 0`
    // every step below degenerates to exactly the non-speculative loop, which is
    // what makes `DRAFT=0` byte-identical to `DRAFT=n`.
    let mut budget = if model.has_mtp { budget } else { 0 };
    let (mut proposed, mut accepted, mut forwards) = (0u64, 0u64, 0u64);

    let mut decode_ms: Vec<f64> = Vec::with_capacity(n_new);
    let mut emitted = 0usize;
    while emitted < n_new {
        let next = argmax(&logit) as i32;
        let keep_going = on_token(next);
        emitted += 1;
        if model.cfg.stop_ids.contains(&next) {
            break;
        }
        if !keep_going || emitted >= n_new {
            break;
        }

        // --- draft ---------------------------------------------------------
        // Auto-off: drafts that are never accepted are pure overhead (on this
        // engine, extra expert streaming). The C disables them below 10%
        // acceptance after 24 proposals; an int4 MTP head lands there.
        if budget > 0 && proposed >= 24 && accepted * 10 < proposed {
            eprintln!(
                "[MTP] {:.0}% acceptance after {proposed} proposals: drafts disabled",
                100.0 * accepted as f64 / proposed as f64
            );
            budget = 0;
        }
        let drafts = if budget > 0 {
            let dr = crate::mtp::draft(model, kv, provider, next, pos, budget, &hlast)?;
            proposed += dr.len() as u64;
            dr
        } else {
            Vec::new()
        };
        // Clamp to what we still owe the caller and to the cache.
        let mut g = drafts.len().min(n_new - emitted);
        if pos + g + 2 > kv.max_t {
            g = (kv.max_t.saturating_sub(pos + 2)).min(g);
        }

        // --- verify --------------------------------------------------------
        // One forward over [next, drafts...]: position i's logits reveal the
        // TRUE token at i+1, which is what each draft is checked against.
        let mut batch = Vec::with_capacity(1 + g);
        batch.push(next);
        batch.extend_from_slice(&drafts[..g]);
        let sb = batch.len();
        let mut h_all = vec![0f32; sb * d];
        let t = std::time::Instant::now();
        forward(model, kv, provider, &batch, pos, &mut h_all)?;
        forwards += 1;
        let ms = t.elapsed().as_secs_f64() * 1e3;
        if timing {
            eprintln!("[timing] decode tok {}: {ms:.1} ms ({:.2} tok/s)", pos - s, 1e3 / ms);
        }
        decode_ms.push(ms);

        let tl = std::time::Instant::now();
        let los: Vec<Vec<f32>> =
            (0..sb).map(|i| logits(model, &h_all[i * d..(i + 1) * d])).collect();
        logits_us += tl.elapsed().as_micros() as u64;

        // Accept the longest prefix that matches what the model itself would
        // have produced — this is why speculation cannot change the output.
        let mut k = 0usize;
        let mut done = false;
        while k < g && emitted < n_new {
            if argmax(&los[k]) as i32 != drafts[k] {
                break; // rejected: everything after it is stale too
            }
            let keep = on_token(drafts[k]);
            emitted += 1;
            k += 1;
            if model.cfg.stop_ids.contains(&drafts[k - 1]) || !keep {
                done = true;
                break;
            }
        }
        accepted += k as u64;

        // Keep the head's KV in sync with the VERIFIED tokens only.
        if k >= 1 {
            crate::mtp::absorb(model, kv, provider, &drafts[..k], &h_all, pos)?;
        }
        // `hlast` must be the last ACCEPTED position, not the end of the batch:
        // the KV past `pos + k` is stale and will simply be overwritten.
        hlast.copy_from_slice(&h_all[k * d..(k + 1) * d]);
        logit = los[k].clone();
        pos += 1 + k;
        if done {
            break;
        }
    }
    if budget > 0 && proposed > 0 {
        eprintln!(
            "[MTP] {accepted}/{proposed} drafts accepted ({:.0}%), {:.2} tok/forward",
            100.0 * accepted as f64 / proposed as f64,
            if forwards > 0 { emitted as f64 / forwards as f64 } else { 0.0 }
        );
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
    Ok(DecodeStats {
        emitted,
        forwards,
        drafts_proposed: proposed,
        drafts_accepted: accepted,
    })
}
