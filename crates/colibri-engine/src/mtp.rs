//! The MTP speculative head's forward paths — port of `mtp_absorb` and
//! `mtp_draft` from `c/glm.c`.
//!
//! The head predicts token `t+2` from the main model's hidden state at `t` and
//! the embedding of `t+1`. Both entry points fuse those two inputs identically
//! (see [`fuse`]) and then run the head's own transformer block at layer index
//! `n_layers`:
//!
//! - [`absorb`] runs it over already-**verified** tokens purely to keep the
//!   head's KV in sync with the main model's; its output is discarded.
//! - [`draft`] chains it to *propose* tokens, feeding each prediction back in.
//!
//! # The two orientations that silently kill acceptance
//!
//! Drafts are only ever *accepted* when they match what the main model would
//! have produced, so a mistake here **cannot corrupt output** — it just makes
//! every draft wrong, and speculation degrades to a slow no-op. That makes the
//! following two details worth stating loudly (the C carries `MTP_PRENORM` /
//! `MTP_SWAP` / `MTP_DEBUG` toggles precisely because they were fought over):
//!
//! 1. **`h` is POST `model.norm`.** The hidden coming out of the main stack is
//!    raw, so `final_norm` is applied *before* `hnorm`. But only for a hidden
//!    that came from the main stack — once [`draft`] chains, `h` is the head
//!    block's own output and must NOT be re-`final_norm`'d.
//! 2. **The concatenation is `[e ; h]`**, embedding first.

use crate::forward::layer_forward;
use crate::linear::{embed_row, matmul_qt};
use crate::math::rmsnorm;
use crate::model::{KvCache, Model, MtpHead};
use crate::moe::ExpertProvider;
use crate::sampling::argmax;
use std::io;

/// Fuse `next_tok`'s embedding with hidden state `h` into the head's block input:
///
/// ```text
/// e  = rmsnorm(embed(next_tok), enorm)
/// h' = rmsnorm(h, hnorm)                      // h_is_raw == false
///    = rmsnorm(rmsnorm(h, final_norm), hnorm) // h_is_raw == true
/// out = eh_proj · [e ; h']
/// ```
///
/// `h_is_raw` says whether `h` came straight from the main stack (pre-`model.norm`).
fn fuse(model: &Model, mtp: &MtpHead, next_tok: i32, h: &[f32], h_is_raw: bool, out: &mut [f32]) {
    let d = model.cfg.hidden as usize;
    let eps = model.cfg.eps;

    let mut cat = vec![0f32; 2 * d];
    // e = rmsnorm(embed(next), enorm)
    let mut e = vec![0f32; d];
    embed_row(&model.embed, next_tok as usize, &mut e);
    rmsnorm(&mut cat[..d], &e, &mtp.enorm, eps);

    // h -> [final_norm if raw] -> hnorm
    if h_is_raw {
        let mut hf = vec![0f32; d];
        rmsnorm(&mut hf, h, &model.final_norm, eps); // vLLM: h POST model.norm
        rmsnorm(&mut cat[d..], &hf, &mtp.hnorm, eps);
    } else {
        rmsnorm(&mut cat[d..], h, &mtp.hnorm, eps);
    }

    // cat is [e ; h'] — embedding first (C default; MTP_SWAP flips it).
    matmul_qt(out, &cat, &mtp.eh_proj, 1);
}

/// Run the head over tokens the main model has already **verified**, so its KV
/// covers the same positions. Port of `mtp_absorb`.
///
/// `hidden[i]` is the main model's raw hidden at position `pos_base + i`, and
/// `next_ids[i]` is the token that followed it. The block's output is discarded:
/// this call exists only for its KV side effect.
///
/// No-op when the model has no MTP head.
pub fn absorb<P: ExpertProvider>(
    model: &Model,
    kv: &mut KvCache,
    provider: &P,
    next_ids: &[i32],
    hidden: &[f32],
    pos_base: usize,
) -> io::Result<()> {
    let mtp = match &model.mtp {
        Some(m) => m,
        None => return Ok(()),
    };
    let s = next_ids.len();
    if s == 0 {
        return Ok(());
    }
    let d = model.cfg.hidden as usize;
    let li = model.cfg.n_layers as usize;
    debug_assert!(hidden.len() >= s * d);

    kv.start_at(li, pos_base);
    let mut hx = vec![0f32; s * d];
    for i in 0..s {
        fuse(
            model,
            mtp,
            next_ids[i],
            &hidden[i * d..(i + 1) * d],
            true,
            &mut hx[i * d..(i + 1) * d],
        );
    }
    let mut nrm = vec![0f32; s * d];
    let mut tmp = vec![0f32; s * d];
    layer_forward(model, kv, provider, &mtp.layer, li, &mut hx, s, pos_base, &mut nrm, &mut tmp)
}

/// Propose up to `g_max` draft tokens by chaining the head. Port of `mtp_draft`.
///
/// `next_tok` is the token just emitted (stored at index `kv_idx`), and
/// `last_hidden` is the main model's raw hidden at the position that produced it
/// — i.e. position `kv_idx - 1`, which is where the head's block runs. Each step
/// feeds its own prediction back in and advances one position.
///
/// Returns an empty vec when there is no head, nothing to draft, or no room in
/// the cache.
pub fn draft<P: ExpertProvider>(
    model: &Model,
    kv: &mut KvCache,
    provider: &P,
    next_tok: i32,
    kv_idx: usize,
    g_max: usize,
    last_hidden: &[f32],
) -> io::Result<Vec<i32>> {
    let mtp = match &model.mtp {
        Some(m) => m,
        None => return Ok(Vec::new()),
    };
    // C: `p = kv - 1; if (p < 0 || G < 1) return 0;`
    if kv_idx == 0 || g_max == 0 {
        return Ok(Vec::new());
    }
    let d = model.cfg.hidden as usize;
    let li = model.cfg.n_layers as usize;
    let p = kv_idx - 1;

    kv.start_at(li, p);
    let mut out = Vec::with_capacity(g_max);
    let mut h = last_hidden[..d].to_vec();
    let mut tok = next_tok;
    let mut nrm = vec![0f32; d];
    let mut tmp = vec![0f32; d];
    let mut hx = vec![0f32; d];
    let mut row = vec![0f32; d];

    for g in 0..g_max {
        let pos = p + g;
        if pos + 2 >= kv.max_t {
            break; // C: no room for this draft plus its verification
        }
        // Only the first step's hidden is raw (it came from the main stack);
        // afterwards `h` is this block's own output.
        fuse(model, mtp, tok, &h, g == 0, &mut hx);
        layer_forward(model, kv, provider, &mtp.layer, li, &mut hx, 1, pos, &mut nrm, &mut tmp)?;

        // the head's own final norm, then the SHARED lm_head
        rmsnorm(&mut row, &hx, &mtp.mtp_norm, model.cfg.eps);
        let mut lo = vec![0f32; model.cfg.vocab as usize];
        matmul_qt(&mut lo, &row, &model.lm_head, 1);

        let t2 = argmax(&lo) as i32;
        out.push(t2);
        tok = t2;
        h.copy_from_slice(&hx);
    }
    Ok(out)
}
