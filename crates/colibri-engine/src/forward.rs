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
    for (i, &tok) in ids.iter().enumerate() {
        embed_row(&model.embed, tok as usize, &mut x[i * d..(i + 1) * d]);
    }

    let mut nrm = vec![0f32; s * d];
    let mut tmp = vec![0f32; s * d];
    for (li, l) in model.layers.iter().enumerate() {
        // in_ln -> attention -> residual
        for si in 0..s {
            rmsnorm(&mut nrm[si * d..(si + 1) * d], &x[si * d..(si + 1) * d], &l.in_ln, cfg.eps);
        }
        attention(cfg, l, li, kv, &nrm, s, pos_base, &mut tmp);
        for j in 0..s * d {
            x[j] += tmp[j];
        }
        // post_ln -> MoE/dense -> residual
        for si in 0..s {
            rmsnorm(&mut nrm[si * d..(si + 1) * d], &x[si * d..(si + 1) * d], &l.post_ln, cfg.eps);
        }
        if l.sparse {
            moe(cfg, l, li, &nrm, s, &mut tmp, true, provider)?;
        } else {
            dense_mlp(l, &nrm, s, &mut tmp);
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

    // prefill
    let s = prompt.len();
    let mut hidden = vec![0f32; s * d];
    forward(model, kv, provider, prompt, 0, &mut hidden)?;
    let mut out = prompt.to_vec();
    let mut logit = logits(model, &hidden[(s - 1) * d..s * d]);
    let mut pos = s;

    for _ in 0..n_new {
        let next = argmax(&logit) as i32;
        out.push(next);
        if model.cfg.stop_ids.contains(&next) {
            break;
        }
        let mut h = vec![0f32; d];
        forward(model, kv, provider, &[next], pos, &mut h)?;
        logit = logits(model, &h);
        pos += 1;
    }
    Ok(out)
}
