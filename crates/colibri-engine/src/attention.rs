//! MLA attention with a compressed KV-cache — port of `attention_rows` from
//! `c/glm.c` (the CPU path).
//!
//! GLM-5.2 uses Multi-head Latent Attention: per token only a normalized latent
//! `[kv_lora]` and a shared rotary key `[qk_rope]` are cached; the per-head
//! `k_nope` and `value` are recovered from the latent through `kv_b`. Two cores
//! compute the same result:
//!
//!   - [`AttnCore::Reconstruct`] — rebuild `k_nope`/`value` for every cached token
//!     via one `kv_b` matmul, then do standard causal attention. The reference.
//!   - [`AttnCore::Absorb`] — DeepSeek weight absorption: fold `W_K` into the
//!     query and apply `W_V` after the softmax, so nothing per-token is
//!     reconstructed (`O(T·kv_lora)` instead of `O(T·H·(nope+vh))`). The decode
//!     fast path.
//!
//! They are algebraically identical; a test asserts they agree numerically.
//! DSA sparse-indexer selection (long-context top-k) is not yet ported — this is
//! the dense path, exact for context ≤ `index_topk`.

use crate::linear::{matmul_qt, qt_addrow, qt_matvec_rows};
use crate::math::{rmsnorm_inplace, rope_interleave, softmax};
use crate::model::{KvCache, Layer};
use colibri_core::Config;

/// Which attention core to use. Both give the same result.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum AttnCore {
    /// Reconstruct k_nope/value from the latent (the reference path).
    Reconstruct,
    /// DeepSeek weight absorption (the decode fast path).
    Absorb,
}

/// MLA attention over `S` new tokens `x[S, hidden]` beginning at `pos_base`,
/// writing `out[S, hidden]`. Uses the reconstruction core.
pub fn attention(
    cfg: &Config,
    l: &Layer,
    layer: usize,
    kv: &mut KvCache,
    x: &[f32],
    s_len: usize,
    pos_base: usize,
    out: &mut [f32],
) {
    attention_with(cfg, l, layer, kv, x, s_len, pos_base, out, AttnCore::Reconstruct);
}

/// As [`attention`], but selecting the core explicitly.
#[allow(clippy::too_many_arguments)]
pub fn attention_with(
    cfg: &Config,
    l: &Layer,
    layer: usize,
    kv: &mut KvCache,
    x: &[f32],
    s_len: usize,
    pos_base: usize,
    out: &mut [f32],
    core: AttnCore,
) {
    let h = cfg.n_heads as usize;
    let qh = cfg.qk_head as usize;
    let qk_nope = cfg.qk_nope as usize;
    let r = cfg.qk_rope as usize;
    let vh = cfg.v_head as usize;
    let kvl = cfg.kv_lora as usize;
    let ql = cfg.q_lora as usize;
    let cw = kvl + r; // comp width
    let eps = cfg.eps;
    let theta = cfg.theta;

    // ---- 1) projections (batched over all S rows; exact kernel) ------------
    let mut qr = vec![0f32; s_len * ql];
    matmul_qt(&mut qr, x, &l.q_a, s_len);
    for s in 0..s_len {
        rmsnorm_inplace(&mut qr[s * ql..(s + 1) * ql], &l.q_a_ln, eps);
    }
    let mut q = vec![0f32; s_len * h * qh];
    matmul_qt(&mut q, &qr, &l.q_b, s_len);
    let mut comp = vec![0f32; s_len * cw];
    matmul_qt(&mut comp, x, &l.kv_a, s_len);

    // ---- 2) RoPE the query rope halves; write the compressed cache ---------
    for s in 0..s_len {
        let pos = pos_base + s;
        for hh in 0..h {
            let base = s * h * qh + hh * qh + qk_nope;
            rope_interleave(&mut q[base..base + r], pos, r, theta);
        }
        // normalized latent
        let latent_src_end = s * cw + kvl;
        {
            let ldst = kv.latent_row_mut(layer, pos);
            ldst.copy_from_slice(&comp[s * cw..latent_src_end]);
            rmsnorm_inplace(ldst, &l.kv_a_ln, eps);
        }
        // roped k_rot (shared across heads)
        {
            let rdst = kv.krot_row_mut(layer, pos);
            rdst.copy_from_slice(&comp[latent_src_end..latent_src_end + r]);
            rope_interleave(rdst, pos, r, theta);
        }
    }

    let st0 = kv.kv_start[layer];
    let mut ctx = vec![0f32; s_len * h * vh];

    // GPU weight-absorption attention core for resident kv_b (falls back to CPU).
    #[cfg(feature = "cuda")]
    let ran_gpu = {
        let tk = pos_base + s_len;
        crate::gpu::try_attention_absorb(
            &l.kv_b,
            &mut ctx,
            &q,
            kv.latent_rows(layer, st0, tk),
            kv.krot_rows(layer, st0, tk),
            s_len,
            h,
            qk_nope,
            r,
            vh,
            kvl,
            tk - st0,
            cfg.attn_scale,
        )
    };
    #[cfg(not(feature = "cuda"))]
    let ran_gpu = false;

    if !ran_gpu {
        match core {
            AttnCore::Reconstruct => {
                reconstruct_core(cfg, l, layer, kv, &q, s_len, pos_base, st0, &mut ctx);
            }
            AttnCore::Absorb => {
                absorb_core(cfg, l, layer, kv, &q, s_len, pos_base, st0, &mut ctx);
            }
        }
    }

    // ---- 4) output projection ----------------------------------------------
    matmul_qt(out, &ctx, &l.o, s_len);
}

/// Reconstruction core: rebuild k_nope/value for all cached tokens via one kv_b
/// matmul, then causal attention. Port of `attention_rows` step 2/3.
#[allow(clippy::too_many_arguments)]
fn reconstruct_core(
    cfg: &Config,
    l: &Layer,
    layer: usize,
    kv: &KvCache,
    q: &[f32],
    s_len: usize,
    pos_base: usize,
    st0: usize,
    ctx: &mut [f32],
) {
    let h = cfg.n_heads as usize;
    let qh = cfg.qk_head as usize;
    let qk_nope = cfg.qk_nope as usize;
    let r = cfg.qk_rope as usize;
    let vh = cfg.v_head as usize;
    let scale = cfg.attn_scale;
    let head_kv = qk_nope + vh;
    let kvb_dim = h * head_kv;
    let tk = pos_base + s_len;
    let nrec = tk - st0;

    // kvb_all[t-st0] = kv_b(latent_t) -> [nrec, kvb_dim]
    let mut kvb_all = vec![0f32; nrec * kvb_dim];
    matmul_qt(&mut kvb_all, kv.latent_rows(layer, st0, tk), &l.kv_b, nrec);

    for s in 0..s_len {
        let pos = pos_base + s;
        for hh in 0..h {
            let qbase = s * h * qh + hh * qh;
            let (qnope, qrope) = q[qbase..qbase + qh].split_at(qk_nope);
            let nt = pos + 1 - st0;
            let mut sc = vec![0f32; nt];
            for (jj, sc_jj) in sc.iter_mut().enumerate() {
                let t = st0 + jj;
                let kn_off = (t - st0) * kvb_dim + hh * head_kv;
                let kn = &kvb_all[kn_off..kn_off + qk_nope];
                let kr = kv.krot_row(layer, t);
                let mut a = 0f32;
                for i in 0..qk_nope {
                    a += qnope[i] * kn[i];
                }
                for d in 0..r {
                    a += qrope[d] * kr[d];
                }
                *sc_jj = a * scale;
            }
            softmax(&mut sc);
            let cx = &mut ctx[(s * h + hh) * vh..(s * h + hh) * vh + vh];
            for (jj, &a) in sc.iter().enumerate() {
                let t = st0 + jj;
                let vv_off = (t - st0) * kvb_dim + hh * head_kv + qk_nope;
                let vv = &kvb_all[vv_off..vv_off + vh];
                for d in 0..vh {
                    cx[d] += a * vv[d];
                }
            }
        }
    }
}

/// Weight-absorption core: fold W_K into the query, apply W_V after softmax.
/// Port of `attention_rows`'s absorb branch.
#[allow(clippy::too_many_arguments)]
fn absorb_core(
    cfg: &Config,
    l: &Layer,
    layer: usize,
    kv: &KvCache,
    q: &[f32],
    s_len: usize,
    pos_base: usize,
    st0: usize,
    ctx: &mut [f32],
) {
    let h = cfg.n_heads as usize;
    let qh = cfg.qk_head as usize;
    let qk_nope = cfg.qk_nope as usize;
    let r = cfg.qk_rope as usize;
    let vh = cfg.v_head as usize;
    let kvl = cfg.kv_lora as usize;
    let scale = cfg.attn_scale;
    let head_kv = qk_nope + vh;

    for s in 0..s_len {
        let pos = pos_base + s;
        for hh in 0..h {
            let qbase = s * h * qh + hh * qh;
            let (qnope, qrope) = q[qbase..qbase + qh].split_at(qk_nope);
            let rbase = hh * head_kv;
            // qabs = W_K^h^T q_nope  (a [kv_lora] vector)
            let mut qabs = vec![0f32; kvl];
            for (d, &qn) in qnope.iter().enumerate() {
                qt_addrow(&l.kv_b, rbase + d, qn, &mut qabs);
            }
            let nt = pos + 1 - st0;
            let mut sc = vec![0f32; nt];
            for (jj, sc_jj) in sc.iter_mut().enumerate() {
                let t = st0 + jj;
                let lt = kv.latent_row(layer, t);
                let kr = kv.krot_row(layer, t);
                let mut a = 0f32;
                for i in 0..kvl {
                    a += qabs[i] * lt[i];
                }
                for d in 0..r {
                    a += qrope[d] * kr[d];
                }
                *sc_jj = a * scale;
            }
            softmax(&mut sc);
            // clat = Σ_t a_t L_t
            let mut clat = vec![0f32; kvl];
            for (jj, &a) in sc.iter().enumerate() {
                let lt = kv.latent_row(layer, st0 + jj);
                for i in 0..kvl {
                    clat[i] += a * lt[i];
                }
            }
            // ctx^h = W_V^h clat
            let cx = &mut ctx[(s * h + hh) * vh..(s * h + hh) * vh + vh];
            qt_matvec_rows(&l.kv_b, rbase + qk_nope, vh, &clat, cx);
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::quantize::qtensor_from_f32;
    use colibri_core::QTensor;

    // Deterministic small "random" weights.
    fn weights(o: usize, i: usize, seed: usize) -> QTensor {
        let w: Vec<f32> = (0..o * i)
            .map(|k| (((k * 7 + seed * 13 + 3) % 11) as f32 - 5.0) * 0.1)
            .collect();
        qtensor_from_f32(&w, o, i, 16) // f32 (exact)
    }

    fn vecf(n: usize, seed: usize) -> Vec<f32> {
        (0..n)
            .map(|k| (((k * 5 + seed * 3 + 1) % 7) as f32 - 3.0) * 0.1 + 1.0)
            .collect()
    }

    // A tiny but structurally faithful attention config.
    fn cfg() -> Config {
        let json = colibri_json::Json::parse(
            r#"{"hidden_size":6,"num_hidden_layers":1,"num_attention_heads":2,
                "n_routed_experts":4,"num_experts_per_tok":2,"moe_intermediate_size":4,
                "intermediate_size":6,"first_k_dense_replace":0,"q_lora_rank":4,
                "kv_lora_rank":4,"qk_nope_head_dim":3,"qk_rope_head_dim":2,"v_head_dim":3,
                "n_shared_experts":1,"vocab_size":10,"n_group":1,"topk_group":1,
                "rms_norm_eps":1e-5,"routed_scaling_factor":1.0,
                "rope_parameters":{"rope_theta":10000.0},"eos_token_id":[9],
                "index_topk":0,"index_n_heads":0,"index_head_dim":0}"#,
        )
        .unwrap();
        Config::from_json(&json).unwrap()
    }

    fn make_layer(c: &Config) -> Layer {
        let h = c.n_heads as usize;
        let d = c.hidden as usize;
        let mut l = Layer::default();
        l.q_a = weights(c.q_lora as usize, d, 1);
        l.q_a_ln = vec![1.0; c.q_lora as usize];
        l.q_b = weights(h * c.qk_head as usize, c.q_lora as usize, 2);
        l.kv_a = weights((c.kv_lora + c.qk_rope) as usize, d, 3);
        l.kv_a_ln = vec![1.0; c.kv_lora as usize];
        l.kv_b = weights(h * (c.qk_nope + c.v_head) as usize, c.kv_lora as usize, 4);
        l.o = weights(d, h * c.v_head as usize, 5);
        l
    }

    #[test]
    fn reconstruct_and_absorb_agree() {
        let c = cfg();
        let l = make_layer(&c);
        let d = c.hidden as usize;
        let s_len = 3;
        let x = vecf(s_len * d, 9);

        let mut kv_a = KvCache::new(1, c.kv_lora as usize, c.qk_rope as usize, 16);
        let mut kv_b = KvCache::new(1, c.kv_lora as usize, c.qk_rope as usize, 16);
        let mut out_recon = vec![0f32; s_len * d];
        let mut out_absorb = vec![0f32; s_len * d];

        attention_with(&c, &l, 0, &mut kv_a, &x, s_len, 0, &mut out_recon, AttnCore::Reconstruct);
        attention_with(&c, &l, 0, &mut kv_b, &x, s_len, 0, &mut out_absorb, AttnCore::Absorb);

        for (a, b) in out_recon.iter().zip(&out_absorb) {
            assert!((a - b).abs() < 1e-4, "reconstruct {a} vs absorb {b}");
        }
        // sanity: not all zero
        assert!(out_recon.iter().any(|v| v.abs() > 1e-6));
    }

    #[test]
    fn single_token_is_value_of_itself() {
        // With one token, softmax over one score is 1.0, so ctx = W_V(L_0) and
        // both cores must produce identical output (no averaging).
        let c = cfg();
        let l = make_layer(&c);
        let d = c.hidden as usize;
        let x = vecf(d, 4);

        let mut kv1 = KvCache::new(1, c.kv_lora as usize, c.qk_rope as usize, 4);
        let mut kv2 = KvCache::new(1, c.kv_lora as usize, c.qk_rope as usize, 4);
        let mut o1 = vec![0f32; d];
        let mut o2 = vec![0f32; d];
        attention_with(&c, &l, 0, &mut kv1, &x, 1, 0, &mut o1, AttnCore::Reconstruct);
        attention_with(&c, &l, 0, &mut kv2, &x, 1, 0, &mut o2, AttnCore::Absorb);
        for (a, b) in o1.iter().zip(&o2) {
            assert!((a - b).abs() < 1e-4);
        }
    }

    #[test]
    fn incremental_decode_matches_batched_prefill() {
        // Feeding 3 tokens at once (prefill) must match feeding them one at a
        // time (decode) — the KV-cache carries the context across calls.
        let c = cfg();
        let l = make_layer(&c);
        let d = c.hidden as usize;
        let x = vecf(3 * d, 9);

        let mut kv_batch = KvCache::new(1, c.kv_lora as usize, c.qk_rope as usize, 16);
        let mut out_batch = vec![0f32; 3 * d];
        attention(&c, &l, 0, &mut kv_batch, &x, 3, 0, &mut out_batch);

        let mut kv_step = KvCache::new(1, c.kv_lora as usize, c.qk_rope as usize, 16);
        let mut out_step = vec![0f32; 3 * d];
        for s in 0..3 {
            let mut o = vec![0f32; d];
            attention(&c, &l, 0, &mut kv_step, &x[s * d..(s + 1) * d], 1, s, &mut o);
            out_step[s * d..(s + 1) * d].copy_from_slice(&o);
        }
        for (a, b) in out_batch.iter().zip(&out_step) {
            assert!((a - b).abs() < 1e-4, "prefill {a} vs decode {b}");
        }
    }
}
