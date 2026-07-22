//! FP8 → colibrì container conversion — Rust port of `c/tools/convert_fp8_to_int4.py`
//! (the default, non-`--mtp`/`--indexer` path).
//!
//! Reads a Hugging Face GLM-5.2 snapshot whose linear weights are **block-scaled
//! FP8** (`F8_E4M3` codes + a `name.weight_scale_inv` F32 grid of 128×128 block
//! scales) and rewrites it as colibrì's own pre-quantized container: resident
//! weights become a `name` `U8` tensor of packed int8 codes plus a `name.qs` `F32`
//! tensor of per-row scales — exactly what [`crate::loader::qt_load`] reads; routed
//! experts become NVFP4 (or e4m3 under `COLI_XFP8`). Norms, the MoE router, biases,
//! and embeddings-passthrough stay `F32`.
//!
//! The MTP speculative head (layer `n_layers`: `eh_proj`/`enorm`/`hnorm`/
//! `shared_head` plus its own attention + MoE block) is **kept by default**, so every
//! container ships MTP-ready and the engine auto-detects `has_mtp = true`. Drafting is
//! still opt-in at runtime (`DRAFT=n`; `MTP=0` forces the head off), so a container
//! with the head is byte-identical to one without it whenever drafting is disabled —
//! the head only costs one extra MoE layer on disk (~one layer's worth of experts).
//!
//! What is dropped by default (matching the reference): the DSA lightning indexer
//! (`self_attn.indexer.*`). The engine then auto-detects `has_dsa = false` from the
//! absent tensors, so attention is exact dense MLA.
//!
//! Set `ConvertOpts::keep_indexer` (`COLI_KEEP_INDEXER=1`) to retain the indexer
//! weights instead — the wk/wq_b/weights_proj matrices quantize at `ebits`, k_norm
//! stays f32 — so the resulting container has `has_dsa = true` and runs DSA sparse
//! attention above `index_topk` context.
//!
//! `COLI_MTP_ONLY=1` emits *only* the head (into `mtp-NNNNN.safetensors` shards), to
//! augment an already-converted head-less container without re-converting the model.
//!
//! The quantizer math is the shared, C-exact [`crate::quantize`] code, so a
//! converted weight is byte-identical to a runtime-quantized one.

use crate::quantize::{qtensor_from_f32, quantize_rows};
use colibri_core::dtype::DType;
use colibri_core::quant::QTensor;
use colibri_safetensors::Shards;
use std::io::{self, BufWriter, Write};
use std::path::Path;

/// FP8 block-scale group size (both dims), as in the checkpoint's
/// `weight_block_size: [128, 128]`.
const BLOCK: usize = 128;

/// NVFP4 block-scale group size (along the input dim), as in modelopt's
/// `group_size: 16`.
const NVFP4_GS: usize = 16;

/// FP4 `e2m1` codebook: 1 sign + 2 exponent + 1 mantissa → 16 non-uniform codes
/// (bit 3 = sign). Verified 1:1 against `ml_dtypes.float4_e2m1fn` by the reference.
const E2M1: [f32; 16] = [
    0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0, -0.0, -0.5, -1.0, -1.5, -2.0, -3.0, -4.0, -6.0,
];

/// Bit-widths for each resident tensor class (routed experts are NVFP4/e4m3, not
/// int-quantized — see `xfp8` / [`quantize_nvfp4_out`]).
///
/// **Resident weights default to 8-bit int8, measured.** The reference converter
/// defaults everything to int4; on GLM-5.2 that wrecks the model. Same source
/// (unsloth/GLM-5.2-FP8), same converter, only `ebits` changed:
///
/// | | perplexity | top-1 |
/// |---|---|---|
/// | `4` resident (reference default) | 48.665 | 32.1% |
/// | `8` resident (ours)              |  6.189 | 57.9% |
///
/// 7.9x the quality — perplexity 48.7 means the model was effectively guessing among
/// ~49 tokens; 6.2 is a healthy frontier-model number. The damage is in the *resident*
/// path, not the experts: attention + dense + shared expert are only 2.5% of the
/// parameters but 42% of what every token touches, and they cross all 78 layers.
#[derive(Debug, Clone, Copy)]
pub struct ConvertOpts {
    /// bits for resident weights (attention, dense MLP, shared expert) — `--ebits`
    pub ebits: u32,
    /// bits for embeddings + `lm_head` — `--io-bits`
    pub io_bits: u32,
    /// number of transformer layers; layer index `>= n_layers` is the MTP head
    pub n_layers: usize,
    /// keep the DSA lightning-indexer weights (`self_attn.indexer.*`) instead of
    /// dropping them — required to run DSA sparse attention. `--indexer`. The wk/wq_b/
    /// weights_proj matrices quantize at `ebits`; k_norm stays f32. Adds ~index-head
    /// weights per layer (small vs the experts).
    pub keep_indexer: bool,
    /// convert ONLY the MTP speculative head (layer index `n_layers`) — `COLI_MTP_ONLY`.
    /// Produces a small shard you drop into an existing container to enable drafting
    /// without re-converting the whole model. See `classify`.
    pub mtp_only: bool,
    /// emit routed experts as per-row-scaled e4m3 fp8 (1 byte/weight) — `COLI_XFP8=1` —
    /// the 8-bit opt-out from the NVFP4 default. 2× the streamed bytes of NVFP4; consumed
    /// by the tiled FP8 expert kernel. Experts are **NVFP4 by default** (see [`quantize_nvfp4`]);
    /// int4 experts are no longer produced.
    pub xfp8: bool,
    /// source is MiniMax-M3 (`minimax_m3_vl`): map its `language_model.*` /
    /// `block_sparse_moe.*` tensor names to the container's GLM-style names, drop the
    /// vision tower and the MTP module (deferred), and fold Gemma-norm `+1`.
    pub minimax: bool,
    /// fold `+1` into RMSNorm weights (MiniMax-M3 Gemma-norm) so the engine's plain
    /// `rmsnorm` computes `x·(1+w)`; requires `minimax`.
    pub gemma_norm: bool,
}

impl Default for ConvertOpts {
    fn default() -> Self {
        ConvertOpts {
            ebits: 8,
            io_bits: 8,
            n_layers: 78,
            keep_indexer: false,
            xfp8: false,
            mtp_only: false,
            minimax: false,
            gemma_norm: false,
        }
    }
}

/// Map a MiniMax-M3 source tensor name to its colibrì-container (GLM-style) name, or
/// `None` to drop it. Strips the `language_model.` prefix, renames the MoE block
/// (`block_sparse_moe` → `mlp`, expert `w1/w2/w3` → `gate/down/up_proj`), and drops the
/// vision tower + multimodal projectors + the MTP/next-n module (text-only MVP). The
/// attention (`self_attn.{q,k,v,o}_proj`, `q_norm`/`k_norm`) and dense/norm names already
/// match the container, so they pass through unchanged. Sidecar scales ride along via the
/// same substring rewrites (`.w1.weight_scale` → `.gate_proj.weight_scale`).
fn m3_container_name(name: &str) -> Option<String> {
    for drop in [
        "vision_tower",
        "multi_modal_projector",
        "patch_merge_mlp",
        "image_",
        "visual",
        ".mtp",
        "nextn",
        "num_nextn",
    ] {
        if name.contains(drop) {
            return None;
        }
    }
    let mut n = name.strip_prefix("language_model.").unwrap_or(name).to_string();
    n = n.replace(".block_sparse_moe.", ".mlp.");
    // Expert sub-weights: w1 = gate, w3 = up, w2 = down.
    n = n
        .replace(".w1.", ".gate_proj.")
        .replace(".w3.", ".up_proj.")
        .replace(".w2.", ".down_proj.");
    Some(n)
}

/// What conversion should do with a tensor. `Skip` folds the Python `"skip"` and
/// `"consumed"` cases (dropped, or handled alongside their weight).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum Kind {
    Skip,
    /// keep as full F32 (norms, router, biases, correction bias)
    F32,
    /// embeddings / lm_head → `io_bits`
    Io,
    /// routed expert weight → NVFP4 (or e4m3 under `xfp8`)
    X,
    /// resident weight (attention / dense MLP / shared) → `ebits`
    Q,
}

/// `model.layers.<i>.…` → `i`, else `-1`. Port of `layer_idx`.
fn layer_idx(name: &str) -> i64 {
    let mut it = name.split('.');
    if it.next() == Some("model") && it.next() == Some("layers") {
        if let Some(s) = it.next() {
            if let Ok(i) = s.parse::<i64>() {
                return i;
            }
        }
    }
    -1
}

/// Tensor classification — port of `classify(name, n_layers)`. `keep_idx` mirrors the
/// reference's `--indexer` pass: retain the DSA lightning-indexer weights instead of
/// dropping them.
fn classify(name: &str, n_layers: usize, keep_idx: bool, mtp_only: bool) -> Kind {
    // scale sidecars are consumed with their weight
    if name.ends_with("_scale_inv") {
        return Kind::Skip;
    }
    if name.ends_with(".weight_scale")
        || name.ends_with(".weight_scale_2")
        || name.ends_with(".input_scale")
    {
        return Kind::Skip;
    }
    let li = layer_idx(name);
    let is_mtp = li >= 0 && li as usize == n_layers;
    // The MTP speculative head lives at layer index `n_layers` (the single
    // `num_nextn_predict_layers` block). It is KEPT by default so every container ships
    // MTP-ready — drafting stays opt-in at runtime (`DRAFT=n`; `MTP=0` forces it off).
    // Only layers ABOVE the head (not part of this architecture) are dropped.
    if li >= 0 && li as usize > n_layers {
        return Kind::Skip;
    }
    if mtp_only && !is_mtp {
        // `COLI_MTP_ONLY`: emit *just* the head, to augment an existing head-less
        // container (base layers, embeddings, lm_head, final norm already live there)
        // without re-converting the whole model.
        return Kind::Skip;
    }
    // DSA lightning indexer (`self_attn.indexer.{wk,wq_b,weights_proj,k_norm}`). Dropped
    // by default; kept when `keep_idx` so the container can run DSA — the wk/wq_b/
    // weights_proj matrices quantize as resident (`Q`), k_norm stays f32.
    if name.contains(".indexer.") {
        if !keep_idx {
            return Kind::Skip;
        }
        return if name.contains("k_norm") { Kind::F32 } else { Kind::Q };
    }
    for k in ["indexers_proj", "eh_proj", "enorm", "hnorm", "shared_head"] {
        if name.contains(k) {
            // `eh_proj`/`enorm`/`hnorm`/`shared_head.norm` ARE the MTP head's own fusion
            // inputs (see `load_mtp`'s required set), so keep them for the head (layer
            // `n_layers`) and drop them for the base model. These names only ever occur
            // on the head, so `is_mtp` selects them exactly. `shared_head.head`
            // duplicates lm_head and `indexers_proj` is unused — always dropped.
            if is_mtp {
                if name.contains("eh_proj") {
                    return Kind::Q;
                }
                if name.contains("enorm") || name.contains("hnorm") || name.contains("shared_head.norm")
                {
                    return Kind::F32;
                }
            }
            return Kind::Skip;
        }
    }
    if name.ends_with("e_score_correction_bias") {
        return Kind::F32;
    }
    if name.ends_with("mlp.gate.weight") {
        return Kind::F32; // router (NOT gate_proj)
    }
    if name.ends_with("norm.weight") || name == "model.norm.weight" {
        return Kind::F32;
    }
    if name == "model.embed_tokens.weight" || name == "lm_head.weight" {
        return Kind::Io;
    }
    if name.contains(".mlp.experts.") && name.ends_with(".weight") {
        return Kind::X; // routed expert (streamed)
    }
    if name.ends_with(".weight") {
        return Kind::Q; // attention / dense MLP / shared (resident)
    }
    Kind::F32
}

/// Materialize a tensor as f32, returning `(data, logical_shape)`. The logical
/// shape differs from the on-disk shape for NVFP4, whose weight is stored
/// *packed* as `[O, I/2]` — callers must quantize against the logical `[O, I]`.
///
/// Dispatch (port of `dequant()`):
///   - `F8_E4M3`/`F8_E5M2` + `name_scale_inv` → 128×128 block-scaled FP8
///   - `U8` + `name_scale`                    → NVFP4 (modelopt)
///   - BF16/F16/F32                           → straight convert
fn dequant(shards: &Shards, name: &str) -> io::Result<(Vec<f32>, Vec<i64>)> {
    let t = shards
        .find(name)
        .ok_or_else(|| io::Error::new(io::ErrorKind::NotFound, format!("missing tensor: {name}")))?;

    match t.dtype {
        // Per-tensor FP8 (modelopt): a single F32 `name_scale`. Real modelopt NVFP4
        // checkpoints use this for the non-expert linears, alongside NVFP4 experts.
        // The weight dtype is what separates this from NVFP4 — both carry `_scale`.
        DType::F8E4M3 | DType::F8E5M2
            if !shards.has(&format!("{name}_scale_inv")) && shards.has(&format!("{name}_scale")) =>
        {
            let sname = format!("{name}_scale");
            let st = shards.find(&sname).unwrap();
            let mut s = vec![0f32; st.numel.max(1) as usize];
            shards.read_f32(&sname, &mut s)?;
            let scale = s[0];
            let mut w = vec![0f32; t.numel.max(0) as usize];
            shards.read_f32(name, &mut w)?;
            for v in w.iter_mut() {
                *v *= scale;
            }
            Ok((w, t.shape.clone()))
        }
        DType::F8E4M3 | DType::F8E5M2 => {
            // Block-scaled FP8. Two layouts, distinguished by the scale dtype:
            //   * F32 `[⌈O/128⌉, ⌈I/128⌉]` — modelopt 128×128 blocks (GLM-5.2).
            //   * U8  `[nbo, nbi]`           — OCP MX-FP8 E8M0 power-of-2 scales
            //     (MiniMax-M3): one exponent per `[O/nbo × I/nbi]` block (per-row ×
            //     block-32 in practice). scale = 2^(e − 127).
            // Both MULTIPLY: W[o,i] = fp8(o,i) · scale(block of (o,i)). The scale
            // sidecar is the weight name with `_scale_inv` appended.
            let (o, i) = two_dims(&t.shape, name)?;
            let sname = format!("{name}_scale_inv");
            let st = shards.find(&sname).ok_or_else(|| {
                io::Error::new(
                    io::ErrorKind::NotFound,
                    format!(
                        "FP8 weight {name} has neither {sname} (block scales) \
                         nor {name}_scale (per-tensor scale)"
                    ),
                )
            })?;
            let (nbo, nbi) = two_dims(&st.shape, &sname)?;

            // fp8 codes → f32 (byte-per-element), then scale by block.
            let mut w = vec![0f32; o * i];
            shards.read_f32(name, &mut w)?; // convert_to_f32 decodes F8_E4M3/E5M2

            match st.dtype {
                DType::U8 => {
                    // MX-FP8: read the raw E8M0 exponent bytes (read_f32 would mangle
                    // them). Block extent = O/nbo along rows, I/nbi along inputs.
                    let bo = (o / nbo.max(1)).max(1);
                    let bi = (i / nbi.max(1)).max(1);
                    let mut raw = vec![0u8; st.nbytes as usize];
                    shards.read_raw(&sname, &mut raw)?;
                    for oo in 0..o {
                        let srow = (oo / bo) * nbi;
                        let wrow = &mut w[oo * i..(oo + 1) * i];
                        for (ii, wv) in wrow.iter_mut().enumerate() {
                            let e = raw[srow + ii / bi] as i32;
                            *wv *= f32::exp2((e - 127) as f32);
                        }
                    }
                }
                _ => {
                    // modelopt 128×128 F32 block grid.
                    let mut scale = vec![0f32; nbo * nbi];
                    shards.read_f32(&sname, &mut scale)?;
                    for oo in 0..o {
                        let srow = (oo / BLOCK) * nbi;
                        let wrow = &mut w[oo * i..(oo + 1) * i];
                        for (ii, wv) in wrow.iter_mut().enumerate() {
                            *wv *= scale[srow + ii / BLOCK];
                        }
                    }
                }
            }
            Ok((w, t.shape.clone()))
        }
        DType::U8 if shards.has(&format!("{name}_scale")) => {
            let (w, o, i) = dequant_nvfp4(shards, name)?;
            Ok((w, vec![o as i64, i as i64]))
        }
        _ => {
            let mut w = vec![0f32; t.numel.max(0) as usize];
            shards.read_f32(name, &mut w)?;
            Ok((w, t.shape.clone()))
        }
    }
}

/// NVIDIA **modelopt** NVFP4 → f32 `[O, I]`. Port of `dequant_nvfp4`.
///
/// Layout:
///   - `name`            `U8`      `[O, I/2]` — two e2m1 nibbles per byte along the
///     input (contraction) dim; **low nibble = even element, high = odd**.
///   - `name_scale`      `F8_E4M3` `[O, ⌈I/16⌉]` — per-16-element block scale.
///   - `name_scale_2`    `F32`     scalar — per-tensor global scale (~amax/(6*448)).
///
/// `W[o,i] = e2m1[nibble] * block_scale[o, i/16] * scale_2` — modelopt **multiplies**
/// both scales.
///
/// FOOTGUN (guarded): llm-compressor/compressed-tensors stores the *reciprocal*
/// (a large global) and **divides**; modelopt stores the small value and multiplies.
/// A `scale_2 >= 1` almost certainly means a compressed-tensors checkpoint, so we
/// refuse rather than silently corrupt every tensor. The block-scale grid must also
/// be the flat per-16 layout (no cutlass/TensorRT swizzle or padding) — verified,
/// not inferred, since inferring `gs = I/ncols` misaligns silently.
fn dequant_nvfp4(shards: &Shards, name: &str) -> io::Result<(Vec<f32>, usize, usize)> {
    let t = shards.find(name).unwrap();
    let (o, ih) = two_dims(&t.shape, name)?;
    let i = ih * 2;

    // per-tensor global scale — and the modelopt-vs-compressed-tensors guard.
    let gname = format!("{name}_scale_2");
    let gt = shards.find(&gname).ok_or_else(|| {
        io::Error::new(
            io::ErrorKind::NotFound,
            format!("NVFP4 weight {name} has no {gname} (global scale)"),
        )
    })?;
    let mut g = vec![0f32; gt.numel.max(1) as usize];
    shards.read_f32(&gname, &mut g)?;
    let gscale = g[0];
    if !(gscale < 1.0) {
        return Err(io::Error::new(
            io::ErrorKind::InvalidData,
            format!(
                "{name}: {gname}={gscale:e} >= 1 looks like the reciprocal \
                 (compressed-tensors/llm-compressor, which DIVIDES); this path \
                 implements modelopt (which MULTIPLIES a small global scale). \
                 Refusing rather than silently corrupting every tensor."
            ),
        ));
    }

    // per-16-block scales, stored f8e4m3 (read_f32 decodes them).
    let sname = format!("{name}_scale");
    let st = shards.find(&sname).unwrap();
    let (nbo, nbi) = two_dims(&st.shape, &sname)?;
    let nb = i.div_ceil(NVFP4_GS);
    if nbi != nb || nbo != o {
        return Err(io::Error::new(
            io::ErrorKind::InvalidData,
            format!(
                "{sname}: block-scale grid is [{nbo},{nbi}], expected [{o},{nb}] \
                 = [O, ceil(I/{NVFP4_GS})]; unexpected layout (swizzled/padded?), refusing"
            ),
        ));
    }
    let mut bscale = vec![0f32; nbo * nbi];
    shards.read_f32(&sname, &mut bscale)?;

    // packed nibbles → e2m1 values, scaled by block and global.
    let mut raw = vec![0u8; t.nbytes as usize];
    shards.read_raw(name, &mut raw)?;
    let mut w = vec![0f32; o * i];
    for oo in 0..o {
        let srow = oo * nbi;
        for iih in 0..ih {
            let b = raw[oo * ih + iih];
            let (i0, i1) = (iih * 2, iih * 2 + 1);
            w[oo * i + i0] = E2M1[(b & 0x0F) as usize] * bscale[srow + i0 / NVFP4_GS] * gscale;
            if i1 < i {
                w[oo * i + i1] =
                    E2M1[((b >> 4) & 0x0F) as usize] * bscale[srow + i1 / NVFP4_GS] * gscale;
            }
        }
    }
    Ok((w, o, i))
}

fn two_dims(shape: &[i64], name: &str) -> io::Result<(usize, usize)> {
    if shape.len() != 2 {
        return Err(io::Error::new(
            io::ErrorKind::InvalidData,
            format!("{name}: expected a 2D tensor, got shape {shape:?}"),
        ));
    }
    Ok((shape[0] as usize, shape[1] as usize))
}

/// One tensor destined for an output shard.
struct OutTensor {
    name: String,
    dtype: &'static str, // "U8" | "F32"
    shape: Vec<i64>,
    bytes: Vec<u8>,
}

fn f32_bytes(v: &[f32]) -> Vec<u8> {
    let mut out = Vec::with_capacity(v.len() * 4);
    for x in v {
        out.extend_from_slice(&x.to_le_bytes());
    }
    out
}

/// Quantize a 2D resident weight to the container form: per-row int8 `U8` codes +
/// `F32` per-row scales. Resident/io weights are int8 (`bits >= 8`); sub-8-bit
/// widths are no longer produced (int4/int2 removed here — routed experts use the
/// NVFP4/e4m3 paths instead).
fn quantize(name: &str, w: &[f32], o: usize, i: usize, bits: u32) -> (OutTensor, OutTensor) {
    assert!(bits >= 8, "quantize() produces int8 only; got bits={bits} (< 8)");
    let (q, s) = quantize_rows(w, o, i, bits);
    let codes: Vec<u8> = q.iter().map(|&x| x as u8).collect();
    let scale = s;
    let codes_t = OutTensor {
        name: name.to_string(),
        dtype: "U8",
        shape: vec![codes.len() as i64],
        bytes: codes,
    };
    let scale_t = OutTensor {
        name: format!("{name}.qs"),
        dtype: "F32",
        shape: vec![scale.len() as i64],
        bytes: f32_bytes(&scale),
    };
    (codes_t, scale_t)
}

/// Encode f32 → e4m3 fp8 (OCP: 1 sign, 4 exp bias-7, 3 mantissa; no infinity; max
/// normal ±448; round-to-nearest on the mantissa). The caller pre-scales into range,
/// so saturation is a safety net. NaN/0 → signed zero.
pub(crate) fn float_to_e4m3(x: f32) -> u8 {
    let sign: u8 = if x.is_sign_negative() { 0x80 } else { 0x00 };
    let a = x.abs();
    if !(a > 0.0) {
        return sign;
    }
    if a >= 448.0 {
        return sign | 0x7E; // saturate to max normal (e=15, m=6 = 448)
    }
    let e = a.log2().floor() as i32; // unbiased exponent: 2^e <= a < 2^(e+1)
    if e < -6 {
        // subnormal: value = m/8 · 2^-6, so m = a · 2^6 · 8 = a · 512. Round-to-even
        // matches the hardware fp8 encoder (__nv_cvt_float_to_fp8).
        let m = (a * 512.0).round_ties_even() as i32;
        if m >= 8 {
            return sign | 0x08; // rounded up into the smallest normal (2^-6)
        }
        return sign | (m.max(0) as u8);
    }
    // normal: value = (1 + m/8) · 2^e
    let mut m = ((a * 2f32.powi(-e) - 1.0) * 8.0).round_ties_even() as i32;
    let mut eb = e + 7;
    if m >= 8 {
        m = 0;
        eb += 1; // mantissa carried into the next binade
    }
    if eb >= 15 && m > 6 {
        return sign | 0x7E; // e=15,m=7 is NaN → saturate to 448
    }
    sign | ((eb as u8) << 3) | (m as u8 & 0x07)
}

/// Per-row absmax e4m3 quantization for routed experts: scale = max|w|/448 per row so
/// the row fits e4m3's range, store e4m3(w/scale) codes + the f32 scale. Same output
/// layout as [`quantize`] (U8 codes + `{name}.qs` F32 scales) but 8-bit fp precision
/// (1 byte/weight) — preserves the source FP8's precision.
fn quantize_e4m3(name: &str, w: &[f32], o: usize, i: usize) -> (OutTensor, OutTensor) {
    let mut codes = vec![0u8; o * i];
    let mut scale = vec![0f32; o];
    for r in 0..o {
        let row = &w[r * i..(r + 1) * i];
        let amax = row.iter().fold(0f32, |m, &x| m.max(x.abs()));
        let s = if amax > 0.0 { amax / 448.0 } else { 1.0 };
        let inv = 1.0 / s;
        for c in 0..i {
            codes[r * i + c] = float_to_e4m3(row[c] * inv);
        }
        scale[r] = s;
    }
    let codes_t = OutTensor {
        name: name.to_string(),
        dtype: "U8",
        shape: vec![(o * i) as i64],
        bytes: codes,
    };
    let scale_t = OutTensor {
        name: format!("{name}.qs"),
        dtype: "F32",
        shape: vec![o as i64],
        bytes: f32_bytes(&scale),
    };
    (codes_t, scale_t)
}

/// Nearest e2m1 code (0..15, bit 3 = sign) for `t`. Mirrors [`e2m1_round`] but returns
/// the packed nibble instead of the value; ties resolve to the first (even) magnitude,
/// matching `e2m1_round`/the CUDA LUT decode.
fn e2m1_code(t: f32) -> u8 {
    let a = t.abs();
    let mut best = 0usize;
    let mut bd = f32::INFINITY;
    for (idx, &c) in E2M1_LEVELS.iter().enumerate() {
        let d = (a - c).abs();
        if d < bd {
            bd = d;
            best = idx;
        }
    }
    (if t.is_sign_negative() { 8u8 } else { 0 }) | best as u8
}

/// Encode `[o, i]` f32 → NVFP4: `(nibbles, block_scales, global)`.
///   - `nibbles`      packed e2m1, 2/byte, low nibble = even column, `o*ceil(i/2)` bytes
///   - `block_scales` one ue4m3 byte per 16 inputs, `o*ceil(i/16)` bytes, row-major
///   - `global`       one f32 (modelopt-style, multiplied): `amax / (6 * 448)`
///
/// The block scale is encoded with [`float_to_e4m3`] and the effective divisor is read
/// **back** from that byte (`f8e4m3_to_f32(code) * global`), so encode and the kernel/CPU
/// decode agree exactly. This is the real quantizer behind [`quantize_nvfp4_sim`]'s
/// reconstruction (which was scored at 9.4% rel-RMS on the real experts).
fn quantize_nvfp4(w: &[f32], o: usize, i: usize) -> (Vec<u8>, Vec<u8>, f32) {
    let amax = w.iter().fold(0f32, |m, &v| m.max(v.abs()));
    let global = (amax / (E2M1_LEVELS[7] * UE4M3_MAX)).max(f32::MIN_POSITIVE);
    let nb = i.div_ceil(NVFP4_BLOCK);
    let rb = i.div_ceil(2);
    let mut nib = vec![0u8; o * rb];
    let mut bsc = vec![0u8; o * nb];
    for r in 0..o {
        for b in 0..nb {
            let c0 = b * NVFP4_BLOCK;
            let c1 = ((b + 1) * NVFP4_BLOCK).min(i);
            let blk = &w[r * i + c0..r * i + c1];
            let bmax = blk.iter().fold(0f32, |m, &v| m.max(v.abs()));
            let code = float_to_e4m3(bmax / E2M1_LEVELS[7] / global);
            bsc[r * nb + b] = code;
            let eff = colibri_core::dtype::f8e4m3_to_f32(code) * global;
            for (k, &v) in blk.iter().enumerate() {
                let cd = if eff > 0.0 { e2m1_code(v / eff) } else { 0 };
                let c = c0 + k;
                if c & 1 == 0 {
                    nib[r * rb + (c >> 1)] |= cd;
                } else {
                    nib[r * rb + (c >> 1)] |= cd << 4;
                }
            }
        }
    }
    (nib, bsc, global)
}

/// gate/up_proj are `[moe_inter, hidden]`; down_proj is `[hidden, moe_inter]`.
fn expert_oi(name: &str, hidden: usize, moe_inter: usize) -> (usize, usize) {
    if name.contains("down_proj") {
        (hidden, moe_inter)
    } else {
        (moe_inter, hidden)
    }
}

/// One re-quantized tensor: dropped (a consumed expert `.qs`), copied through raw, or an
/// expert weight turned into (weight-blob U8 = nibbles++block-scales, global F32). The
/// block scales are CONCATENATED onto the nibbles in a single U8 tensor so the loader's
/// coalesced gate/up/down read grabs them together, zero-copy — a separate `.bs` sidecar
/// would cost one uncoalesced random-seek pread per expert (measured 15x slower decode).
enum ReqOut {
    Skip,
    Raw(OutTensor),
    Nvfp4(OutTensor, OutTensor),
}

/// Copy one tensor through unchanged (raw on-disk bytes, same dtype + shape).
fn copy_raw(name: &str, shards: &Shards) -> io::Result<ReqOut> {
    let t = shards
        .find(name)
        .ok_or_else(|| io::Error::new(io::ErrorKind::NotFound, format!("missing tensor: {name}")))?;
    let mut bytes = vec![0u8; t.nbytes as usize];
    shards.read_raw(name, &mut bytes)?;
    Ok(ReqOut::Raw(OutTensor {
        name: name.to_string(),
        dtype: t.dtype.safetensors_str(),
        shape: t.shape.clone(),
        bytes,
    }))
}

/// Re-quantize one container tensor: expert e4m3 weight → NVFP4; its `.qs` dropped;
/// everything else copied through byte-for-byte.
fn requant_one(
    name: &str,
    shards: &Shards,
    n_layers: usize,
    hidden: usize,
    moe_inter: usize,
) -> io::Result<ReqOut> {
    // A per-row `.qs` belonging to an expert weight is consumed by that weight's NVFP4
    // encoding (which reads it to dequant e4m3); drop it. Resident `.qs` copies through.
    if let Some(base) = name.strip_suffix(".qs") {
        if classify(base, n_layers, true, false) == Kind::X {
            return Ok(ReqOut::Skip);
        }
        return copy_raw(name, shards);
    }
    if name.ends_with(".weight") && classify(name, n_layers, true, false) == Kind::X {
        let (o, i) = expert_oi(name, hidden, moe_inter);
        let t = shards.find(name).ok_or_else(|| {
            io::Error::new(io::ErrorKind::NotFound, format!("missing tensor: {name}"))
        })?;
        if t.numel as usize != o * i {
            return Err(io::Error::new(
                io::ErrorKind::InvalidData,
                format!("{name}: expected {o}x{i}={} e4m3 codes, got {}", o * i, t.numel),
            ));
        }
        let mut codes = vec![0u8; o * i];
        shards.read_raw(name, &mut codes)?;
        let mut qs = vec![0f32; o];
        shards.read_f32(&format!("{name}.qs"), &mut qs)?;
        let mut w = vec![0f32; o * i];
        for r in 0..o {
            let s = qs[r];
            for c in 0..i {
                w[r * i + c] = colibri_core::dtype::f8e4m3_to_f32(codes[r * i + c]) * s;
            }
        }
        let (mut blob, bsc, g) = quantize_nvfp4(&w, o, i);
        blob.extend_from_slice(&bsc); // weight = nibbles ++ block-scales (one coalesced read)
        return Ok(ReqOut::Nvfp4(
            OutTensor { name: name.to_string(), dtype: "U8", shape: vec![blob.len() as i64], bytes: blob },
            OutTensor { name: format!("{name}.g"), dtype: "F32", shape: vec![1], bytes: f32_bytes(&[g]) },
        ));
    }
    copy_raw(name, shards)
}

/// Re-quantize the routed experts of an existing colibrì **e4m3 container** to NVFP4,
/// copying every other tensor through byte-for-byte. Container→container because the
/// source FP8 checkpoint is no longer on disk; the e4m3 experts are 8-bit, so decoding
/// them to f32 and re-quantizing to 4-bit NVFP4 loses essentially nothing beyond NVFP4's
/// own floor. Only `Kind::X` expert weights change; resident/router/norm/indexer stay
/// exactly as they were (quality-critical, already 8-bit int).
///
/// Per expert weight `W`: `W` U8 = packed e2m1 nibbles (`O*ceil(I/2)`) CONCATENATED with
/// ue4m3 block scales (`O*ceil(I/16)`), and `W.g` F32 global. Concatenating means the
/// loader's one coalesced gate/up/down read grabs the block scales too, zero-copy — a
/// separate `.bs` tensor cost one uncoalesced random-seek pread per expert (15x slower
/// decode, measured). Expert blobs stay contiguous per shard so `load_expert` coalesces
/// gate/up/down. `keep_indexer=true` (DSA container) is assumed.
pub fn requant_experts_nvfp4(
    indir: impl AsRef<Path>,
    outdir: impl AsRef<Path>,
    n_layers: usize,
    hidden: usize,
    moe_inter: usize,
    mut progress: impl FnMut(usize, usize, &ConvertStats),
) -> io::Result<ConvertStats> {
    let indir = indir.as_ref();
    let outdir = outdir.as_ref();
    std::fs::create_dir_all(outdir)?;
    let shards = Shards::open(indir)?;
    let nfiles = shards.num_files();
    let mut by_file: Vec<Vec<&str>> = vec![Vec::new(); nfiles];
    for t in shards.tensors() {
        by_file[t.file_idx].push(&t.name);
    }
    let cap = std::env::var("COLI_CONVERT_THREADS")
        .ok()
        .and_then(|v| v.parse::<usize>().ok())
        .filter(|&t| t > 0)
        .unwrap_or_else(|| std::thread::available_parallelism().map(|p| p.get()).unwrap_or(4));

    // Reborrow as `&Shards` so the worker `move` closures copy the reference instead of
    // moving the owned `Shards` (which `by_file` still borrows). Mirrors `process_names_parallel`.
    let sref: &Shards = &shards;
    let mut stats = ConvertStats::default();
    for (fi, names) in by_file.iter().enumerate() {
        // Parallel dequant + NVFP4 encode across cores, order preserved.
        let n = names.len();
        let mut parts: Vec<io::Result<Vec<ReqOut>>> = Vec::new();
        let nthreads = cap.min(n.max(1));
        let chunk = n.div_ceil(nthreads.max(1));
        std::thread::scope(|scope| {
            let handles: Vec<_> = names
                .chunks(chunk.max(1))
                .map(|slice| {
                    scope.spawn(move || {
                        slice
                            .iter()
                            .map(|&nm| requant_one(nm, sref, n_layers, hidden, moe_inter))
                            .collect::<io::Result<Vec<_>>>()
                    })
                })
                .collect();
            for h in handles {
                parts.push(h.join().unwrap());
            }
        });
        let mut codes: Vec<OutTensor> = Vec::new();
        let mut floats: Vec<OutTensor> = Vec::new();
        for part in parts {
            for out in part? {
                match out {
                    ReqOut::Skip => stats.tensors_skipped += 1,
                    ReqOut::Raw(t) => {
                        if t.dtype == "U8" {
                            codes.push(t);
                        } else {
                            floats.push(t);
                        }
                        stats.tensors_f32 += 1;
                    }
                    ReqOut::Nvfp4(blob, g) => {
                        codes.push(blob); // weight = nibbles++block-scales, kept contiguous
                        floats.push(g);
                        stats.tensors_quantized += 1;
                    }
                }
            }
        }
        if !codes.is_empty() || !floats.is_empty() {
            codes.extend(floats);
            let path = outdir.join(format!("out-{fi:05}.safetensors"));
            write_shard(&path, &codes)?;
            stats.shards_written += 1;
            stats.bytes_out += codes.iter().map(|t| t.bytes.len() as u64).sum::<u64>();
        }
        progress(fi, nfiles, &stats);
    }
    for fname in [
        "config.json",
        "tokenizer.json",
        "tokenizer_config.json",
        "generation_config.json",
        "special_tokens_map.json",
    ] {
        let src = indir.join(fname);
        if src.exists() {
            std::fs::copy(&src, outdir.join(fname))?;
        }
    }
    Ok(stats)
}

/// Emit a routed expert as an NVFP4 container tensor pair, matching what the loader
/// (`moe::expert_from_views` fmt=5) reads: `name` = U8 blob of packed e2m1 nibbles
/// CONCATENATED with ue4m3 block scales (so the coalesced gate/up/down read grabs both),
/// and `{name}.g` = the F32 per-tensor global. Same shape as [`quantize`]'s output pair
/// (codes + scale sidecar), so it drops into the existing `TensorOut::Quant` path.
fn quantize_nvfp4_out(name: &str, w: &[f32], o: usize, i: usize) -> (OutTensor, OutTensor) {
    let (mut blob, bsc, g) = quantize_nvfp4(w, o, i);
    blob.extend_from_slice(&bsc);
    (
        OutTensor { name: name.to_string(), dtype: "U8", shape: vec![blob.len() as i64], bytes: blob },
        OutTensor { name: format!("{name}.g"), dtype: "F32", shape: vec![1], bytes: f32_bytes(&[g]) },
    )
}

/// Write a safetensors shard from tensors already materialized in memory.
fn write_shard(path: &Path, tensors: &[OutTensor]) -> io::Result<()> {
    let mut header = String::from("{");
    let mut off = 0usize;
    for (k, t) in tensors.iter().enumerate() {
        if k > 0 {
            header.push(',');
        }
        let shape: Vec<String> = t.shape.iter().map(|d| d.to_string()).collect();
        header.push_str(&format!(
            "\"{}\":{{\"dtype\":\"{}\",\"shape\":[{}],\"data_offsets\":[{},{}]}}",
            t.name,
            t.dtype,
            shape.join(","),
            off,
            off + t.bytes.len()
        ));
        off += t.bytes.len();
    }
    header.push('}');
    let hb = header.as_bytes();
    let mut f = BufWriter::new(std::fs::File::create(path)?);
    f.write_all(&(hb.len() as u64).to_le_bytes())?;
    f.write_all(hb)?;
    for t in tensors {
        f.write_all(&t.bytes)?;
    }
    f.flush()
}

/// What kind of snapshot a directory holds, keyed on the distinctive scale
/// sidecar each format carries.
/// What re-quantizing one source tensor at some bit width costs, measured against
/// the checkpoint's own values.
#[derive(Debug, Clone)]
pub struct TensorErr {
    pub name: String,
    pub o: usize,
    pub i: usize,
    /// RMS(error) / RMS(reference) — scale-free, so tensors are comparable.
    pub rms_rel: f64,
    /// Largest single-weight absolute error, relative to the tensor's RMS.
    pub max_rel: f64,
    /// Signal-to-noise of the round trip, dB. Higher is better; +6 dB ≈ 1 extra bit.
    pub snr_db: f64,
}

/// Which quantization scheme to score in [`quant_error`].
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Scheme {
    /// What we ship: per-row linear int-N with one f32 scale per output row.
    Int(u32),
    /// NVFP4: e2m1 data, one ue4m3 scale per 16 inputs, plus a per-tensor f32 scale.
    /// Simulated numerically here — no kernel, no container change.
    Nvfp4,
}

/// The eight magnitudes e2m1 can represent (1 sign, 2 exponent, 1 mantissa bit).
const E2M1_LEVELS: [f32; 8] = [0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0];
/// NVFP4 scales one ue4m3 factor per this many inputs (vs one per-row scale for int-N).
const NVFP4_BLOCK: usize = 16;
/// Largest finite ue4m3 scale (e=15,m=6; m=7 is NaN in the `fn` variant).
const UE4M3_MAX: f32 = 448.0;

fn e2m1_round(v: f32) -> f32 {
    let a = v.abs();
    let mut best = 0f32;
    let mut bd = f32::INFINITY;
    for &c in &E2M1_LEVELS {
        let d = (a - c).abs();
        if d < bd {
            bd = d;
            best = c;
        }
    }
    if v.is_sign_negative() {
        -best
    } else {
        best
    }
}

/// Round a positive scale to the nearest representable unsigned e4m3 value.
fn ue4m3_round(v: f32) -> f32 {
    if !(v > 0.0) {
        return 0.0;
    }
    let mut best = 0f32;
    let mut bd = f32::INFINITY;
    let mut consider = |c: f32| {
        let d = (v - c).abs();
        if d < bd {
            bd = d;
            best = c;
        }
    };
    for m in 0..8 {
        consider(2f32.powi(-6) * (m as f32 / 8.0)); // subnormals
    }
    for e in 1..16 {
        for m in 0..8 {
            if e == 15 && m == 7 {
                continue; // NaN
            }
            consider(2f32.powi(e - 7) * (1.0 + m as f32 / 8.0));
        }
    }
    best
}

/// Reconstruct what NVFP4 would actually represent, using the standard two-level
/// recipe: a per-tensor f32 scale brings block scales into ue4m3's range, then each
/// 16-input block gets its own ue4m3 scale and the values become e2m1 codes.
///
/// Simulating rather than implementing means the format's accuracy can be scored
/// before committing to block-scaled MMA kernels and a container change. NVFP4 is
/// ~0.56 bytes/weight against int4's 0.5 (e2m1 plus one ue4m3 per 16 inputs) — what
/// that byte difference costs in throughput is NOT measured here and should not be
/// inferred from it.
pub(crate) fn quantize_nvfp4_sim(w: &[f32], o: usize, i: usize) -> Vec<f32> {
    let amax = w.iter().fold(0f32, |m, &v| m.max(v.abs()));
    let global = (amax / (E2M1_LEVELS[7] * UE4M3_MAX)).max(f32::MIN_POSITIVE);
    let mut out = vec![0f32; o * i];
    for r in 0..o {
        let mut c = 0;
        while c < i {
            let end = (c + NVFP4_BLOCK).min(i);
            let blk = &w[r * i + c..r * i + end];
            let bmax = blk.iter().fold(0f32, |m, &v| m.max(v.abs()));
            let sf = ue4m3_round(bmax / E2M1_LEVELS[7] / global);
            let eff = sf * global;
            for (k, &v) in blk.iter().enumerate() {
                out[r * i + c + k] = if eff > 0.0 { e2m1_round(v / eff) * eff } else { 0.0 };
            }
            c = end;
        }
    }
    out
}

/// Reconstruct the f32 values a [`QTensor`] actually represents — the inverse of
/// [`qtensor_from_f32`], i.e. what the kernels will really multiply.
fn dequantize_qtensor(t: &QTensor) -> Vec<f32> {
    let (o, i) = (t.o as usize, t.i as usize);
    let mut out = vec![0f32; o * i];
    match t.fmt_code {
        0 => out.copy_from_slice(&t.qf),
        1 => {
            for r in 0..o {
                for c in 0..i {
                    out[r * i + c] = t.q8[r * i + c] as f32 * t.s[r];
                }
            }
        }
        _ => {} // int2 unused for resident weights; leave zeroed rather than lie
    }
    out
}

/// Error of re-quantizing the source's own weights under `scheme`, per tensor.
///
/// **Why this exists.** The converter reads block-scaled FP8 (e4m3 + 128x128 scales),
/// dequantizes to f32, then re-quantizes with our own per-row scales. Native FP8
/// compute would instead pass the checkpoint's bytes through untouched — worth
/// building only if that round trip is actually losing something. This measures the
/// loss directly, with no kernels and no conversion.
///
/// **Scope.** This reports weight-reconstruction error and nothing else. It does not
/// measure perplexity, throughput, or bytes moved per token, and none of those follow
/// from it: a scheme with lower error may be slower, larger, or no better end-to-end.
/// Treat the numbers as one input to that question, not the answer.
///
/// `experts` selects which population to report:
/// - `false` → resident weights ([`Kind::Q`]): attention/dense/shared. 2.5% of params,
///   but measured to matter enormously — 4-bit resident put perplexity at 48.665,
///   int8 at 6.189, which is why resident weights ship int8.
/// - `true` → routed experts ([`Kind::X`]): 97.5% of params and 58% of the weights a
///   token touches, shipped as NVFP4 (4-bit block-scaled). This probe reaches them
///   without converting anything, so an expert scheme can be scored on the real
///   weights before committing to a container.
pub fn quant_error(
    indir: impl AsRef<Path>,
    scheme: Scheme,
    n_layers: usize,
    limit: usize,
    experts: bool,
) -> io::Result<Vec<TensorErr>> {
    let want = if experts { Kind::X } else { Kind::Q };
    let shards = Shards::open(indir.as_ref())?;
    let mut names: Vec<&str> = shards
        .tensors()
        .iter()
        .map(|t| t.name.as_str())
        .filter(|n| classify(n, n_layers, false, false) == want)
        .collect();
    names.sort_unstable();

    // Stride the sample across the whole population instead of taking the first N.
    // Sorted names cluster by layer, so first-N collapses onto layer 0 — which for
    // resident weights means *no attention tensors at all*, exactly where the error
    // is worst, and layer 0 is a dense layer rather than one of the 75 MoE ones. A
    // first-N sample silently answers a different question than the one asked.
    let stride = (names.len() / limit.max(1)).max(1);
    let names: Vec<&str> = names.into_iter().step_by(stride).take(limit).collect();

    let mut out = Vec::new();
    for name in names {
        let (w, shape) = dequant(&shards, name)?;
        if shape.len() != 2 {
            continue;
        }
        let (o, i) = (shape[0] as usize, shape[1] as usize);
        let approx = match scheme {
            Scheme::Int(bits) => dequantize_qtensor(&qtensor_from_f32(&w, o, i, bits)),
            Scheme::Nvfp4 => quantize_nvfp4_sim(&w, o, i),
        };

        let mut sq_ref = 0f64;
        let mut sq_err = 0f64;
        let mut max_abs = 0f64;
        for (&r, &a) in w.iter().zip(&approx) {
            let e = (r - a) as f64;
            sq_ref += (r as f64) * (r as f64);
            sq_err += e * e;
            max_abs = max_abs.max(e.abs());
        }
        let n = w.len() as f64;
        let rms_ref = (sq_ref / n).sqrt();
        let rms_err = (sq_err / n).sqrt();
        out.push(TensorErr {
            name: name.to_string(),
            o,
            i,
            rms_rel: if rms_ref > 0.0 { rms_err / rms_ref } else { 0.0 },
            max_rel: if rms_ref > 0.0 { max_abs / rms_ref } else { 0.0 },
            snr_db: if rms_err > 0.0 && rms_ref > 0.0 {
                20.0 * (rms_ref / rms_err).log10()
            } else {
                f64::INFINITY
            },
        });
    }
    Ok(out)
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum SourceFormat {
    /// already a colibrì container (`name` U8 codes + `name.qs` scales) — serve directly
    Container,
    /// block-scaled FP8 (`*_scale_inv`) — needs [`convert_snapshot`]
    Fp8,
    /// modelopt NVFP4 (`*_scale_2`) — needs [`convert_snapshot`]
    Nvfp4,
    /// no recognizable quantization marker (e.g. a plain bf16 checkpoint)
    Unknown,
}

impl SourceFormat {
    pub fn as_str(self) -> &'static str {
        match self {
            SourceFormat::Container => "container",
            SourceFormat::Fp8 => "fp8",
            SourceFormat::Nvfp4 => "nvfp4",
            SourceFormat::Unknown => "unknown",
        }
    }
    /// Whether this snapshot must be converted before the engine can load it.
    pub fn needs_convert(self) -> bool {
        matches!(self, SourceFormat::Fp8 | SourceFormat::Nvfp4)
    }
}

/// Detect a snapshot's format. Checked in precedence order: a `.qs` scale means
/// it is already our container; `_scale_2` is modelopt NVFP4 (note NVFP4's *block*
/// scales are themselves F8_E4M3, so dtype alone would misread it as FP8);
/// `_scale_inv` is block-scaled FP8.
pub fn detect_format(snap: impl AsRef<Path>) -> io::Result<SourceFormat> {
    let shards = Shards::open(snap)?;
    let mut fp8 = false;
    for t in shards.tensors() {
        if t.name.ends_with(".qs") {
            return Ok(SourceFormat::Container);
        }
        if t.name.ends_with("_scale_2") {
            return Ok(SourceFormat::Nvfp4);
        }
        // block-scaled FP8, or a per-tensor-FP8 weight (an F8 *weight*, not an
        // F8 sidecar — NVFP4's own block scales are stored as F8_E4M3).
        if t.name.ends_with("_scale_inv")
            || (matches!(t.dtype, DType::F8E4M3 | DType::F8E5M2) && t.name.ends_with(".weight"))
        {
            fp8 = true;
        }
    }
    Ok(if fp8 {
        SourceFormat::Fp8
    } else {
        SourceFormat::Unknown
    })
}

/// Result summary of a conversion run.
#[derive(Debug, Default, Clone, Copy)]
pub struct ConvertStats {
    pub shards_written: usize,
    pub tensors_quantized: usize,
    pub tensors_f32: usize,
    pub tensors_skipped: usize,
    pub bytes_out: u64,
}

/// One processed source tensor: dropped, passed through as F32, or quantized to a
/// (codes, scales) pair. Produced by [`process_one`] so the per-tensor dequant +
/// quantize (the CPU-bound work) can run in parallel and be reassembled in order.
enum TensorOut {
    Skip,
    F32(OutTensor),
    Quant(OutTensor, OutTensor),
}

/// Dequant + (re)quantize one source tensor. Pure w.r.t. shared state (reads `shards`,
/// which is `Sync`), so many run concurrently.
fn process_one(name: &str, shards: &Shards, opts: &ConvertOpts) -> io::Result<TensorOut> {
    // Container (output) name. MiniMax-M3 remaps `language_model.*`/`block_sparse_moe.*`
    // to GLM-style names and drops the vision tower + the MTP module (layer >= n_layers).
    // Reads still use the ORIGINAL source `name` (and its sidecars); only the output name
    // and classification use `out_name`.
    let out_name: String = if opts.minimax {
        match m3_container_name(name) {
            Some(n) if layer_idx(&n) < 0 || (layer_idx(&n) as usize) < opts.n_layers => n,
            _ => return Ok(TensorOut::Skip),
        }
    } else {
        name.to_string()
    };
    let f32_out = |nm: &str, shape: Vec<i64>, w: &[f32]| OutTensor {
        name: nm.to_string(),
        dtype: "F32",
        shape,
        bytes: f32_bytes(w),
    };
    match classify(&out_name, opts.n_layers, opts.keep_indexer, opts.mtp_only) {
        Kind::Skip => Ok(TensorOut::Skip),
        Kind::F32 => {
            let (mut w, shape) = dequant(shards, name)?;
            // Gemma-norm (MiniMax-M3): fold +1 into RMSNorm weights so the engine's plain
            // rmsnorm computes x*(1+w). Norms only — never the router or the bias.
            if opts.gemma_norm && out_name.ends_with("norm.weight") {
                for v in w.iter_mut() {
                    *v += 1.0;
                }
            }
            Ok(TensorOut::F32(f32_out(&out_name, shape, &w)))
        }
        kind @ (Kind::Io | Kind::X | Kind::Q) => {
            // Dequant first: the *logical* shape is authoritative (NVFP4 is stored
            // packed as [O, I/2], so the on-disk shape would lie).
            let (w, shape) = dequant(shards, name)?;
            // Only 2D weights quantize; anything else stays F32.
            if shape.len() != 2 {
                return Ok(TensorOut::F32(f32_out(&out_name, shape, &w)));
            }
            let (o, i) = (shape[0] as usize, shape[1] as usize);
            let (codes_t, scale_t) = if matches!(kind, Kind::X) {
                // Routed experts are **NVFP4** by default (4-bit block-scaled, ~2× faster
                // than e4m3 at <1% perplexity); `COLI_XFP8=1` opts into 8-bit e4m3.
                // int4 experts are no longer produced — NVFP4 supersedes them.
                if opts.xfp8 {
                    quantize_e4m3(&out_name, &w, o, i)
                } else {
                    quantize_nvfp4_out(&out_name, &w, o, i)
                }
            } else {
                // Resident weights (attention/dense/shared) at `ebits`, embeddings/lm_head
                // at `io_bits` — int8 by default.
                let bits = if matches!(kind, Kind::Io) { opts.io_bits } else { opts.ebits };
                quantize(&out_name, &w, o, i, bits)
            };
            Ok(TensorOut::Quant(codes_t, scale_t))
        }
    }
}

/// Process a shard's tensors across cores (the dequant + quantize is single-thread
/// CPU-bound otherwise — 1 of 20 cores). Contiguous index ranges preserve the input
/// order, so a routed expert's gate/up/down stay adjacent on disk. Cap with
/// `COLI_CONVERT_THREADS`.
fn process_names_parallel(
    names: &[&str],
    shards: &Shards,
    opts: &ConvertOpts,
) -> io::Result<Vec<TensorOut>> {
    let n = names.len();
    if n == 0 {
        return Ok(Vec::new());
    }
    let cap = std::env::var("COLI_CONVERT_THREADS")
        .ok()
        .and_then(|v| v.parse::<usize>().ok())
        .filter(|&t| t > 0)
        .unwrap_or_else(|| {
            std::thread::available_parallelism().map(|p| p.get()).unwrap_or(4)
        });
    let nthreads = cap.min(n);
    let chunk = n.div_ceil(nthreads);
    let mut parts: Vec<io::Result<Vec<TensorOut>>> = Vec::new();
    std::thread::scope(|scope| {
        let handles: Vec<_> = names
            .chunks(chunk)
            .map(|slice| {
                scope.spawn(move || {
                    slice.iter().map(|&nm| process_one(nm, shards, opts)).collect::<io::Result<Vec<_>>>()
                })
            })
            .collect();
        for h in handles {
            parts.push(h.join().unwrap());
        }
    });
    let mut out = Vec::with_capacity(n);
    for p in parts {
        out.extend(p?);
    }
    Ok(out)
}

/// Convert a local FP8 snapshot directory to a colibrì container directory (int8
/// resident, NVFP4/e4m3 experts). One output shard (`out-NNNNN.safetensors`) per
/// input shard; `config.json` and
/// tokenizer files are copied through. `progress` is called once per input shard
/// with `(shard_index, total_shards)`.
pub fn convert_snapshot(
    indir: impl AsRef<Path>,
    outdir: impl AsRef<Path>,
    opts: ConvertOpts,
    mut progress: impl FnMut(usize, usize, &ConvertStats),
) -> io::Result<ConvertStats> {
    let indir = indir.as_ref();
    let outdir = outdir.as_ref();
    std::fs::create_dir_all(outdir)?;

    let shards = Shards::open(indir)?;
    let nfiles = shards.num_files();

    // Group tensor names by their source shard so we stream one input file at a
    // time (bounds peak RAM to roughly one shard's worth of output).
    let mut by_file: Vec<Vec<&str>> = vec![Vec::new(); nfiles];
    for t in shards.tensors() {
        by_file[t.file_idx].push(&t.name);
    }

    let mut stats = ConvertStats::default();
    for (fi, names) in by_file.iter().enumerate() {
        // Emit all `U8` code tensors first, then the `F32` scales/passthrough. A
        // routed expert's gate/up/down are processed consecutively, so grouping
        // codes keeps those three adjacent on disk — which lets the engine's
        // `load_expert` coalesce them into ONE chunked `read_raw_shared` read
        // instead of three. Scales (`name.qs`) are read separately by name, so
        // their placement after the code block is irrelevant.
        let mut codes: Vec<OutTensor> = Vec::new();
        let mut floats: Vec<OutTensor> = Vec::new();
        // Dequant + quantize this shard's tensors across cores (order preserved).
        for out in process_names_parallel(names, &shards, &opts)? {
            match out {
                TensorOut::Skip => stats.tensors_skipped += 1,
                TensorOut::F32(t) => {
                    floats.push(t);
                    stats.tensors_f32 += 1;
                }
                TensorOut::Quant(codes_t, scale_t) => {
                    codes.push(codes_t);
                    floats.push(scale_t);
                    stats.tensors_quantized += 1;
                }
            }
        }
        if !codes.is_empty() || !floats.is_empty() {
            codes.extend(floats); // code block first, then all F32 tensors
            // `mtp_only` emits `mtp-NNNNN` so its head shards can be dropped straight
            // into an existing head-less container without colliding with its
            // `out-NNNNN` shards (the loader globs every `*.safetensors`).
            let path = if opts.mtp_only {
                outdir.join(format!("mtp-{fi:05}.safetensors"))
            } else {
                outdir.join(format!("out-{fi:05}.safetensors"))
            };
            write_shard(&path, &codes)?;
            stats.shards_written += 1;
            stats.bytes_out += codes.iter().map(|t| t.bytes.len() as u64).sum::<u64>();
        }
        progress(fi, nfiles, &stats);
    }

    // Copy config + tokenizer through so the output is a self-contained snapshot. The
    // `mtp_only` augment pass writes only head shards into an EXISTING container, whose
    // config/tokenizer are already present — leave them untouched.
    if !opts.mtp_only {
        for fname in [
            "config.json",
            "tokenizer.json",
            "tokenizer_config.json",
            "generation_config.json",
            "special_tokens_map.json",
        ] {
            let src = indir.join(fname);
            if src.exists() {
                std::fs::copy(&src, outdir.join(fname))?;
            }
        }
    }

    Ok(stats)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn m3_container_name_maps_and_drops() {
        let m = |s: &str| m3_container_name(s);
        // Prefix strip + MoE rename + expert w1/w2/w3 -> gate/down/up.
        assert_eq!(
            m("language_model.model.layers.5.block_sparse_moe.experts.7.w1.weight").as_deref(),
            Some("model.layers.5.mlp.experts.7.gate_proj.weight")
        );
        assert_eq!(
            m("language_model.model.layers.5.block_sparse_moe.experts.7.w2.weight_scale").as_deref(),
            Some("model.layers.5.mlp.experts.7.down_proj.weight_scale")
        );
        assert_eq!(
            m("language_model.model.layers.5.block_sparse_moe.experts.7.w3.weight").as_deref(),
            Some("model.layers.5.mlp.experts.7.up_proj.weight")
        );
        // Router, its bias, and the shared expert.
        assert_eq!(
            m("language_model.model.layers.3.block_sparse_moe.gate.weight").as_deref(),
            Some("model.layers.3.mlp.gate.weight")
        );
        // The router bias sits directly under the MoE block in M3 (not under `.gate.`);
        // it maps to `mlp.e_score_correction_bias`, which the loader accepts.
        assert_eq!(
            m("language_model.model.layers.3.block_sparse_moe.e_score_correction_bias").as_deref(),
            Some("model.layers.3.mlp.e_score_correction_bias")
        );
        assert_eq!(
            m("language_model.model.layers.3.block_sparse_moe.shared_experts.up_proj.weight").as_deref(),
            Some("model.layers.3.mlp.shared_experts.up_proj.weight")
        );
        // Attention (GQA) + norms + lm_head pass through after the prefix strip.
        assert_eq!(
            m("language_model.model.layers.0.self_attn.q_norm.weight").as_deref(),
            Some("model.layers.0.self_attn.q_norm.weight")
        );
        assert_eq!(m("language_model.lm_head.weight").as_deref(), Some("lm_head.weight"));
        // Dropped: vision tower, multimodal projectors, MTP/next-n module.
        assert!(m("vision_tower.vision_model.embeddings.patch_embedding.weight").is_none());
        assert!(m("multi_modal_projector.linear_1.weight").is_none());
        assert!(m("language_model.model.mtp.layers.0.weight").is_none());
    }

    /// `float_to_e4m3` must match the hardware fp8 encoder (`__nv_cvt_float_to_fp8`,
    /// __NV_SATFINITE, __NV_E4M3) — reference bytes generated on the GB10. A wrong
    /// encoder silently degrades every converted expert weight.
    #[test]
    fn float_to_e4m3_matches_hardware() {
        let cases: &[(f32, u8)] = &[
            (0.0, 0x00), (0.1, 0x1D), (0.5, 0x30), (1.0, 0x38), (1.5, 0x3C),
            (2.0, 0x40), (3.14159, 0x45), (7.0, 0x4E), (15.5, 0x58), (100.0, 0x6C),
            (448.0, 0x7E), (500.0, 0x7E), (-1.0, 0xB8), (-0.5, 0xB0), (-256.0, 0xF8),
            (0.015625, 0x08), (0.0078125, 0x04), (0.001, 0x01), (2.5, 0x42),
            (0.3, 0x2A), (0.017, 0x09), (255.9, 0x78),
        ];
        for &(x, want) in cases {
            let got = float_to_e4m3(x);
            assert_eq!(got, want, "e4m3({x}) = 0x{got:02X}, want 0x{want:02X}");
        }
    }

    /// The metric must react to a loss it is *told* is there, in the right direction
    /// and roughly the right size — otherwise a near-zero reading off the real
    /// checkpoint is unreadable: "no headroom" and "broken probe" look identical.
    #[test]
    fn quant_error_metric_tracks_bit_width() {
        // A row whose values span three orders of magnitude — the case per-row linear
        // int8 handles worst and e4m3's exponent handles well. If the probe can't see
        // a difference here it can't see one anywhere.
        let o = 4usize;
        let i = 256usize;
        let mut w = vec![0f32; o * i];
        for r in 0..o {
            for c in 0..i {
                let mag = 10f32.powi(-(c as i32 % 3));
                w[r * i + c] = mag * if (r + c) % 2 == 0 { 1.0 } else { -1.0 };
            }
        }
        let err = |bits: u32| -> f64 {
            let approx = dequantize_qtensor(&qtensor_from_f32(&w, o, i, bits));
            let (mut sr, mut se) = (0f64, 0f64);
            for (&r, &a) in w.iter().zip(&approx) {
                sr += (r as f64) * (r as f64);
                se += ((r - a) as f64) * ((r - a) as f64);
            }
            (se / sr).sqrt()
        };
        let (e16, e8) = (err(16), err(8));
        assert!(e16 < 1e-9, "f32 round trip must be exact, got {e16}");
        assert!(e8 > 1e-6, "int8 on a wide-dynamic-range row should show real error, got {e8}");
        // The metric must react in the right direction: int8 loses measurably more
        // than the exact f32 baseline it approximates.
        assert!(e8 > e16, "int8 error ({e8}) must exceed the exact f32 baseline ({e16})");
    }

    #[test]
    fn e2m1_and_ue4m3_round_to_their_real_grids() {
        // Exactly-representable values must survive untouched, or the simulator is
        // measuring its own rounding bug rather than the format.
        for &v in &E2M1_LEVELS {
            assert_eq!(e2m1_round(v), v, "e2m1 level {v} not preserved");
            assert_eq!(e2m1_round(-v), -v, "e2m1 level -{v} not preserved");
        }
        // 5.0 is an exact tie between 4.0 (code 6) and 6.0 (code 7); ties-to-even
        // picks the even code, i.e. 4.0. Asserted because a tie is where a rounding
        // implementation silently drifts from the hardware's.
        assert_eq!(e2m1_round(5.0), 4.0, "tie 4/6 must resolve to the even code");
        assert_eq!(e2m1_round(100.0), 6.0, "saturates at the max magnitude");
        assert_eq!(e2m1_round(0.2), 0.0, "below half the first step -> 0");
        // ue4m3: powers of two and the documented max are exact.
        for e in -6..=8 {
            let p = 2f32.powi(e);
            assert_eq!(ue4m3_round(p), p, "ue4m3 power of two {p} not preserved");
        }
        assert_eq!(ue4m3_round(UE4M3_MAX), UE4M3_MAX);
        assert!(ue4m3_round(1e30) <= UE4M3_MAX, "must not invent a scale past the max");
    }

    #[test]
    fn nvfp4_beats_per_row_int8_when_dynamic_range_is_wide() {
        // The whole premise of block scaling: one scale per row is hostage to that
        // row's largest value, so small values quantize to nothing. Per-16 scales
        // track the local magnitude instead. Compared against per-row int8 (what the
        // resident path ships) — if NVFP4's block scales don't help here, they have no
        // mechanism to help the experts and the measurement below means nothing.
        let (o, i) = (2usize, 512usize);
        let mut w = vec![0f32; o * i];
        for r in 0..o {
            for c in 0..i {
                // magnitude sweeps across four decades along the row
                let mag = 10f32.powi(-((c / 128) as i32));
                w[r * i + c] = mag * if c % 3 == 0 { -1.0 } else { 1.0 };
            }
        }
        let rel = |approx: &[f32]| -> f64 {
            let (mut sr, mut se) = (0f64, 0f64);
            for (&r, &a) in w.iter().zip(approx) {
                sr += (r as f64) * (r as f64);
                se += ((r - a) as f64) * ((r - a) as f64);
            }
            (se / sr).sqrt()
        };
        let int8 = rel(&dequantize_qtensor(&qtensor_from_f32(&w, o, i, 8)));
        let nvfp4 = rel(&quantize_nvfp4_sim(&w, o, i));
        assert!(nvfp4 < int8, "nvfp4 {nvfp4:.4} should beat per-row int8 {int8:.4}");
    }

    #[test]
    fn nvfp4_sim_is_not_secretly_lossless() {
        // A simulator that returns its input would make NVFP4 look perfect. e2m1 has
        // eight levels; random data must show real error.
        let (o, i) = (2usize, 256usize);
        let w: Vec<f32> = (0..o * i)
            .map(|k| ((k * 2654435761usize) % 1000) as f32 / 500.0 - 1.0)
            .collect();
        let approx = quantize_nvfp4_sim(&w, o, i);
        let diff = w.iter().zip(&approx).filter(|(a, b)| a != b).count();
        assert!(diff > w.len() / 4, "only {diff}/{} values changed — sim is a no-op?", w.len());
    }

    #[test]
    fn nvfp4_encode_decode_matches_sim() {
        // The REAL quantizer's decoded output must equal the validated reconstruction
        // (quantize_nvfp4_sim), scored at 9.4% rel-RMS on real experts. If the packed
        // container decodes to something else, the shipped model no longer matches what
        // the quality gate measured — this test ties the two together. It also exercises
        // exactly the nibble packing (low=even col), per-16 block-scale indexing, and
        // global application that the CPU (linear.rs fmt=5) and CUDA kernels decode.
        let (o, i) = (8usize, 128usize);
        let mut w = vec![0f32; o * i];
        for r in 0..o {
            for c in 0..i {
                let mag = 10f32.powi(-((c / 16) as i32 % 4)); // wide dynamic range per row
                w[r * i + c] = mag * if (r + c) % 3 == 0 { -1.0 } else { 1.0 };
            }
        }
        let (nib, bs, g) = quantize_nvfp4(&w, o, i);
        let nb = i.div_ceil(NVFP4_BLOCK);
        let rb = i.div_ceil(2);
        assert_eq!(nib.len(), o * rb);
        assert_eq!(bs.len(), o * nb);
        let mut dec = vec![0f32; o * i];
        for r in 0..o {
            for c in 0..i {
                let byte = nib[r * rb + (c >> 1)];
                let code = if c & 1 == 1 { byte >> 4 } else { byte & 0x0f } as usize;
                let bsc = colibri_core::dtype::f8e4m3_to_f32(bs[r * nb + c / NVFP4_BLOCK]);
                dec[r * i + c] = E2M1[code] * bsc * g;
            }
        }
        let sim = quantize_nvfp4_sim(&w, o, i);
        let (mut se, mut sr) = (0f64, 0f64);
        for (&a, &b) in dec.iter().zip(&sim) {
            se += ((a - b) as f64).powi(2);
            sr += (b as f64).powi(2);
        }
        let rel = (se / sr.max(1e-30)).sqrt();
        assert!(rel < 1e-4, "decode(quantize_nvfp4) vs sim rel-RMS {rel:e} too large");
    }

    #[test]
    fn default_is_8bit_resident_nvfp4_experts() {
        // Measured on unsloth/GLM-5.2-FP8, same converter, only ebits changed:
        //   4-bit resident  perplexity 48.665  top-1 32.1%
        //   8-bit resident  perplexity  6.189  top-1 57.9%
        // 8-bit resident is worth 7.9x the quality. Its throughput cost is unresolved
        // and deliberately not asserted here — see ConvertOpts' docs for why the
        // 0.52-vs-0.35 reading is confounded by the swap cliff.
        let d = ConvertOpts::default();
        assert_eq!(d.ebits, 8, "resident weights (attention/dense/shared) must default to 8-bit");
        assert_eq!(d.io_bits, 8);
        // Routed experts default to NVFP4 (4-bit block-scaled), independent of `ebits`:
        // raising the resident width must not flip experts onto the 8-bit e4m3 path.
        assert!(!d.xfp8, "routed experts must default to NVFP4, not e4m3");
        let hi = ConvertOpts { ebits: 16, ..Default::default() };
        assert!(!hi.xfp8, "raising ebits must not drag the experts up with it");
    }
    use super::*;
    use std::path::PathBuf;

    fn tmp() -> PathBuf {
        use std::sync::atomic::{AtomicU64, Ordering};
        static N: AtomicU64 = AtomicU64::new(0);
        let base = std::env::var("TMPDIR").unwrap_or_else(|_| "/tmp".into());
        let mut p = PathBuf::from(base);
        p.push(format!(
            "colibri-convert-{}-{}",
            std::process::id(),
            N.fetch_add(1, Ordering::Relaxed)
        ));
        std::fs::create_dir_all(&p).unwrap();
        p
    }

    /// Write a one-shard input safetensors file from `(name, dtype, shape, bytes)`.
    fn write_input(path: &std::path::Path, entries: &[(&str, &str, &[i64], Vec<u8>)]) {
        let mut data = Vec::new();
        let mut header = String::from("{");
        for (k, (name, dtype, shape, bytes)) in entries.iter().enumerate() {
            if k > 0 {
                header.push(',');
            }
            let off = data.len();
            data.extend_from_slice(bytes);
            let end = data.len();
            let shp: Vec<String> = shape.iter().map(|d| d.to_string()).collect();
            header.push_str(&format!(
                "\"{}\":{{\"dtype\":\"{}\",\"shape\":[{}],\"data_offsets\":[{},{}]}}",
                name,
                dtype,
                shp.join(","),
                off,
                end
            ));
        }
        header.push('}');
        let hb = header.as_bytes();
        let mut f = std::fs::File::create(path).unwrap();
        f.write_all(&(hb.len() as u64).to_le_bytes()).unwrap();
        f.write_all(hb).unwrap();
        f.write_all(&data).unwrap();
    }

    fn bf16_bytes(vals: &[f32]) -> Vec<u8> {
        vals.iter()
            .flat_map(|f| ((f.to_bits() >> 16) as u16).to_le_bytes())
            .collect()
    }

    /// f8e4m3 encode for test fixtures (block scales are stored as F8_E4M3).
    fn f8e4m3_byte(v: f32) -> u8 {
        // only the exact powers/values the tests use
        match v {
            x if x == 1.0 => 0x38,
            x if x == 2.0 => 0x40,
            x if x == 0.5 => 0x30,
            x if x == 4.0 => 0x48,
            _ => panic!("add {v} to the tiny f8e4m3 encoder"),
        }
    }

    /// Per-tensor FP8 (modelopt): F8_E4M3 weight + a scalar F32 `name_scale`.
    /// Real modelopt NVFP4 checkpoints use this for non-expert linears
    /// (observed on nvidia/Qwen3.6-35B-A3B-NVFP4), so it must not be mistaken for
    /// the 128×128 block-scaled FP8 form.
    #[test]
    fn per_tensor_fp8_dequant() {
        let indir = tmp();
        let w = "model.layers.0.self_attn.q_proj.weight";
        // 1.0, 2.0, -1.0, 0.5 in e4m3; scalar scale 3.0 → [3, 6, -3, 1.5]
        write_input(
            &indir.join("m.safetensors"),
            &[
                (w, "F8_E4M3", &[2, 2], vec![0x38, 0x40, 0xB8, 0x30]),
                (&format!("{w}_scale"), "F32", &[], 3.0f32.to_le_bytes().to_vec()),
            ],
        );
        let shards = Shards::open(&indir).unwrap();
        let (got, shape) = dequant(&shards, w).unwrap();
        assert_eq!(shape, vec![2, 2]);
        assert_eq!(got, vec![3.0, 6.0, -3.0, 1.5]);
        // and it is detected as fp8 (no _scale_inv, no _scale_2)
        assert_eq!(detect_format(&indir).unwrap(), SourceFormat::Fp8);
        std::fs::remove_dir_all(&indir).ok();
    }

    /// modelopt NVFP4: packed e2m1 nibbles × per-16-block f8 scale × small global
    /// scale. Low nibble = even element, high = odd.
    #[test]
    fn nvfp4_dequant_matches_reference_math() {
        let indir = tmp();
        // [O=1, I=4] -> packed [1,2]. codes: e2m1 idx 2(=1.0), 4(=2.0), 10(=-1.0), 5(=3.0)
        // byte0: low=2 (elem0), high=4 (elem1) -> 0x42
        // byte1: low=10 (elem2), high=5 (elem3) -> 0x5A
        let packed = vec![0x42u8, 0x5A];
        let bscale = vec![f8e4m3_byte(2.0)]; // [1, ceil(4/16)=1] = 2.0
        let gscale = 0.5f32.to_le_bytes().to_vec(); // < 1 → modelopt convention
        let w = "model.layers.3.mlp.experts.0.gate_proj.weight";
        write_input(
            &indir.join("model-00000.safetensors"),
            &[
                (w, "U8", &[1, 2], packed),
                (&format!("{w}_scale"), "F8_E4M3", &[1, 1], bscale),
                (&format!("{w}_scale_2"), "F32", &[], gscale),
            ],
        );
        let shards = Shards::open(&indir).unwrap();
        let (got, o, i) = dequant_nvfp4(&shards, w).unwrap();
        assert_eq!((o, i), (1, 4));
        // expected = e2m1 * 2.0 * 0.5 = e2m1
        assert_eq!(got, vec![1.0, 2.0, -1.0, 3.0]);
        std::fs::remove_dir_all(&indir).ok();
    }

    /// A global scale >= 1 means a compressed-tensors checkpoint (stores the
    /// reciprocal and DIVIDES). Refuse rather than silently corrupt every tensor.
    #[test]
    fn nvfp4_rejects_compressed_tensors_reciprocal() {
        let indir = tmp();
        let w = "model.layers.3.mlp.experts.0.gate_proj.weight";
        write_input(
            &indir.join("model-00000.safetensors"),
            &[
                (w, "U8", &[1, 2], vec![0x42u8, 0x5A]),
                (&format!("{w}_scale"), "F8_E4M3", &[1, 1], vec![f8e4m3_byte(1.0)]),
                // 2048.0 >= 1 → looks like the reciprocal
                (&format!("{w}_scale_2"), "F32", &[], 2048.0f32.to_le_bytes().to_vec()),
            ],
        );
        let shards = Shards::open(&indir).unwrap();
        let err = dequant_nvfp4(&shards, w).unwrap_err();
        assert!(
            err.to_string().contains("compressed-tensors"),
            "expected the modelopt-vs-compressed-tensors guard, got: {err}"
        );
        std::fs::remove_dir_all(&indir).ok();
    }

    /// The block-scale grid must be the flat per-16 layout; a swizzled/padded
    /// grid is refused instead of silently misaligning.
    #[test]
    fn nvfp4_rejects_unexpected_scale_grid() {
        let indir = tmp();
        let w = "model.layers.3.mlp.experts.0.gate_proj.weight";
        write_input(
            &indir.join("model-00000.safetensors"),
            &[
                // I = 64 → expects ceil(64/16)=4 scale columns; give 8 (padded/swizzled)
                (w, "U8", &[1, 32], vec![0x42u8; 32]),
                (
                    &format!("{w}_scale"),
                    "F8_E4M3",
                    &[1, 8],
                    vec![f8e4m3_byte(1.0); 8],
                ),
                (&format!("{w}_scale_2"), "F32", &[], 0.5f32.to_le_bytes().to_vec()),
            ],
        );
        let shards = Shards::open(&indir).unwrap();
        let err = dequant_nvfp4(&shards, w).unwrap_err();
        assert!(
            err.to_string().contains("swizzled"),
            "expected the scale-grid layout guard, got: {err}"
        );
        std::fs::remove_dir_all(&indir).ok();
    }

    #[test]
    fn detect_format_by_sidecar() {
        // fp8
        let d1 = tmp();
        write_input(
            &d1.join("m.safetensors"),
            &[
                ("a.weight", "F8_E4M3", &[2, 2], vec![0x38; 4]),
                ("a.weight_scale_inv", "F32", &[1, 1], 1.0f32.to_le_bytes().to_vec()),
            ],
        );
        assert_eq!(detect_format(&d1).unwrap(), SourceFormat::Fp8);
        assert!(SourceFormat::Fp8.needs_convert());

        // nvfp4 (note its block scales are F8_E4M3 — must not be read as fp8)
        let d2 = tmp();
        write_input(
            &d2.join("m.safetensors"),
            &[
                ("a.weight", "U8", &[1, 2], vec![0x42, 0x5A]),
                ("a.weight_scale", "F8_E4M3", &[1, 1], vec![f8e4m3_byte(1.0)]),
                ("a.weight_scale_2", "F32", &[], 0.5f32.to_le_bytes().to_vec()),
            ],
        );
        assert_eq!(detect_format(&d2).unwrap(), SourceFormat::Nvfp4);

        // already our container
        let d3 = tmp();
        write_input(
            &d3.join("m.safetensors"),
            &[
                ("a.weight", "U8", &[2], vec![0x12, 0x34]),
                ("a.weight.qs", "F32", &[1], 1.0f32.to_le_bytes().to_vec()),
            ],
        );
        assert_eq!(detect_format(&d3).unwrap(), SourceFormat::Container);
        assert!(!SourceFormat::Container.needs_convert());

        for d in [d1, d2, d3] {
            std::fs::remove_dir_all(&d).ok();
        }
    }

    /// The three code tensors of an expert must land contiguously so the engine's
    /// `read_raw_shared([gate, up, down])` coalesces them into ONE shared buffer —
    /// even when a float tensor was interleaved among the weights in the *input*.
    #[test]
    fn expert_codes_are_contiguous_for_coalesced_read() {
        let indir = tmp();
        let fp8 = vec![0x38u8, 0x38, 0x38, 0x38]; // [2,2], all 1.0
        let sc = 1.0f32.to_le_bytes().to_vec(); // [1,1] block scale
        let g = "model.layers.3.mlp.experts.0.gate_proj.weight";
        let u = "model.layers.3.mlp.experts.0.up_proj.weight";
        let d = "model.layers.3.mlp.experts.0.down_proj.weight";
        let norm = "model.layers.3.input_layernorm.weight";
        // Input deliberately interleaves a float tensor (`norm`) between the
        // expert weights to prove the reorder still groups the three codes.
        write_input(
            &indir.join("model-00000.safetensors"),
            &[
                (d, "F8_E4M3", &[2, 2], fp8.clone()),
                (&format!("{d}_scale_inv"), "F32", &[1, 1], sc.clone()),
                (norm, "BF16", &[2], bf16_bytes(&[1.0, 1.0])),
                (g, "F8_E4M3", &[2, 2], fp8.clone()),
                (&format!("{g}_scale_inv"), "F32", &[1, 1], sc.clone()),
                (u, "F8_E4M3", &[2, 2], fp8.clone()),
                (&format!("{u}_scale_inv"), "F32", &[1, 1], sc.clone()),
            ],
        );

        let outdir = tmp();
        convert_snapshot(&indir, &outdir, ConvertOpts::default(), |_, _, _| {}).unwrap();

        let out = Shards::open(&outdir).unwrap();
        // The engine's exact expert read: gate, up, down → one coalesced buffer.
        let r = out.read_raw_shared(&[g, u, d], 4).unwrap();
        assert!(
            std::sync::Arc::ptr_eq(&r[0].0, &r[1].0) && std::sync::Arc::ptr_eq(&r[1].0, &r[2].0),
            "expert gate/up/down code tensors are not contiguous — coalesced read won't fire"
        );

        std::fs::remove_dir_all(&indir).ok();
        std::fs::remove_dir_all(&outdir).ok();
    }

    #[test]
    fn classify_rules() {
        assert_eq!(classify("model.embed_tokens.weight", 78, false, false), Kind::Io);
        assert_eq!(classify("lm_head.weight", 78, false, false), Kind::Io);
        assert_eq!(classify("model.norm.weight", 78, false, false), Kind::F32);
        assert_eq!(
            classify("model.layers.3.input_layernorm.weight", 78, false, false),
            Kind::F32
        );
        assert_eq!(classify("model.layers.3.mlp.gate.weight", 78, false, false), Kind::F32); // router
        assert_eq!(
            classify("model.layers.3.mlp.gate.e_score_correction_bias", 78, false, false),
            Kind::F32
        );
        assert_eq!(
            classify("model.layers.3.mlp.experts.7.gate_proj.weight", 78, false, false),
            Kind::X
        );
        assert_eq!(
            classify("model.layers.0.mlp.gate_proj.weight", 78, false, false),
            Kind::Q // dense MLP (layer < first MoE)
        );
        assert_eq!(
            classify("model.layers.5.self_attn.kv_b_proj.weight", 78, false, false),
            Kind::Q
        );
        // dropped classes
        assert_eq!(
            classify("model.layers.3.mlp.experts.7.gate_proj.weight_scale_inv", 78, false, false),
            Kind::Skip
        );
        assert_eq!(
            classify("model.layers.0.self_attn.indexer.wk.weight", 78, false, false),
            Kind::Skip
        );

        // MTP head (layer index n_layers) is KEPT by default so the container ships
        // MTP-ready. Its own fusion inputs, attention, router, norms, and experts all
        // classify as they would for a normal layer.
        assert_eq!(classify("model.layers.78.eh_proj.weight", 78, false, false), Kind::Q);
        assert_eq!(classify("model.layers.78.enorm.weight", 78, false, false), Kind::F32);
        assert_eq!(classify("model.layers.78.hnorm.weight", 78, false, false), Kind::F32);
        assert_eq!(
            classify("model.layers.78.shared_head.norm.weight", 78, false, false),
            Kind::F32
        );
        assert_eq!(
            classify("model.layers.78.input_layernorm.weight", 78, false, false),
            Kind::F32
        );
        assert_eq!(classify("model.layers.78.mlp.gate.weight", 78, false, false), Kind::F32);
        assert_eq!(
            classify("model.layers.78.self_attn.kv_b_proj.weight", 78, false, false),
            Kind::Q
        );
        assert_eq!(
            classify("model.layers.78.mlp.shared_experts.gate_proj.weight", 78, false, false),
            Kind::Q
        );
        assert_eq!(
            classify("model.layers.78.mlp.experts.0.gate_proj.weight", 78, false, false),
            Kind::X // routed expert, streamed (NVFP4 by default)
        );
        // The duplicate lm_head inside the head is still dropped (we reuse lm_head).
        assert_eq!(
            classify("model.layers.78.shared_head.head.weight", 78, false, false),
            Kind::Skip
        );
        // Anything above the single nextn head is not part of the architecture.
        assert_eq!(classify("model.layers.79.eh_proj.weight", 78, false, false), Kind::Skip);
    }

    #[test]
    fn mtp_only_emits_just_the_head() {
        // The augment pass keeps ONLY layer n_layers; every base tensor is skipped.
        assert_eq!(classify("model.layers.78.eh_proj.weight", 78, false, true), Kind::Q);
        assert_eq!(
            classify("model.layers.78.mlp.experts.5.up_proj.weight", 78, false, true),
            Kind::X
        );
        assert_eq!(classify("model.layers.78.mlp.gate.weight", 78, false, true), Kind::F32);
        // Base-model tensors dropped on the augment pass (they already live in the container).
        assert_eq!(classify("lm_head.weight", 78, false, true), Kind::Skip);
        assert_eq!(classify("model.embed_tokens.weight", 78, false, true), Kind::Skip);
        assert_eq!(
            classify("model.layers.3.self_attn.kv_b_proj.weight", 78, false, true),
            Kind::Skip
        );
    }

    #[test]
    fn keep_indexer_retains_the_dsa_weights() {
        // With keep_idx: wk/wq_b/weights_proj quantize (Q), k_norm stays f32; and the
        // FP8 scale sidecar is still consumed. Default (false) drops them all.
        let n = "model.layers.0.self_attn.indexer";
        assert_eq!(classify(&format!("{n}.wk.weight"), 78, true, false), Kind::Q);
        assert_eq!(classify(&format!("{n}.wq_b.weight"), 78, true, false), Kind::Q);
        assert_eq!(classify(&format!("{n}.weights_proj.weight"), 78, true, false), Kind::Q);
        assert_eq!(classify(&format!("{n}.k_norm.weight"), 78, true, false), Kind::F32);
        assert_eq!(classify(&format!("{n}.k_norm.bias"), 78, true, false), Kind::F32);
        assert_eq!(classify(&format!("{n}.wk.weight_scale_inv"), 78, true, false), Kind::Skip);
        // Default path still drops them.
        assert_eq!(classify(&format!("{n}.wk.weight"), 78, false, false), Kind::Skip);
        // keep_idx does NOT resurrect MTP-head tensors.
        assert_eq!(classify("model.layers.3.eh_proj.weight", 78, true, false), Kind::Skip);
    }

    /// Write a minimal FP8 shard: one `[2,2]` F8_E4M3 weight + its `[1,1]`
    /// block scale, plus a bf16 norm, then convert and verify the container.
    #[test]
    fn convert_tiny_fp8_shard() {
        let indir = tmp();
        // weight fp8 codes: 1.0(0x38), 2.0(0x40), -1.0(0xB8), 0.5(0x30)  → [2,2]
        // block scale (1x1, since 2<=128): 2.0  → dequant = [[2,4],[-2,1]]
        let wbytes = vec![0x38u8, 0x40, 0xB8, 0x30];
        let scale = 2.0f32.to_le_bytes();
        let norm = [1.0f32, 2.0]; // bf16 stored
        let norm_bytes: Vec<u8> = norm
            .iter()
            .flat_map(|f| {
                let bits = (f.to_bits() >> 16) as u16; // f32→bf16 truncation (exact for these)
                bits.to_le_bytes()
            })
            .collect();

        // build header + data
        let name = "model.layers.0.self_attn.o_proj.weight";
        let sname = "model.layers.0.self_attn.o_proj.weight_scale_inv";
        let nname = "model.layers.0.input_layernorm.weight";
        let mut data = Vec::new();
        let o0 = data.len();
        data.extend_from_slice(&wbytes);
        let e0 = data.len();
        let o1 = data.len();
        data.extend_from_slice(&scale);
        let e1 = data.len();
        let o2 = data.len();
        data.extend_from_slice(&norm_bytes);
        let e2 = data.len();
        let header = format!(
            "{{\"{name}\":{{\"dtype\":\"F8_E4M3\",\"shape\":[2,2],\"data_offsets\":[{o0},{e0}]}},\
             \"{sname}\":{{\"dtype\":\"F32\",\"shape\":[1,1],\"data_offsets\":[{o1},{e1}]}},\
             \"{nname}\":{{\"dtype\":\"BF16\",\"shape\":[2],\"data_offsets\":[{o2},{e2}]}}}}"
        );
        let hb = header.as_bytes();
        let mut f = std::fs::File::create(indir.join("model-00000.safetensors")).unwrap();
        f.write_all(&(hb.len() as u64).to_le_bytes()).unwrap();
        f.write_all(hb).unwrap();
        f.write_all(&data).unwrap();
        drop(f);

        let outdir = tmp();
        let opts = ConvertOpts { ebits: 8, io_bits: 8, n_layers: 78, ..Default::default() };
        let stats = convert_snapshot(&indir, &outdir, opts, |_, _, _| {}).unwrap();
        assert_eq!(stats.tensors_quantized, 1); // o_proj
        assert_eq!(stats.tensors_f32, 1); // norm
        assert_eq!(stats.shards_written, 1);

        // Read the container back and check the weight round-trips through int8.
        let out = Shards::open(&outdir).unwrap();
        assert!(out.has(name)); // U8 codes
        assert!(out.has(&format!("{name}.qs"))); // scales
        assert!(out.has(nname)); // norm passthrough as F32

        // qt_load the int8 weight and check row 0: dequant target [[2,4],[-2,1]].
        // per-row int8: row0 amax=4 → s=4/127; codes round(2/s)=64, round(4/s)=127.
        let qt = crate::loader::qt_load(&out, name, 2, 2, 8).unwrap();
        assert_eq!(qt.fmt_code, 1); // int8
        assert_eq!(qt.o, 2);
        assert_eq!(qt.i, 2);
        assert!((qt.s[0] - 4.0 / 127.0).abs() < 1e-6);

        std::fs::remove_dir_all(&indir).ok();
        std::fs::remove_dir_all(&outdir).ok();
    }
}
