//! FP8 → int4 container conversion — Rust port of `c/tools/convert_fp8_to_int4.py`
//! (the default, non-`--mtp`/`--indexer` path).
//!
//! Reads a Hugging Face GLM-5.2 snapshot whose linear weights are **block-scaled
//! FP8** (`F8_E4M3` codes + a `name.weight_scale_inv` F32 grid of 128×128 block
//! scales) and rewrites it as colibrì's own pre-quantized container: for each
//! quantized weight, a `name` `U8` tensor of packed int4/int8 codes plus a
//! `name.qs` `F32` tensor of per-row scales — exactly what [`crate::loader::qt_load`]
//! reads. Norms, the MoE router, biases, and embeddings-passthrough stay `F32`.
//!
//! What is dropped (matching the reference): the DSA lightning indexer
//! (`self_attn.indexer.*`) and the MTP head (layer `n_layers`,
//! `eh_proj`/`enorm`/`hnorm`/`shared_head`). The engine then auto-detects
//! `has_dsa = false` / `has_mtp = false` from the absent tensors, so attention is
//! exact dense MLA. Add those back later with dedicated passes if wanted.
//!
//! The quantizer math is the shared, C-exact [`crate::quantize`] code, so a
//! converted weight is byte-identical to a runtime-quantized one.

use crate::quantize::{pack_int2, pack_int4, quantize_rows};
use colibri_core::dtype::DType;
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

/// Bit-widths for each tensor class. Defaults mirror the reference converter's
/// non-`--mtp` invocation: int4 dense + experts, int8 embeddings/head.
#[derive(Debug, Clone, Copy)]
pub struct ConvertOpts {
    /// bits for resident weights (attention, dense MLP, shared expert) — `--ebits`
    pub ebits: u32,
    /// bits for embeddings + `lm_head` — `--io-bits`
    pub io_bits: u32,
    /// bits for routed (streamed) experts — `--xbits`, defaults to `ebits`
    pub xbits: u32,
    /// number of transformer layers; layer index `>= n_layers` is the MTP head
    pub n_layers: usize,
}

impl Default for ConvertOpts {
    fn default() -> Self {
        ConvertOpts { ebits: 4, io_bits: 8, xbits: 4, n_layers: 78 }
    }
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
    /// routed expert weight → `xbits`
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

/// Tensor classification — faithful port of `classify(name, n_layers)` on the
/// default path (no `keep_mtp`/`keep_idx`).
fn classify(name: &str, n_layers: usize) -> Kind {
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
    if li >= 0 && li as usize >= n_layers {
        return Kind::Skip; // MTP head lives at layer index n_layers
    }
    for k in [
        "indexer",
        "indexers_proj",
        "eh_proj",
        "enorm",
        "hnorm",
        "shared_head",
    ] {
        if name.contains(k) {
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
            // Block-scaled FP8: W[o,i] = fp8(o,i) * scale_inv[o/128, i/128].
            // The scale sidecar is the weight name (…proj.weight) with `_scale_inv`
            // appended → …proj.weight_scale_inv (NOT a further `.weight_scale_inv`).
            let (o, i) = two_dims(&t.shape, name)?;
            let sname = format!("{name}_scale_inv");
            let st = shards.find(&sname).ok_or_else(|| {
                io::Error::new(
                    io::ErrorKind::NotFound,
                    format!(
                        "FP8 weight {name} has neither {sname} (128x128 block scales) \
                         nor {name}_scale (per-tensor scale)"
                    ),
                )
            })?;
            let (nbo, nbi) = two_dims(&st.shape, &sname)?;
            debug_assert_eq!(nbo, o.div_ceil(BLOCK));
            debug_assert_eq!(nbi, i.div_ceil(BLOCK));

            // fp8 codes → f32 (byte-per-element), then scale by block.
            let mut w = vec![0f32; o * i];
            shards.read_f32(name, &mut w)?; // convert_to_f32 decodes F8_E4M3/E5M2
            let mut scale = vec![0f32; nbo * nbi];
            shards.read_f32(&sname, &mut scale)?;

            for oo in 0..o {
                let srow = (oo / BLOCK) * nbi;
                let wrow = &mut w[oo * i..(oo + 1) * i];
                for (ii, wv) in wrow.iter_mut().enumerate() {
                    *wv *= scale[srow + ii / BLOCK];
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

/// Quantize a 2D weight to the container form: packed `U8` codes + `F32` per-row
/// scales. `bits` selects int2/int4/int8 exactly as the reference does.
fn quantize(name: &str, w: &[f32], o: usize, i: usize, bits: u32) -> (OutTensor, OutTensor) {
    let (codes, scale) = if bits <= 2 {
        pack_int2(w, o, i, bits)
    } else if bits <= 4 {
        pack_int4(w, o, i, bits)
    } else {
        let (q, s) = quantize_rows(w, o, i, bits);
        (q.iter().map(|&x| x as u8).collect(), s)
    };
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

/// Convert a local FP8 snapshot directory to a colibrì int4 container directory.
/// One output shard (`out-NNNNN.safetensors`) per input shard; `config.json` and
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
        for &name in names {
            match classify(name, opts.n_layers) {
                Kind::Skip => {
                    stats.tensors_skipped += 1;
                }
                Kind::F32 => {
                    let (w, shape) = dequant(&shards, name)?;
                    floats.push(OutTensor {
                        name: name.to_string(),
                        dtype: "F32",
                        shape,
                        bytes: f32_bytes(&w),
                    });
                    stats.tensors_f32 += 1;
                }
                kind @ (Kind::Io | Kind::X | Kind::Q) => {
                    // Dequant first: the *logical* shape is authoritative (NVFP4 is
                    // stored packed as [O, I/2], so the on-disk shape would lie).
                    let (w, shape) = dequant(&shards, name)?;
                    // Only 2D weights quantize; anything else stays F32 (matches
                    // the reference's `if w.ndim != 2` guard).
                    if shape.len() != 2 {
                        floats.push(OutTensor {
                            name: name.to_string(),
                            dtype: "F32",
                            shape,
                            bytes: f32_bytes(&w),
                        });
                        stats.tensors_f32 += 1;
                        continue;
                    }
                    let bits = match kind {
                        Kind::Io => opts.io_bits,
                        Kind::X => opts.xbits,
                        _ => opts.ebits,
                    };
                    let (o, i) = (shape[0] as usize, shape[1] as usize);
                    let (codes_t, scale_t) = quantize(name, &w, o, i, bits);
                    codes.push(codes_t);
                    floats.push(scale_t);
                    stats.tensors_quantized += 1;
                }
            }
        }
        if !codes.is_empty() || !floats.is_empty() {
            codes.extend(floats); // code block first, then all F32 tensors
            let path = outdir.join(format!("out-{fi:05}.safetensors"));
            write_shard(&path, &codes)?;
            stats.shards_written += 1;
            stats.bytes_out += codes.iter().map(|t| t.bytes.len() as u64).sum::<u64>();
        }
        progress(fi, nfiles, &stats);
    }

    // Copy config + tokenizer through so the output is a self-contained snapshot.
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

#[cfg(test)]
mod tests {
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
        assert_eq!(classify("model.embed_tokens.weight", 78), Kind::Io);
        assert_eq!(classify("lm_head.weight", 78), Kind::Io);
        assert_eq!(classify("model.norm.weight", 78), Kind::F32);
        assert_eq!(
            classify("model.layers.3.input_layernorm.weight", 78),
            Kind::F32
        );
        assert_eq!(classify("model.layers.3.mlp.gate.weight", 78), Kind::F32); // router
        assert_eq!(
            classify("model.layers.3.mlp.gate.e_score_correction_bias", 78),
            Kind::F32
        );
        assert_eq!(
            classify("model.layers.3.mlp.experts.7.gate_proj.weight", 78),
            Kind::X
        );
        assert_eq!(
            classify("model.layers.0.mlp.gate_proj.weight", 78),
            Kind::Q // dense MLP (layer < first MoE)
        );
        assert_eq!(
            classify("model.layers.5.self_attn.kv_b_proj.weight", 78),
            Kind::Q
        );
        // dropped classes
        assert_eq!(
            classify("model.layers.3.mlp.experts.7.gate_proj.weight_scale_inv", 78),
            Kind::Skip
        );
        assert_eq!(
            classify("model.layers.0.self_attn.indexer.wk.weight", 78),
            Kind::Skip
        );
        assert_eq!(classify("model.layers.78.eh_proj.weight", 78), Kind::Skip);
        assert_eq!(
            classify("model.layers.78.mlp.experts.0.gate_proj.weight", 78),
            Kind::Skip // MTP layer
        );
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
        let opts = ConvertOpts { ebits: 4, io_bits: 8, xbits: 4, n_layers: 78 };
        let stats = convert_snapshot(&indir, &outdir, opts, |_, _, _| {}).unwrap();
        assert_eq!(stats.tensors_quantized, 1); // o_proj
        assert_eq!(stats.tensors_f32, 1); // norm
        assert_eq!(stats.shards_written, 1);

        // Read the container back and check the weight round-trips through int4.
        let out = Shards::open(&outdir).unwrap();
        assert!(out.has(name)); // U8 codes
        assert!(out.has(&format!("{name}.qs"))); // scales
        assert!(out.has(nname)); // norm passthrough as F32

        // qt_load the int4 weight and dequantize row 0: dequant target [[2,4],[-2,1]].
        // per-row int4: row0 amax=4 → s=4/7; codes round(2/s)=4, round(4/s)=7.
        let qt = crate::loader::qt_load(&out, name, 2, 2, 4).unwrap();
        assert_eq!(qt.fmt_code, 2); // int4
        assert_eq!(qt.o, 2);
        assert_eq!(qt.i, 2);
        assert!((qt.s[0] - 4.0 / 7.0).abs() < 1e-6);

        std::fs::remove_dir_all(&indir).ok();
        std::fs::remove_dir_all(&outdir).ok();
    }
}
