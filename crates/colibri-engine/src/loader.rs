//! Weight loading from safetensors shards — port of `qt_from_disk` / `qt_load` /
//! `ld` from `c/glm.c`.
//!
//! A tensor is loaded one of two ways:
//!   - **pre-quantized container:** if `name.qs` exists, `name` holds the raw
//!     int8/int2 codes (safetensors `U8`) and `name.qs` holds the per-row
//!     f32 scales — read directly, no requantization. The format is inferred from
//!     the byte count (`O*I` → int8, else int2).
//!   - **full tensor:** otherwise `name` is a full f32/bf16 tensor that gets
//!     runtime-quantized to `bits` (the tiny oracle / full-precision path).

use crate::quantize::qtensor_from_f32;
use colibri_core::QTensor;
use colibri_safetensors::Shards;
use std::io;

/// Load a `[O, I]` weight tensor as a [`QTensor`] at `bits`. Port of
/// `qt_from_disk` + `qt_load`.
pub fn qt_load(shards: &Shards, name: &str, o: usize, i: usize, bits: u32) -> io::Result<QTensor> {
    let qs = format!("{name}.qs");
    if shards.has(&qs) {
        // Pre-quantized container: raw codes + separate f32 scales.
        let nb = shards.nbytes(name);
        if nb < 0 {
            return Err(missing(name));
        }
        let nb = nb as usize;
        // int8 (`O*I` bytes) vs int2 (`O*ceil(I/4)`), inferred from the byte count.
        // int4 containers are no longer produced.
        let fmt = if nb == o * i { 1 } else { 3 };
        let mut t = QTensor {
            fmt_code: fmt,
            o: o as i32,
            i: i as i32,
            ..Default::default()
        };
        let mut raw = vec![0u8; nb];
        shards.read_raw(name, &mut raw)?;
        if fmt == 1 {
            // reinterpret the code bytes as signed int8
            t.q8 = raw.into_iter().map(|b| b as i8).collect();
        } else {
            t.q4 = raw.into();
        }
        // scales: O per-row f32 in `name.qs`
        let mut s = vec![0f32; o];
        shards.read_f32(&qs, &mut s)?;
        t.s = s;
        Ok(t)
    } else {
        // Full tensor -> runtime quantize to `bits`.
        let numel = shards.numel(name);
        if numel < 0 {
            return Err(missing(name));
        }
        let mut tmp = vec![0f32; (o * i).max(numel as usize)];
        shards.read_f32(name, &mut tmp)?;
        tmp.truncate(o * i);
        Ok(qtensor_from_f32(&tmp, o, i, bits))
    }
}

/// Load a 1D resident f32 tensor (norms / biases). Port of `ld`.
pub fn ld(shards: &Shards, name: &str) -> io::Result<Vec<f32>> {
    let n = shards.numel(name);
    if n < 0 {
        return Err(missing(name));
    }
    let mut v = vec![0f32; n as usize];
    shards.read_f32(name, &mut v)?;
    Ok(v)
}

fn missing(name: &str) -> io::Error {
    io::Error::new(io::ErrorKind::NotFound, format!("missing tensor: {name}"))
}

/// Concatenate several `[Oⱼ, I]` weights row-wise into one `[ΣOⱼ, I]` QTensor, so a set
/// of projections that share the same input `x` can run as ONE matmul instead of N. Used
/// to fuse MiniMax-M3's q/k/v projections: at S=1 decode each was a separate synchronized
/// GPU dispatch (measured ~25% of decode across q/k/v/o × 60 layers), and one fused
/// matmul cuts the q/k/v three into one. All parts must share `i` and `fmt_code`; supports
/// the resident formats f32 (0) and int8 (1) — the only ones projections ship as.
pub fn concat_rows(parts: &[&QTensor]) -> QTensor {
    assert!(!parts.is_empty(), "concat_rows: no parts");
    let (i, fmt) = (parts[0].i, parts[0].fmt_code);
    assert!(
        parts.iter().all(|p| p.i == i && p.fmt_code == fmt),
        "concat_rows: all parts must share i and fmt_code"
    );
    let mut out = parts[0].clone();
    out.o = parts.iter().map(|p| p.o).sum();
    match fmt {
        0 => {
            out.qf = parts.iter().flat_map(|p| p.qf.iter().copied()).collect();
        }
        1 => {
            out.q8 = parts.iter().flat_map(|p| p.q8.iter().copied()).collect();
            out.s = parts.iter().flat_map(|p| p.s.iter().copied()).collect();
        }
        _ => panic!("concat_rows: unsupported fmt_code {fmt} (projections ship f32/int8)"),
    }
    out.gpu_eligible = parts.iter().all(|p| p.gpu_eligible);
    out
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::linear::matmul_qt;
    use std::fs::File;
    use std::io::Write;
    use std::path::PathBuf;

    // Minimal safetensors writer for tests: tensors given as (name, dtype, bytes).
    fn write_st(dir: &std::path::Path, tensors: &[(&str, &str, Vec<u8>)]) {
        // Build header JSON with sequential offsets.
        let mut header = String::from("{");
        let mut off = 0usize;
        let mut first = true;
        for (name, dtype, bytes) in tensors {
            if !first {
                header.push(',');
            }
            first = false;
            let shape = bytes.len(); // 1D shape for simplicity (numel = byte/elem)
            let elem = match *dtype {
                "F32" => 4,
                _ => 1,
            };
            header.push_str(&format!(
                "\"{}\":{{\"dtype\":\"{}\",\"shape\":[{}],\"data_offsets\":[{},{}]}}",
                name,
                dtype,
                shape / elem,
                off,
                off + bytes.len()
            ));
            off += bytes.len();
        }
        header.push('}');
        let hb = header.as_bytes();
        let path = dir.join("model.safetensors");
        let mut f = File::create(&path).unwrap();
        f.write_all(&(hb.len() as u64).to_le_bytes()).unwrap();
        f.write_all(hb).unwrap();
        for (_, _, bytes) in tensors {
            f.write_all(bytes).unwrap();
        }
    }

    fn temp_dir() -> PathBuf {
        use std::sync::atomic::{AtomicU64, Ordering};
        static N: AtomicU64 = AtomicU64::new(0);
        let base = std::env::var("TMPDIR").unwrap_or_else(|_| "/tmp".into());
        let mut p = PathBuf::from(base);
        p.push(format!(
            "colibri-loader-{}-{}",
            std::process::id(),
            N.fetch_add(1, Ordering::Relaxed)
        ));
        std::fs::create_dir_all(&p).unwrap();
        p
    }

    fn f32_bytes(v: &[f32]) -> Vec<u8> {
        v.iter().flat_map(|x| x.to_le_bytes()).collect()
    }

    #[test]
    fn loads_full_tensor_runtime_quantized() {
        let dir = temp_dir();
        // O=2, I=3 full f32 tensor "w"
        let w = vec![0.1f32, -0.2, 0.3, 0.4, -0.5, 0.6];
        write_st(&dir, &[("w", "F32", f32_bytes(&w))]);
        let shards = Shards::open(&dir).unwrap();
        let qt = qt_load(&shards, "w", 2, 3, 8).unwrap();
        assert_eq!(qt.fmt_code, 1); // int8
        // applying it should be close to the exact f32 matmul
        let x = vec![1.0f32, 1.0, 1.0];
        let mut y = vec![0f32; 2];
        matmul_qt(&mut y, &x, &qt, 1);
        let exact0 = w[0] + w[1] + w[2];
        let exact1 = w[3] + w[4] + w[5];
        assert!((y[0] - exact0).abs() < 0.02);
        assert!((y[1] - exact1).abs() < 0.02);
        std::fs::remove_dir_all(&dir).ok();
    }

    #[test]
    fn loads_prequantized_container() {
        let dir = temp_dir();
        // O=2, I=4 int8 container "wq" + scales "wq.qs".
        // codes row0: [1,2,3,4] scale 0.5 ; row1: [-1,-2,-3,-4] scale 0.25
        let codes: Vec<u8> = [1i8, 2, 3, 4, -1, -2, -3, -4]
            .iter()
            .map(|&c| c as u8)
            .collect();
        let scales = vec![0.5f32, 0.25];
        write_st(
            &dir,
            &[
                ("wq", "I8", codes.clone()),
                ("wq.qs", "F32", f32_bytes(&scales)),
            ],
        );
        let shards = Shards::open(&dir).unwrap();
        let qt = qt_load(&shards, "wq", 2, 4, 8).unwrap();
        assert_eq!(qt.fmt_code, 1);
        assert_eq!(qt.q8, vec![1i8, 2, 3, 4, -1, -2, -3, -4]);
        assert_eq!(qt.s, vec![0.5, 0.25]);
        // y = (Σ x_i * code) * scale, with x all ones
        let x = vec![1.0f32; 4];
        let mut y = vec![0f32; 2];
        matmul_qt(&mut y, &x, &qt, 1);
        assert!((y[0] - (1 + 2 + 3 + 4) as f32 * 0.5).abs() < 1e-6);
        assert!((y[1] - (-1 - 2 - 3 - 4) as f32 * 0.25).abs() < 1e-6);
        std::fs::remove_dir_all(&dir).ok();
    }

    #[test]
    fn missing_tensor_is_error() {
        let dir = temp_dir();
        write_st(&dir, &[("w", "F32", f32_bytes(&[1.0, 2.0]))]);
        let shards = Shards::open(&dir).unwrap();
        assert!(qt_load(&shards, "nope", 1, 2, 8).is_err());
        std::fs::remove_dir_all(&dir).ok();
    }
}
