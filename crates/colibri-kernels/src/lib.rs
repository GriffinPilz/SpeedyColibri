//! Quantization and integer-dot kernels — port of the kernel routines in
//! `c/glm.c` (the `QScratch`/`idot`/`qt_alloc`/dequant paths).
//!
//! Status: this crate currently provides **portable scalar reference**
//! implementations that are numerically correct. The AVX2 (`maddubs`) and
//! NEON SIMD paths, and the *shape-dependent* rounding that makes the C engine's
//! quantized output byte-exact, are a tracked follow-up (see PORTING.md). The
//! reference kernels below are the oracle those SIMD paths must match.
//!
//! Formats follow [`colibri_core::quant::QFormat`]: int8 is one byte/param with a
//! per-row f32 scale; int4/int2 pack 2/4 values per byte with a per-row scale.

/// Symmetric per-row int8 quantization (Q8_0 style): find the row's max-abs,
/// scale so it maps to 127, and round to nearest. Returns `(codes, scale)` where
/// `x[i] ≈ codes[i] as f32 * scale`.
pub fn quantize_row_i8(x: &[f32]) -> (Vec<i8>, f32) {
    let amax = x.iter().fold(0.0f32, |m, &v| m.max(v.abs()));
    let scale = if amax > 0.0 { amax / 127.0 } else { 0.0 };
    let inv = if scale > 0.0 { 1.0 / scale } else { 0.0 };
    let codes = x
        .iter()
        .map(|&v| {
            let q = (v * inv).round();
            q.clamp(-127.0, 127.0) as i8
        })
        .collect();
    (codes, scale)
}

/// Integer dot product of two int8 vectors, accumulated in i32.
///
/// This is the reference for the AVX2 `maddubs`-based `idot`. Note the C SIMD
/// path uses an unsigned·signed product with an offset; that reassociation is
/// what changes low-bit rounding on wide shapes. This scalar version is the
/// exact-arithmetic oracle.
pub fn dot_i8(a: &[i8], b: &[i8]) -> i32 {
    debug_assert_eq!(a.len(), b.len());
    a.iter().zip(b).map(|(&x, &y)| x as i32 * y as i32).sum()
}

/// int8 matrix·vector: `out[o] = scale_w[o] * scale_x * Σ_i W[o,i] * xq[i]`.
///
/// `w` is row-major `[o_dim, i_dim]` int8 with per-row scales `w_scale`; `xq` is
/// the int8-quantized activation with scalar scale `x_scale`.
pub fn matvec_i8(
    w: &[i8],
    w_scale: &[f32],
    xq: &[i8],
    x_scale: f32,
    o_dim: usize,
    i_dim: usize,
    out: &mut [f32],
) {
    debug_assert_eq!(w.len(), o_dim * i_dim);
    debug_assert_eq!(xq.len(), i_dim);
    for o in 0..o_dim {
        let row = &w[o * i_dim..(o + 1) * i_dim];
        let acc = dot_i8(row, xq);
        out[o] = acc as f32 * w_scale[o] * x_scale;
    }
}

/// Unpack a packed int4 nibble stream (2 values/byte, low nibble first) into
/// signed values in `[-8, 7]`. Port of the int4 dequant unpack.
pub fn unpack_i4(packed: &[u8], n: usize, out: &mut [i8]) {
    debug_assert!(out.len() >= n);
    for k in 0..n {
        let byte = packed[k / 2];
        let nib = if k % 2 == 0 { byte & 0x0F } else { byte >> 4 };
        // signed 4-bit: values 8..15 are negative
        out[k] = if nib >= 8 {
            nib as i8 - 16
        } else {
            nib as i8
        };
    }
}

/// f32 dot product — the fallback path used when int quant measured slower
/// (int4 single-row on some shapes stays f32 in the C engine).
pub fn dot_f32(a: &[f32], b: &[f32]) -> f32 {
    debug_assert_eq!(a.len(), b.len());
    a.iter().zip(b).map(|(&x, &y)| x * y).sum()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn quant_dequant_roundtrip_is_close() {
        let x = vec![0.5, -1.0, 0.25, 2.0, -2.0, 0.0];
        let (codes, scale) = quantize_row_i8(&x);
        for (orig, &c) in x.iter().zip(&codes) {
            let deq = c as f32 * scale;
            assert!((deq - orig).abs() <= scale, "‖{deq}-{orig}‖ > {scale}");
        }
    }

    #[test]
    fn dot_matches_manual() {
        let a = [1i8, 2, 3, -4];
        let b = [5i8, -6, 7, 8];
        assert_eq!(dot_i8(&a, &b), 1 * 5 + 2 * -6 + 3 * 7 + -4 * 8);
    }

    #[test]
    fn matvec_shape() {
        // W = [[1,2],[3,4]] int8, scales 1.0; x = [1,1] scale 1.0 -> [3, 7]
        let w = [1i8, 2, 3, 4];
        let ws = [1.0f32, 1.0];
        let xq = [1i8, 1];
        let mut out = [0f32; 2];
        matvec_i8(&w, &ws, &xq, 1.0, 2, 2, &mut out);
        assert_eq!(out, [3.0, 7.0]);
    }

    #[test]
    fn i4_unpack_signed() {
        // byte 0x0F -> low nibble 15 -> -1 ; high nibble 0 -> 0
        // byte 0x87 -> low nibble 7 -> 7 ; high nibble 8 -> -8
        let packed = [0x0F, 0x87];
        let mut out = [0i8; 4];
        unpack_i4(&packed, 4, &mut out);
        assert_eq!(out, [-1, 0, 7, -8]);
    }
}
