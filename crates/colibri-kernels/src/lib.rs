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

/// Symmetric per-row int8 activation quantization — port of `qrow_i8` in
/// `c/glm.c` (the IDOT activation path).
///
/// `scale = max|x| / 127` (floored at 1e-12), `codes[i] = round_ties_even(x[i] /
/// scale)`. Ties-to-even matches C's `lrintf`. Returns `(codes, scale)` with
/// `x[i] ≈ codes[i] as f32 * scale`.
pub fn quantize_row_i8(x: &[f32]) -> (Vec<i8>, f32) {
    let amax = x.iter().fold(0.0f32, |m, &v| m.max(v.abs()));
    let scale = (amax / 127.0).max(1e-12);
    let inv = 1.0 / scale;
    let codes = x
        .iter()
        .map(|&v| (v * inv).round_ties_even().clamp(-128.0, 127.0) as i8)
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

// ---- exact int·f32 dots (weights dequantized, f32 activations) -------------
//
// These back the exact `matmul_qt` path (attention projections, expert FFN,
// lm_head). On aarch64 (the DGX Spark Grace CPU) they use NEON with the same
// two-accumulator / `vfmaq` / `vaddvq` structure as the C engine's `matmul_i4`
// / `matmul_q`; elsewhere they fall back to the scalar reference. The f32 path
// stays scalar (see `dot_f32`) so it remains byte-exact with the C f32 kernel.

/// `Σ (nibble_i − 8) · x[i]` over `n` int4 weights packed 2/byte (low nibble
/// first). NEON on aarch64, scalar otherwise.
#[inline]
pub fn dot_i4_f32(w4: &[u8], x: &[f32], n: usize) -> f32 {
    debug_assert!(w4.len() >= n.div_ceil(2));
    debug_assert!(x.len() >= n);
    #[cfg(target_arch = "aarch64")]
    // SAFETY: NEON is baseline on aarch64; bounds checked via the asserts above.
    unsafe {
        dot_i4_f32_neon(w4, x, n)
    }
    #[cfg(not(target_arch = "aarch64"))]
    {
        dot_i4_f32_scalar(w4, x, n)
    }
}

/// Scalar reference for [`dot_i4_f32`] — also the oracle for the NEON path.
pub fn dot_i4_f32_scalar(w4: &[u8], x: &[f32], n: usize) -> f32 {
    let mut a = 0f32;
    let mut i = 0;
    while i + 1 < n {
        let b = w4[i >> 1];
        a += ((b & 0x0F) as i32 - 8) as f32 * x[i];
        a += ((b >> 4) as i32 - 8) as f32 * x[i + 1];
        i += 2;
    }
    if i < n {
        let b = w4[i >> 1];
        a += ((b & 0x0F) as i32 - 8) as f32 * x[i];
    }
    a
}

/// `Σ w[i] · x[i]` over `n` int8 weights (as f32). NEON on aarch64, scalar else.
#[inline]
pub fn dot_i8_f32(w8: &[i8], x: &[f32], n: usize) -> f32 {
    debug_assert!(w8.len() >= n);
    debug_assert!(x.len() >= n);
    #[cfg(target_arch = "aarch64")]
    // SAFETY: NEON is baseline on aarch64; bounds checked via the asserts above.
    unsafe {
        dot_i8_f32_neon(w8, x, n)
    }
    #[cfg(not(target_arch = "aarch64"))]
    {
        dot_i8_f32_scalar(w8, x, n)
    }
}

/// Scalar reference for [`dot_i8_f32`].
pub fn dot_i8_f32_scalar(w8: &[i8], x: &[f32], n: usize) -> f32 {
    let mut a = 0f32;
    for i in 0..n {
        a += x[i] * w8[i] as f32;
    }
    a
}

#[cfg(target_arch = "aarch64")]
#[target_feature(enable = "neon")]
unsafe fn dot_i4_f32_neon(w4: &[u8], x: &[f32], n: usize) -> f32 {
    use std::arch::aarch64::*;
    let m4 = vdup_n_u8(0x0F);
    let b8 = vdup_n_s8(8);
    let mut ac0 = vdupq_n_f32(0.0);
    let mut ac1 = vdupq_n_f32(0.0);
    let mut i = 0usize;
    while i + 16 <= n {
        // 8 bytes = 16 nibbles; de-interleave low/high nibbles into index order.
        let by = vld1_u8(w4.as_ptr().add(i >> 1));
        let z = vzip_u8(vand_u8(by, m4), vshr_n_u8::<4>(by));
        let w0 = vmovl_s8(vsub_s8(vreinterpret_s8_u8(z.0), b8)); // nibbles 0..8, value−8
        let w1 = vmovl_s8(vsub_s8(vreinterpret_s8_u8(z.1), b8)); // nibbles 8..16
        ac0 = vfmaq_f32(ac0, vld1q_f32(x.as_ptr().add(i)), vcvtq_f32_s32(vmovl_s16(vget_low_s16(w0))));
        ac1 = vfmaq_f32(ac1, vld1q_f32(x.as_ptr().add(i + 4)), vcvtq_f32_s32(vmovl_s16(vget_high_s16(w0))));
        ac0 = vfmaq_f32(ac0, vld1q_f32(x.as_ptr().add(i + 8)), vcvtq_f32_s32(vmovl_s16(vget_low_s16(w1))));
        ac1 = vfmaq_f32(ac1, vld1q_f32(x.as_ptr().add(i + 12)), vcvtq_f32_s32(vmovl_s16(vget_high_s16(w1))));
        i += 16;
    }
    let mut a = vaddvq_f32(vaddq_f32(ac0, ac1));
    while i + 1 < n {
        let b = *w4.get_unchecked(i >> 1);
        a += ((b & 0x0F) as i32 - 8) as f32 * *x.get_unchecked(i);
        a += ((b >> 4) as i32 - 8) as f32 * *x.get_unchecked(i + 1);
        i += 2;
    }
    if i < n {
        let b = *w4.get_unchecked(i >> 1);
        a += ((b & 0x0F) as i32 - 8) as f32 * *x.get_unchecked(i);
    }
    a
}

#[cfg(target_arch = "aarch64")]
#[target_feature(enable = "neon")]
unsafe fn dot_i8_f32_neon(w8: &[i8], x: &[f32], n: usize) -> f32 {
    use std::arch::aarch64::*;
    let mut ac0 = vdupq_n_f32(0.0);
    let mut ac1 = vdupq_n_f32(0.0);
    let mut i = 0usize;
    while i + 8 <= n {
        let w16 = vmovl_s8(vld1_s8(w8.as_ptr().add(i)));
        ac0 = vfmaq_f32(ac0, vld1q_f32(x.as_ptr().add(i)), vcvtq_f32_s32(vmovl_s16(vget_low_s16(w16))));
        ac1 = vfmaq_f32(ac1, vld1q_f32(x.as_ptr().add(i + 4)), vcvtq_f32_s32(vmovl_s16(vget_high_s16(w16))));
        i += 8;
    }
    let mut a = vaddvq_f32(vaddq_f32(ac0, ac1));
    while i < n {
        a += *x.get_unchecked(i) * *w8.get_unchecked(i) as f32;
        i += 1;
    }
    a
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

    // deterministic pseudo-random f32 / packed-int4 for kernel tests
    fn xs(n: usize) -> Vec<f32> {
        (0..n).map(|k| (((k * 37 + 11) % 97) as f32 - 48.0) * 0.02).collect()
    }
    fn packed_i4(n: usize) -> Vec<u8> {
        (0..n.div_ceil(2)).map(|k| ((k * 53 + 7) % 251) as u8).collect()
    }

    #[test]
    fn dot_i4_f32_neon_matches_scalar() {
        // exercise multiples of 16, non-multiples, and odd n (tail paths)
        for &n in &[0usize, 1, 3, 15, 16, 17, 31, 64, 100, 6144, 6145] {
            let w = packed_i4(n);
            let x = xs(n);
            let simd = dot_i4_f32(&w, &x, n);
            let scal = dot_i4_f32_scalar(&w, &x, n);
            let tol = 1e-3 * (1.0 + scal.abs());
            assert!((simd - scal).abs() <= tol, "n={n}: simd {simd} vs scalar {scal}");
        }
    }

    #[test]
    fn dot_i8_f32_neon_matches_scalar() {
        for &n in &[0usize, 1, 7, 8, 9, 63, 100, 6144, 6145] {
            let w: Vec<i8> = (0..n).map(|k| ((k * 29 + 3) % 255) as i8).collect();
            let x = xs(n);
            let simd = dot_i8_f32(&w, &x, n);
            let scal = dot_i8_f32_scalar(&w, &x, n);
            let tol = 1e-3 * (1.0 + scal.abs());
            assert!((simd - scal).abs() <= tol, "n={n}: simd {simd} vs scalar {scal}");
        }
    }

    // Micro-benchmark: `cargo test -p colibri-kernels --release -- --ignored --nocapture bench_dot_i4`
    #[test]
    #[ignore]
    fn bench_dot_i4() {
        let n = 6144usize; // GLM-5.2 hidden
        let w = packed_i4(n);
        let x = xs(n);
        let iters = 2_000_000u64;
        let t0 = std::time::Instant::now();
        let mut acc = 0f32;
        for _ in 0..iters {
            acc += dot_i4_f32(&w, &x, n);
        }
        let simd_s = t0.elapsed().as_secs_f64();
        let t1 = std::time::Instant::now();
        let mut acc2 = 0f32;
        for _ in 0..iters {
            acc2 += dot_i4_f32_scalar(&w, &x, n);
        }
        let scal_s = t1.elapsed().as_secs_f64();
        let flops = iters as f64 * n as f64 * 2.0;
        println!(
            "dot_i4_f32 n={n} x{iters}: SIMD {:.3}s ({:.1} GFLOP/s) | scalar {:.3}s ({:.1} GFLOP/s) | {:.2}x  (chk {} {})",
            simd_s, flops / simd_s / 1e9, scal_s, flops / scal_s / 1e9, scal_s / simd_s, acc, acc2
        );
    }
}
