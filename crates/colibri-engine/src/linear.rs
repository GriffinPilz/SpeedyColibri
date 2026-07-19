//! Quantized linear layer — the exact (f32-activation) path of `matmul_qt` from
//! `c/glm.c`, plus `embed_row`.
//!
//! Computes `y[S, O] = x[S, I] @ W^T` where `W` is a [`QTensor`] `[O, I]`,
//! dequantizing weights on use. This is the `allow_idot = 0` path: activations
//! stay f32, so it is the numerically-exact reference. The int8-activation IDOT
//! fast path (`colibri-kernels`, ~1.4–2.5× faster) will be layered on top later
//! for the matmuls that tolerate it; attention projections must keep this exact
//! path (IDOT there costs ~+12% perplexity, measured — see the C comment).

use colibri_core::QTensor;

/// `y[S, O] = x[S, I] @ W^T`, exact (f32 activations). Ports `matmul` /
/// `matmul_q` / `matmul_i4` / `matmul_i2` under `matmul_qt_ex(..., allow_idot=0)`.
///
/// `x` is row-major `[S, I]`, `y` is row-major `[S, O]`. Panics if the shapes
/// don't line up with `w.o`/`w.i`.
pub fn matmul_qt(y: &mut [f32], x: &[f32], w: &QTensor, s: usize) {
    let i = w.i as usize;
    let o = w.o as usize;
    assert_eq!(x.len(), s * i, "x must be [S,I]");
    assert_eq!(y.len(), s * o, "y must be [S,O]");

    // GPU fast path for resident weights (falls through to CPU otherwise).
    #[cfg(feature = "cuda")]
    {
        if crate::gpu::try_matmul_qt(y, x, w, s) {
            return;
        }
    }

    match w.fmt_code {
        0 => {
            // f32
            let wf = &w.qf;
            for row in 0..o {
                let wr = &wf[row * i..(row + 1) * i];
                for si in 0..s {
                    let xs = &x[si * i..(si + 1) * i];
                    y[si * o + row] = dot_f32(xs, wr);
                }
            }
        }
        1 => {
            // int8: y = (Σ x_i * (f32)q_oi) * scale_o  — NEON dot on aarch64
            let q = &w.q8;
            for row in 0..o {
                let wr = &q[row * i..(row + 1) * i];
                let sc = w.s[row];
                for si in 0..s {
                    let xs = &x[si * i..(si + 1) * i];
                    y[si * o + row] = colibri_kernels::dot_i8_f32(wr, xs, i) * sc;
                }
            }
        }
        2 => {
            // int4 packed 2/byte, value = nibble - 8  — NEON dot on aarch64
            let q4 = &w.q4;
            let rb = (i + 1) / 2;
            for row in 0..o {
                let wr = &q4[row * rb..(row + 1) * rb];
                let sc = w.s[row];
                for si in 0..s {
                    let xs = &x[si * i..(si + 1) * i];
                    y[si * o + row] = colibri_kernels::dot_i4_f32(wr, xs, i) * sc;
                }
            }
        }
        3 => {
            // int2 packed 4/byte, value = field - 2
            let q2 = &w.q4;
            let rb = (i + 3) / 4;
            for row in 0..o {
                let wr = &q2[row * rb..(row + 1) * rb];
                let sc = w.s[row];
                for si in 0..s {
                    let xs = &x[si * i..(si + 1) * i];
                    let mut a = 0f32;
                    for k in 0..i {
                        let byte = wr[k >> 2];
                        let sh = (k & 3) * 2;
                        a += (((byte >> sh) & 3) as i32 - 2) as f32 * xs[k];
                    }
                    y[si * o + row] = a * sc;
                }
            }
        }
        4 => {
            // e4m3 fp8, 1 byte/weight, per-row scale (see moe::int4_to_e4m3). CPU
            // reference / fallback for the tiled GPU kernel.
            let q = &w.q4;
            for row in 0..o {
                let wr = &q[row * i..(row + 1) * i];
                let sc = w.s[row];
                for si in 0..s {
                    let xs = &x[si * i..(si + 1) * i];
                    let mut a = 0f32;
                    for k in 0..i {
                        a += e4m3_to_f32(wr[k]) * xs[k];
                    }
                    y[si * o + row] = a * sc;
                }
            }
        }
        other => panic!("matmul_qt: unknown QTensor format {other}"),
    }
}

/// Decode one e4m3 byte (OCP FP8: 1 sign, 4 exp bias-7, 3 mantissa; no infinity,
/// S.1111.111 = NaN) to f32. Used for the fp8 expert CPU reference/fallback.
fn e4m3_to_f32(b: u8) -> f32 {
    let sign = if b & 0x80 != 0 { -1.0 } else { 1.0 };
    let e = ((b >> 3) & 0x0f) as i32;
    let m = (b & 0x07) as f32;
    if e == 0 {
        sign * (m / 8.0) * 2f32.powi(-6) // subnormal
    } else {
        sign * (1.0 + m / 8.0) * 2f32.powi(e - 7)
    }
}

fn dot_f32(a: &[f32], b: &[f32]) -> f32 {
    let mut acc = 0f32;
    for (&x, &y) in a.iter().zip(b) {
        acc += x * y;
    }
    acc
}

/// `y[S, O] = x[S, I] @ W^T` with a plain f32 weight `W[O, I]`. Port of `matmul`;
/// used for the numerically-sensitive MoE router, which stays f32.
pub fn matmul_f32(y: &mut [f32], x: &[f32], w: &[f32], s: usize, i: usize, o: usize) {
    assert_eq!(w.len(), o * i);
    assert_eq!(x.len(), s * i);
    assert_eq!(y.len(), s * o);
    for row in 0..o {
        let wr = &w[row * i..(row + 1) * i];
        for si in 0..s {
            let xs = &x[si * i..(si + 1) * i];
            y[si * o + row] = dot_f32(xs, wr);
        }
    }
}

/// Dequantize one row `row` of a [`QTensor`] `[O, I]` into a fresh `Vec<f32>` of
/// length `I`. Backs the weight-absorption attention helpers (`qt_addrow`,
/// `qt_matvec_rows`). Allocation-per-call is fine for the reference path.
pub fn qt_row_dequant(w: &QTensor, row: usize) -> Vec<f32> {
    let i = w.i as usize;
    let mut out = vec![0f32; i];
    match w.fmt_code {
        0 => out.copy_from_slice(&w.qf[row * i..(row + 1) * i]),
        1 => {
            let q = &w.q8[row * i..(row + 1) * i];
            let s = w.s[row];
            for (dst, &c) in out.iter_mut().zip(q) {
                *dst = c as f32 * s;
            }
        }
        2 => {
            let rb = (i + 1) / 2;
            let q = &w.q4[row * rb..(row + 1) * rb];
            let s = w.s[row];
            let mut k = 0;
            while k < i {
                let byte = q[k >> 1];
                out[k] = ((byte & 0x0F) as i32 - 8) as f32 * s;
                if k + 1 < i {
                    out[k + 1] = ((byte >> 4) as i32 - 8) as f32 * s;
                }
                k += 2;
            }
        }
        3 => {
            let rb = (i + 3) / 4;
            let q = &w.q4[row * rb..(row + 1) * rb];
            let s = w.s[row];
            for (k, dst) in out.iter_mut().enumerate() {
                let byte = q[k >> 2];
                let sh = (k & 3) * 2;
                *dst = (((byte >> sh) & 3) as i32 - 2) as f32 * s;
            }
        }
        other => panic!("qt_row_dequant: unknown format {other}"),
    }
    out
}

/// `acc += scale * W[row, :]` (dequantized). Port of `qt_addrow`.
pub fn qt_addrow(w: &QTensor, row: usize, scale: f32, acc: &mut [f32]) {
    let r = qt_row_dequant(w, row);
    for (a, v) in acc.iter_mut().zip(&r) {
        *a += scale * v;
    }
}

/// `out[d] = W[row0 + d, :] · vec` for `d in 0..nrows`. Port of `qt_matvec_rows`.
pub fn qt_matvec_rows(w: &QTensor, row0: usize, nrows: usize, vec: &[f32], out: &mut [f32]) {
    for d in 0..nrows {
        let r = qt_row_dequant(w, row0 + d);
        out[d] = dot_f32(&r, vec);
    }
}

/// Dequantize one row `tok` of an embedding [`QTensor`] `[vocab, D]` into `out`.
/// Port of `embed_row`.
pub fn embed_row(embed: &QTensor, tok: usize, out: &mut [f32]) {
    let d = embed.i as usize;
    assert_eq!(out.len(), d, "out must be [hidden]");
    match embed.fmt_code {
        0 => out.copy_from_slice(&embed.qf[tok * d..(tok + 1) * d]),
        1 => {
            let q = &embed.q8[tok * d..(tok + 1) * d];
            let s = embed.s[tok];
            for (dst, &c) in out.iter_mut().zip(q) {
                *dst = c as f32 * s;
            }
        }
        2 => {
            let rb = (d + 1) / 2;
            let q = &embed.q4[tok * rb..(tok + 1) * rb];
            let s = embed.s[tok];
            let mut k = 0;
            while k < d {
                let byte = q[k >> 1];
                out[k] = ((byte & 0x0F) as i32 - 8) as f32 * s;
                if k + 1 < d {
                    out[k + 1] = ((byte >> 4) as i32 - 8) as f32 * s;
                }
                k += 2;
            }
        }
        3 => {
            let rb = (d + 3) / 4;
            let q = &embed.q4[tok * rb..(tok + 1) * rb];
            let s = embed.s[tok];
            for (k, dst) in out.iter_mut().enumerate() {
                let byte = q[k >> 2];
                let sh = (k & 3) * 2;
                *dst = (((byte >> sh) & 3) as i32 - 2) as f32 * s;
            }
        }
        other => panic!("embed_row: unknown format {other}"),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::quantize::qtensor_from_f32;

    fn f32_matmul(x: &[f32], w: &[f32], s: usize, i: usize, o: usize) -> Vec<f32> {
        let mut y = vec![0f32; s * o];
        for row in 0..o {
            for si in 0..s {
                let mut a = 0f32;
                for k in 0..i {
                    a += x[si * i + k] * w[row * i + k];
                }
                y[si * o + row] = a;
            }
        }
        y
    }

    #[test]
    fn f32_format_is_exact() {
        // W = [[1,2,3],[4,5,6]] (O=2,I=3); x = two rows.
        let w = vec![1.0f32, 2.0, 3.0, 4.0, 5.0, 6.0];
        let x = vec![1.0f32, 0.0, -1.0, 2.0, 2.0, 2.0];
        let qt = qtensor_from_f32(&w, 2, 3, 16);
        let mut y = vec![0f32; 2 * 2];
        matmul_qt(&mut y, &x, &qt, 2);
        assert_eq!(y, f32_matmul(&x, &w, 2, 3, 2));
    }

    #[test]
    fn int8_matmul_close_to_f32() {
        let w: Vec<f32> = (0..12).map(|k| (k as f32 - 6.0) * 0.1).collect(); // O=3,I=4
        let x = vec![0.5f32, -0.5, 1.0, -1.0];
        let qt = qtensor_from_f32(&w, 3, 4, 8);
        let mut y = vec![0f32; 3];
        matmul_qt(&mut y, &x, &qt, 1);
        let exact = f32_matmul(&x, &w, 1, 4, 3);
        for (a, b) in y.iter().zip(&exact) {
            assert!((a - b).abs() < 0.05, "int8 {a} vs f32 {b}");
        }
    }

    #[test]
    fn int4_matmul_close_to_f32() {
        let w: Vec<f32> = (0..8).map(|k| ((k % 5) as f32 - 2.0) * 0.3).collect(); // O=2,I=4
        let x = vec![1.0f32, 1.0, 1.0, 1.0];
        let qt = qtensor_from_f32(&w, 2, 4, 4);
        let mut y = vec![0f32; 2];
        matmul_qt(&mut y, &x, &qt, 1);
        let exact = f32_matmul(&x, &w, 1, 4, 2);
        for (a, b) in y.iter().zip(&exact) {
            assert!((a - b).abs() < 0.3, "int4 {a} vs f32 {b}");
        }
    }

    #[test]
    fn embed_matches_matmul_dequant() {
        // Embedding row dequant should equal a one-hot matmul against the table.
        let table: Vec<f32> = (0..12).map(|k| k as f32 * 0.25 - 1.0).collect(); // vocab=3,D=4
        let qt = qtensor_from_f32(&table, 3, 4, 8);
        let mut row = vec![0f32; 4];
        embed_row(&qt, 2, &mut row);
        // compare to dequant of the raw int8 row
        let s = qt.s[2];
        for k in 0..4 {
            let expect = qt.q8[2 * 4 + k] as f32 * s;
            assert!((row[k] - expect).abs() < 1e-6);
        }
    }
}
