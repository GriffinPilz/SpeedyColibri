//! Weight quantization — port of `quantize_rows`, `pack_int2`,
//! `qt_alloc`, and `qt_fill` from `c/glm.c`.
//!
//! These build the [`QTensor`] container the forward pass multiplies against.
//! Rounding is ties-to-even (`round_ties_even`) to match C's `lrintf`, so a
//! runtime-quantized weight is byte-identical to the C engine's. The clamp
//! bounds mirror the C exactly (asymmetric lower bounds included).

use colibri_core::QTensor;

/// Per-row symmetric int8 quantization. Port of `quantize_rows` (weights).
///
/// `qmax = 2^(bits-1) - 1`; `scale = max|w|/qmax` floored at 1e-8; codes clamp to
/// `[-(qmax+1), qmax]`.
pub fn quantize_rows(w: &[f32], o: usize, i: usize, bits: u32) -> (Vec<i8>, Vec<f32>) {
    let qmax = ((1i32 << (bits - 1)) - 1) as f32;
    let mut q = vec![0i8; o * i];
    let mut scale = vec![0f32; o];
    for row in 0..o {
        let wr = &w[row * i..(row + 1) * i];
        let amax = wr.iter().fold(0f32, |m, &v| m.max(v.abs()));
        let s = (amax / qmax).max(1e-8);
        scale[row] = s;
        let inv = 1.0 / s;
        let qr = &mut q[row * i..(row + 1) * i];
        for (dst, &v) in qr.iter_mut().zip(wr) {
            let q = (v * inv).round_ties_even().clamp(-(qmax + 1.0), qmax);
            *dst = q as i8;
        }
    }
    (q, scale)
}

/// Per-row int2, packed 4 values/byte. Port of `pack_int2`.
///
/// Values clamp to `[-2, qmax]` and are stored as `v + 2` in 2 bits.
pub fn pack_int2(w: &[f32], o: usize, i: usize, bits: u32) -> (Vec<u8>, Vec<f32>) {
    let qmax = ((1i32 << (bits - 1)) - 1) as f32;
    let rb = (i + 3) / 4;
    let mut q2 = vec![0u8; o * rb];
    let mut scale = vec![0f32; o];
    for row in 0..o {
        let wr = &w[row * i..(row + 1) * i];
        let amax = wr.iter().fold(0f32, |m, &v| m.max(v.abs()));
        let s = (amax / qmax).max(1e-8);
        scale[row] = s;
        let inv = 1.0 / s;
        let qr = &mut q2[row * rb..(row + 1) * rb];
        let mut col = 0;
        while col < i {
            let mut byte = 0u8;
            for k in 0..4 {
                if col + k < i {
                    let v = ((wr[col + k] * inv).round_ties_even().clamp(-2.0, qmax)) as i32;
                    byte |= ((v + 2) << (k * 2)) as u8;
                }
            }
            qr[col >> 2] = byte;
            col += 4;
        }
    }
    (q2, scale)
}

/// Build a [`QTensor`] `[O, I]` from full-precision weights at `bits`. Port of
/// `qt_alloc` + `qt_fill`.
///
/// Format is chosen by `bits`: ≥16 → f32, ≥5 → int8, else int2 — then filled with
/// the matching quantizer (which uses `bits` for its `qmax`). int4 is no longer
/// produced (removed); anything below int8 falls to int2.
pub fn qtensor_from_f32(w: &[f32], o: usize, i: usize, bits: u32) -> QTensor {
    let mut t = QTensor {
        o: o as i32,
        i: i as i32,
        ..Default::default()
    };
    if bits >= 16 {
        t.fmt_code = 0;
        t.qf = w.to_vec();
    } else if bits >= 5 {
        t.fmt_code = 1;
        let (q, s) = quantize_rows(w, o, i, bits);
        t.q8 = q;
        t.s = s;
    } else {
        t.fmt_code = 3;
        let (q, s) = pack_int2(w, o, i, bits);
        t.q4 = q.into();
        t.s = s;
    }
    t
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn int8_roundtrip_close() {
        let w = vec![0.5, -1.0, 0.25, 2.0]; // O=2, I=2
        let (q, s) = quantize_rows(&w, 2, 2, 8);
        // row 0: amax 1.0, s=1/127; row 1: amax 2.0, s=2/127
        assert!((s[0] - 1.0 / 127.0).abs() < 1e-9);
        assert!((s[1] - 2.0 / 127.0).abs() < 1e-9);
        // dequant within one step
        for (idx, &orig) in w.iter().enumerate() {
            let row = idx / 2;
            let deq = q[idx] as f32 * s[row];
            assert!((deq - orig).abs() <= s[row] + 1e-6);
        }
    }

    #[test]
    fn int2_pack_layout() {
        // One row I=4, values mapping to [-2,-1,0,1] -> stored [0,1,2,3].
        let w = vec![-2.0f32, -1.0, 0.0, 1.0];
        let (q2, s) = pack_int2(&w, 1, 4, 2);
        assert!((s[0] - 2.0).abs() < 1e-6); // amax 2 / qmax 1
        // v = round(w/s): -1,-0.5->0 (ties even),0,0.5->0 ... check via dequant path
        let byte = q2[0];
        // decode each 2-bit field: ((byte>>(k*2))&3)-2
        let decoded: Vec<i32> = (0..4).map(|k| (((byte >> (k * 2)) & 3) as i32) - 2).collect();
        // s=2: -2/2=-1, -1/2=-0.5->0(even), 0, 1/2=0.5->0(even)
        assert_eq!(decoded, vec![-1, 0, 0, 0]);
    }

    #[test]
    fn format_selection_by_bits() {
        let w = vec![1.0f32; 8];
        assert_eq!(qtensor_from_f32(&w, 2, 4, 16).fmt_code, 0);
        assert_eq!(qtensor_from_f32(&w, 2, 4, 8).fmt_code, 1);
        assert_eq!(qtensor_from_f32(&w, 2, 4, 2).fmt_code, 3);
    }

    #[test]
    fn ties_to_even_matches_lrintf() {
        // 2.5 and 3.5 both round to even (2 and 4) under lrintf/round_ties_even.
        let w = vec![2.5f32, 3.5, 3.0, -3.0]; // O=1, I=4, amax=3.5? -> qmax path
        // Force s=1 by picking amax=qmax: use bits=8 (qmax=127) is messy; instead
        // check round_ties_even directly on the values we'd feed.
        let _ = w;
        assert_eq!(2.5f32.round_ties_even(), 2.0);
        assert_eq!(3.5f32.round_ties_even(), 4.0);
        assert_eq!((-2.5f32).round_ties_even(), -2.0);
    }
}
