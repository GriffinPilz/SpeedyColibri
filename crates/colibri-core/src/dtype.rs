//! On-disk element types and their conversion to `f32`.
//!
//! Port of the dtype handling in `c/st.h` (`st_dtype_code`, `bf16_to_f32`,
//! `f16_to_f32`). The engine always materializes weights as `f32` on read,
//! except for the already-quantized container tensors which are read raw
//! (`DType::U8`).

/// safetensors element type, as recognized by the engine.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum DType {
    /// bfloat16
    Bf16,
    /// IEEE float16
    F16,
    /// float32
    F32,
    /// raw bytes — quantized container codes: int8/int2, or nvfp4 nibbles / e4m3
    /// (safetensors `U8`/`I8`)
    U8,
    /// float8 e4m3 (`fn` finite variant) — block-scaled FP8 weights. Read-only:
    /// used by the FP8/NVFP4 converter, never on the inference path.
    F8E4M3,
    /// float8 e5m2 — the other FP8 weight variant (has inf/nan). Converter-only.
    F8E5M2,
    /// signed 64-bit integer. NOT a weight type — it exists because DeepSeek-V4 ships an
    /// index table (`ffn.gate.tid2eid`, `[vocab, top_k]`, a token-id -> expert-id lookup
    /// used by its hash-routing layers) inside the weight shards. Without an arm here
    /// `DType::parse` returns None and the reader rejects the ENTIRE checkpoint over three
    /// tensors, reported as "unsupported dtype: I64" with no hint which tensor caused it.
    /// Never converted to f32; only its size is ever needed, to skip past it.
    I64,
}

impl DType {
    /// Parse a safetensors dtype string. `None` for anything unsupported, where
    /// the C code would `exit(1)`.
    pub fn parse(s: &str) -> Option<DType> {
        match s {
            "BF16" => Some(DType::Bf16),
            "F16" => Some(DType::F16),
            "F32" => Some(DType::F32),
            // `F8_E8M0` is the MX block-scale type: one byte holding a raw exponent, which
            // the MXFP4 path consumes verbatim exactly as it does K3's `U8` scales. It maps
            // to the raw-bytes variant for the same reason `I8` does — the reader hands the
            // bytes through and the format tag decides how to read them.
            //
            // Without this arm `DType::parse` returns None and EVERY scale tensor in a
            // DeepSeek-V4 checkpoint is rejected at load, which reads as a corrupt file
            // rather than an unsupported dtype.
            "U8" | "I8" | "F8_E8M0" => Some(DType::U8),
            "F8_E4M3" | "F8_E4M3FN" => Some(DType::F8E4M3),
            "F8_E5M2" => Some(DType::F8E5M2),
            "I64" => Some(DType::I64),
            _ => None,
        }
    }

    /// The safetensors dtype string (inverse of [`parse`], round-tripping through
    /// the reader). `U8` is emitted for the raw quantized-container variant (the
    /// reader parses both `U8` and `I8` to `U8` and reads bytes verbatim, so the
    /// distinction is immaterial). Used by the shard writer.
    pub fn safetensors_str(self) -> &'static str {
        match self {
            DType::Bf16 => "BF16",
            DType::F16 => "F16",
            DType::F32 => "F32",
            DType::U8 => "U8",
            DType::F8E4M3 => "F8_E4M3",
            DType::F8E5M2 => "F8_E5M2",
            DType::I64 => "I64",
        }
    }

    /// Bytes per element on disk.
    pub fn elem_size(self) -> usize {
        match self {
            DType::F32 => 4,
            DType::Bf16 | DType::F16 => 2,
            DType::U8 | DType::F8E4M3 | DType::F8E5M2 => 1,
            DType::I64 => 8,
        }
    }

    /// Numeric code matching the C `st_tensor.dtype` field (0=BF16,1=F16,2=F32,3=U8/I8).
    /// FP8 codes (4,5) extend past the C enum — FP8 is a converter-only input dtype
    /// that never reaches the C-compatible inference path.
    pub fn code(self) -> i32 {
        match self {
            DType::Bf16 => 0,
            DType::F16 => 1,
            DType::F32 => 2,
            DType::U8 => 3,
            DType::F8E4M3 => 4,
            DType::F8E5M2 => 5,
            // Past the C enum on purpose: an I64 index table must never be handed to the
            // C-compatible inference path, and giving it a code that path knows would let
            // it try.
            DType::I64 => 6,
        }
    }
}

/// Decode a float8 **e4m3** (`e4m3fn`: finite, no infinities — the ML/safetensors
/// `F8_E4M3` variant) bit pattern to f32. Layout: 1 sign / 4 exponent (bias 7) /
/// 3 mantissa; `S.1111.111` is the sole NaN and max finite magnitude is 448.
/// Every representable value is exact in f32.
#[inline]
pub fn f8e4m3_to_f32(b: u8) -> f32 {
    let sign = if b & 0x80 != 0 { -1.0f32 } else { 1.0 };
    let exp = ((b >> 3) & 0x0F) as i32;
    let man = (b & 0x07) as f32;
    if exp == 0 {
        // subnormal / zero: 2^(1-7) * (man/8) = man * 2^-9
        sign * man * (1.0 / 512.0)
    } else if exp == 0x0F && man == 7.0 {
        f32::NAN
    } else {
        // normal: 2^(exp-7) * (1 + man/8)
        sign * (1.0 + man / 8.0) * 2.0f32.powi(exp - 7)
    }
}

/// Decode an OCP **E8M0** block-scale byte to f32: an unsigned, mantissa-less
/// power of two, `2^(b - 127)`. `0xFF` is the sole NaN.
///
/// This is the MX scale encoding — the shared per-32 exponent of an MXFP4 block (and
/// the per-32 scale of the MX-FP8 checkpoints the M3 converter already reads). Unlike
/// e4m3 it carries no mantissa, so every scale is exactly a power of two; that is what
/// makes an MXFP4 -> NVFP4 transcode lossless when the target block scale is constrained
/// to a power of two, and lossy (measured 6.4% rel-RMS on real Kimi-K3 experts) when it
/// is chosen as `blockmax/6` in the usual way.
///
/// `2^-127` is the smallest normal here, which is within f32 range, so no value
/// underflows; `powi` is exact for every power of two in range.
#[inline]
pub fn f8e8m0_to_f32(b: u8) -> f32 {
    if b == 0xFF {
        return f32::NAN;
    }
    2.0f32.powi(b as i32 - 127)
}

/// Decode a float8 **e5m2** bit pattern to f32. Layout: 1 sign / 5 exponent
/// (bias 15) / 2 mantissa; `S.11111.00` is ±inf and `S.11111.xx` (xx≠0) is NaN.
#[inline]
pub fn f8e5m2_to_f32(b: u8) -> f32 {
    let sign = if b & 0x80 != 0 { -1.0f32 } else { 1.0 };
    let exp = ((b >> 2) & 0x1F) as i32;
    let man = (b & 0x03) as f32;
    if exp == 0 {
        // subnormal / zero: 2^(1-15) * (man/4)
        sign * man * (1.0 / 4.0) * 2.0f32.powi(-14)
    } else if exp == 0x1F {
        if man == 0.0 {
            sign * f32::INFINITY
        } else {
            f32::NAN
        }
    } else {
        // normal: 2^(exp-15) * (1 + man/4)
        sign * (1.0 + man / 4.0) * 2.0f32.powi(exp - 15)
    }
}

/// Reinterpret a bf16 bit pattern as f32 (zero-extend into the high half).
#[inline]
pub fn bf16_to_f32(h: u16) -> f32 {
    f32::from_bits((h as u32) << 16)
}

/// Round an f32 to the nearest bf16 bit pattern, ties-to-even — the inverse of
/// [`bf16_to_f32`], and **exact** for any value that came from a bf16 in the first place.
///
/// That exactness is the whole point: a checkpoint stored in BF16 loses nothing by being
/// kept in BF16 rather than widened to F32, so the round trip
/// `bf16_to_f32(f32_to_bf16(v)) == v` is what lets the BF16 IO tier claim bit-identical
/// logits while reading half the bytes. Callers that depend on that should *verify* the
/// round trip rather than assume it — an f32 source with real low mantissa bits will not
/// survive, and silently rounding it would be a quality regression disguised as a
/// storage win.
///
/// NaN is preserved as a NaN (the payload may be truncated, never promoted to infinity).
#[inline]
pub fn f32_to_bf16(v: f32) -> u16 {
    let bits = v.to_bits();
    if v.is_nan() {
        // Keep it a NaN: truncation alone could clear every mantissa bit and turn this
        // into an infinity.
        return ((bits >> 16) as u16) | 0x0040;
    }
    // Round-to-nearest-even on the 16 bits being discarded.
    let rounded = bits + 0x7fff + ((bits >> 16) & 1);
    (rounded >> 16) as u16
}

/// Convert an IEEE float16 bit pattern to f32. Handles subnormals, inf, and nan
/// exactly as `f16_to_f32` in `c/st.h`.
#[inline]
pub fn f16_to_f32(h: u16) -> f32 {
    let sign = ((h & 0x8000) as u32) << 16;
    let mut exp = ((h >> 10) & 0x1F) as u32;
    let mut man = (h & 0x3FF) as u32;
    let u = if exp == 0 {
        if man == 0 {
            sign // ±0
        } else {
            // subnormal: normalize
            exp = 127 - 15 + 1;
            while man & 0x400 == 0 {
                man <<= 1;
                exp -= 1;
            }
            man &= 0x3FF;
            sign | (exp << 23) | (man << 13)
        }
    } else if exp == 0x1F {
        sign | 0x7F80_0000 | (man << 13) // inf / nan
    } else {
        sign | ((exp - 15 + 127) << 23) | (man << 13)
    };
    f32::from_bits(u)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn bf16_roundtrip_simple() {
        // 1.0f32 = 0x3F800000; bf16 keeps the top 16 bits: 0x3F80.
        assert_eq!(bf16_to_f32(0x3F80), 1.0);
        assert_eq!(bf16_to_f32(0x0000), 0.0);
        assert_eq!(bf16_to_f32(0xBF80), -1.0);
    }

    #[test]
    fn f32_to_bf16_round_trips_every_bf16() {
        // Exhaustive over all 65536 bit patterns: widening to f32 and narrowing back must
        // be the identity. This is the property the BF16 IO tier rests on — a checkpoint
        // stored in bf16 loses nothing by staying in bf16 — so it is checked in full
        // rather than sampled.
        for h in 0u32..=0xFFFF {
            let h = h as u16;
            let v = bf16_to_f32(h);
            if v.is_nan() {
                assert!(bf16_to_f32(f32_to_bf16(v)).is_nan(), "NaN {h:#06x} lost");
            } else {
                assert_eq!(f32_to_bf16(v), h, "round trip failed for {h:#06x}");
            }
        }
    }

    #[test]
    fn f32_to_bf16_rounds_to_nearest_even() {
        // Exactly halfway between two bf16s (low half = 0x8000) goes to the EVEN
        // neighbour. Truncation instead of rounding would bias every converted weight
        // toward zero — small per weight, systematic across 311M of them.
        let up = f32::from_bits((0x3F80u32 << 16) | 0x8000); // 1.0 + half a ulp
        assert_eq!(f32_to_bf16(up), 0x3F80); // ties to even -> stays 1.0
        let dn = f32::from_bits((0x3F81u32 << 16) | 0x8000); // next bf16 + half a ulp
        assert_eq!(f32_to_bf16(dn), 0x3F82); // ties to even -> rounds up
        assert_eq!(f32_to_bf16(f32::INFINITY), 0x7F80);
        assert_eq!(f32_to_bf16(-0.0), 0x8000);
    }

    #[test]
    fn f32_to_bf16_refuses_nothing_but_reports_loss() {
        // A value with real low mantissa bits does NOT survive, which is exactly why the
        // converter verifies the round trip per tensor instead of assuming it.
        let v = f32::from_bits((0x3F80u32 << 16) | 0x1234);
        assert_ne!(bf16_to_f32(f32_to_bf16(v)), v);
    }

    #[test]
    fn f16_values() {
        assert_eq!(f16_to_f32(0x3C00), 1.0); // 1.0
        assert_eq!(f16_to_f32(0x0000), 0.0);
        assert_eq!(f16_to_f32(0xC000), -2.0);
        assert!(f16_to_f32(0x7C00).is_infinite()); // +inf
        assert!(f16_to_f32(0x7E00).is_nan()); // nan
                                              // smallest positive subnormal 2^-24
        assert!((f16_to_f32(0x0001) - 2f32.powi(-24)).abs() < 1e-30);
    }

    #[test]
    fn dtype_parse() {
        assert_eq!(DType::parse("BF16"), Some(DType::Bf16));
        assert_eq!(DType::parse("I8"), Some(DType::U8));
        assert_eq!(DType::parse("F8_E4M3"), Some(DType::F8E4M3));
        assert_eq!(DType::parse("F8_E5M2"), Some(DType::F8E5M2));
        assert_eq!(DType::parse("garbage"), None);
        assert_eq!(DType::F32.elem_size(), 4);
        assert_eq!(DType::F8E4M3.elem_size(), 1);
    }

    #[test]
    fn f8e4m3_known_values() {
        // sign(1) exp(4, bias 7) mantissa(3)
        assert_eq!(f8e4m3_to_f32(0x00), 0.0); // +0
        assert_eq!(f8e4m3_to_f32(0x38), 1.0); // exp7 man0 -> 1.0
        assert_eq!(f8e4m3_to_f32(0x40), 2.0); // exp8 man0 -> 2.0
        assert_eq!(f8e4m3_to_f32(0x30), 0.5); // exp6 man0 -> 0.5
        assert_eq!(f8e4m3_to_f32(0x3C), 1.5); // exp7 man4 -> 1+0.5
        assert_eq!(f8e4m3_to_f32(0xB8), -1.0); // sign + exp7 man0
        assert_eq!(f8e4m3_to_f32(0x7E), 448.0); // exp15 man6 -> max finite
        assert!(f8e4m3_to_f32(0x7F).is_nan()); // S.1111.111
                                               // smallest positive subnormal: man1, exp0 -> 2^-9
        assert_eq!(f8e4m3_to_f32(0x01), 2f32.powi(-9));
        assert_eq!(f8e4m3_to_f32(0x80), 0.0); // -0 reads as 0.0 == -0.0
    }

    #[test]
    fn f8e8m0_known_values() {
        // Unsigned, mantissa-less: the whole byte is a biased exponent, value 2^(b-127).
        assert_eq!(f8e8m0_to_f32(127), 1.0);
        assert_eq!(f8e8m0_to_f32(128), 2.0);
        assert_eq!(f8e8m0_to_f32(126), 0.5);
        assert_eq!(f8e8m0_to_f32(0), 2f32.powi(-127));
        assert_eq!(f8e8m0_to_f32(254), 2f32.powi(127));
        assert!(f8e8m0_to_f32(0xFF).is_nan());
        // Every value is exactly a power of two — no mantissa to round. This is the
        // property the lossless MXFP4 -> NVFP4 transcode depends on.
        //
        // Checked over b >= 1 only: b == 0 is 2^-127, and f32's smallest NORMAL is
        // 2^-126, so that one value is stored subnormal (0.5 x 2^-126) and carries a
        // set mantissa bit despite being an exact power of two. It is still exact and
        // still round-trips — real K3 expert scales sit in 2^-16..2^-5, nowhere near it.
        for b in 1u8..=254 {
            let v = f8e8m0_to_f32(b);
            assert!(v > 0.0 && v.is_finite());
            assert_eq!(
                v.to_bits() & 0x007F_FFFF,
                0,
                "byte {b} is not a power of two"
            );
        }
        assert!(!f8e8m0_to_f32(0).is_normal() && f8e8m0_to_f32(0) > 0.0);
        // Independent construction: halving from 1.0 must reproduce the decode exactly.
        let mut expect = 1.0f32;
        for b in (1..=127u8).rev() {
            assert_eq!(f8e8m0_to_f32(b), expect, "byte {b}");
            expect /= 2.0;
        }
        // The real Kimi-K3 expert scales measured on disk span 2^-16..2^-5.
        assert_eq!(f8e8m0_to_f32(127 - 16), 2f32.powi(-16));
        assert_eq!(f8e8m0_to_f32(127 - 5), 2f32.powi(-5));
    }

    #[test]
    fn f8e5m2_known_values() {
        // sign(1) exp(5, bias 15) mantissa(2)
        assert_eq!(f8e5m2_to_f32(0x00), 0.0);
        assert_eq!(f8e5m2_to_f32(0x3C), 1.0); // exp15 man0 -> 1.0
        assert_eq!(f8e5m2_to_f32(0x40), 2.0); // exp16 man0 -> 2.0
        assert_eq!(f8e5m2_to_f32(0x3E), 1.5); // exp15 man2 -> 1+0.5
        assert!(f8e5m2_to_f32(0x7C).is_infinite() && f8e5m2_to_f32(0x7C) > 0.0);
        assert!(f8e5m2_to_f32(0x7D).is_nan());
        assert_eq!(f8e5m2_to_f32(0x01), 2f32.powi(-16)); // smallest subnormal
    }
}
