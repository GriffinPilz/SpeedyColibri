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
    /// raw bytes — quantized int4/int8 container (safetensors `U8`/`I8`)
    U8,
    /// float8 e4m3 (`fn` finite variant) — block-scaled FP8 weights. Read-only:
    /// used by the FP8→int4 converter, never on the inference path.
    F8E4M3,
    /// float8 e5m2 — the other FP8 weight variant (has inf/nan). Converter-only.
    F8E5M2,
}

impl DType {
    /// Parse a safetensors dtype string. `None` for anything unsupported, where
    /// the C code would `exit(1)`.
    pub fn parse(s: &str) -> Option<DType> {
        match s {
            "BF16" => Some(DType::Bf16),
            "F16" => Some(DType::F16),
            "F32" => Some(DType::F32),
            "U8" | "I8" => Some(DType::U8),
            "F8_E4M3" | "F8_E4M3FN" => Some(DType::F8E4M3),
            "F8_E5M2" => Some(DType::F8E5M2),
            _ => None,
        }
    }

    /// Bytes per element on disk.
    pub fn elem_size(self) -> usize {
        match self {
            DType::F32 => 4,
            DType::Bf16 | DType::F16 => 2,
            DType::U8 | DType::F8E4M3 | DType::F8E5M2 => 1,
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
