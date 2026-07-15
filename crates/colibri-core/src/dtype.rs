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
            _ => None,
        }
    }

    /// Bytes per element on disk.
    pub fn elem_size(self) -> usize {
        match self {
            DType::F32 => 4,
            DType::Bf16 | DType::F16 => 2,
            DType::U8 => 1,
        }
    }

    /// Numeric code matching the C `st_tensor.dtype` field (0=BF16,1=F16,2=F32,3=U8/I8).
    pub fn code(self) -> i32 {
        match self {
            DType::Bf16 => 0,
            DType::F16 => 1,
            DType::F32 => 2,
            DType::U8 => 3,
        }
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
        assert_eq!(DType::parse("garbage"), None);
        assert_eq!(DType::F32.elem_size(), 4);
    }
}
