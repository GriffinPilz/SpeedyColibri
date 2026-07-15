//! Quantized-tensor representation — port of the `QT` struct and `qt_bytes`
//! from `c/glm.c`.
//!
//! A weight tensor `[O, I]` is stored in one of four formats. int4 is what keeps
//! the dense part resident in ~10 GB (0.5 byte/param); the router weights stay
//! f32 because they are numerically sensitive.

/// Storage format of a quantized tensor. The discriminants match the C `fmt`
/// field (0 F32, 1 INT8, 2 INT4 packed 2/byte, 3 INT2 packed 4/byte).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum QFormat {
    F32 = 0,
    Int8 = 1,
    Int4 = 2,
    Int2 = 3,
}

impl QFormat {
    pub fn from_code(fmt: i32) -> Option<QFormat> {
        match fmt {
            0 => Some(QFormat::F32),
            1 => Some(QFormat::Int8),
            2 => Some(QFormat::Int4),
            3 => Some(QFormat::Int2),
            _ => None,
        }
    }

    /// Bits per weight in this format.
    pub fn bits(self) -> i32 {
        match self {
            QFormat::F32 => 32,
            QFormat::Int8 => 8,
            QFormat::Int4 => 4,
            QFormat::Int2 => 2,
        }
    }
}

/// A quantized tensor of logical shape `[O, I]` (rows × cols).
///
/// Exactly one of the payload buffers is populated per `fmt`:
///   - `F32`  → `qf`
///   - `Int8` → `q8` (1 byte/param) + per-row scale `s`
///   - `Int4` → `q4` (2 values/byte, packed) + per-row scale `s`
///   - `Int2` → `q4` (4 values/byte, packed) + per-row scale `s`
///
/// The heavy `unsafe`/SIMD matmul kernels that consume this live in
/// `colibri-kernels`; this type is just the container.
#[derive(Debug, Clone, Default)]
pub struct QTensor {
    pub fmt_code: i32,
    pub qf: Vec<f32>,
    pub q8: Vec<i8>,
    pub q4: Vec<u8>,
    /// per-row scales (length `O`), empty for `F32`
    pub s: Vec<f32>,
    /// rows (output dim)
    pub o: i32,
    /// cols (input dim)
    pub i: i32,
}

impl QTensor {
    pub fn format(&self) -> Option<QFormat> {
        QFormat::from_code(self.fmt_code)
    }

    /// Resident byte count — port of `qt_bytes`.
    pub fn bytes(&self) -> i64 {
        let n = self.o as i64 * self.i as i64;
        match self.fmt_code {
            0 => n * 4,
            1 => n + self.o as i64 * 4,
            3 => self.o as i64 * ((self.i as i64 + 3) / 4) + self.o as i64 * 4,
            _ => self.o as i64 * ((self.i as i64 + 1) / 2) + self.o as i64 * 4, // int4
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn qt(fmt: i32, o: i32, i: i32) -> QTensor {
        QTensor {
            fmt_code: fmt,
            o,
            i,
            ..Default::default()
        }
    }

    #[test]
    fn byte_counts_match_c() {
        // f32: O*I*4
        assert_eq!(qt(0, 10, 20).bytes(), 10 * 20 * 4);
        // int8: O*I + O*4
        assert_eq!(qt(1, 10, 20).bytes(), 10 * 20 + 10 * 4);
        // int4: O*ceil(I/2) + O*4
        assert_eq!(qt(2, 10, 21).bytes(), 10 * 11 + 10 * 4);
        // int2: O*ceil(I/4) + O*4
        assert_eq!(qt(3, 10, 21).bytes(), 10 * 6 + 10 * 4);
    }

    #[test]
    fn format_bits() {
        assert_eq!(QFormat::from_code(2), Some(QFormat::Int4));
        assert_eq!(QFormat::Int4.bits(), 4);
        assert_eq!(QFormat::from_code(9), None);
    }
}
