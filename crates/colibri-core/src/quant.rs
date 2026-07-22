//! Quantized-tensor representation — port of the `QT` struct and `qt_bytes`
//! from `c/glm.c`.
//!
//! A weight tensor `[O, I]` is stored in one of several formats. int8 keeps the
//! dense part resident (~1 byte/param); the router weights stay f32 because they
//! are numerically sensitive. e4m3(4)/nvfp4(5) experts are handled by raw
//! `fmt_code` checks elsewhere, not by this enum.

/// Storage format of a quantized tensor. The discriminants match the C `fmt`
/// field (0 F32, 1 INT8, 3 INT2 packed 4/byte).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum QFormat {
    F32 = 0,
    Int8 = 1,
    Int2 = 3,
}

impl QFormat {
    pub fn from_code(fmt: i32) -> Option<QFormat> {
        match fmt {
            0 => Some(QFormat::F32),
            1 => Some(QFormat::Int8),
            3 => Some(QFormat::Int2),
            _ => None,
        }
    }

    /// Bits per weight in this format.
    pub fn bits(self) -> i32 {
        match self {
            QFormat::F32 => 32,
            QFormat::Int8 => 8,
            QFormat::Int2 => 2,
        }
    }
}

/// A read buffer whose allocation is **recycled through a global pool** instead
/// of being freed. Streaming decode loads ~180 experts/token, each an ~18 MB
/// buffer; with plain allocation every load pays a fresh `mmap` plus a zero-fill
/// page fault per 4 KiB page (~14 ms/expert — 8× the cost of the read itself,
/// measured warm on the GB10). Recycling keeps the pages faulted-in, so a
/// steady-state load is just the `pread`. Only buffers ≥ 1 MiB are pooled (expert
/// payloads), and the pool is bounded, so small/one-off reads are unaffected.
pub struct SharedBuf {
    data: Vec<u8>,
}

/// Recycled allocations, largest-capacity-agnostic FIFO. Bounded by
/// [`pool_max`]; entries are ~uniform in practice (one expert span).
static BUF_POOL: std::sync::Mutex<Vec<Vec<u8>>> = std::sync::Mutex::new(Vec::new());

/// Don't pool buffers smaller than this — tiny reads don't pay the fault cost.
const POOL_MIN_BYTES: usize = 1 << 20;

/// Max pooled entries (`COLI_BUF_POOL`, default 32 ≈ 600 MB of 18 MB experts;
/// `0` disables recycling).
fn pool_max() -> usize {
    static N: std::sync::OnceLock<usize> = std::sync::OnceLock::new();
    *N.get_or_init(|| {
        std::env::var("COLI_BUF_POOL").ok().and_then(|s| s.parse().ok()).unwrap_or(32)
    })
}

impl SharedBuf {
    /// A buffer of exactly `len` bytes: recycled from the pool when one with
    /// enough capacity is available (contents are stale, caller overwrites),
    /// freshly zero-allocated otherwise.
    pub fn with_len(len: usize) -> SharedBuf {
        if len >= POOL_MIN_BYTES {
            let mut pool = BUF_POOL.lock().unwrap();
            if let Some(i) = pool.iter().position(|v| v.capacity() >= len) {
                let mut v = pool.swap_remove(i);
                drop(pool);
                // Stale bytes are fine: previously written, about to be overwritten.
                v.truncate(len);
                v.resize(len, 0);
                return SharedBuf { data: v };
            }
        }
        SharedBuf { data: vec![0u8; len] }
    }

    #[inline]
    pub fn as_mut_slice(&mut self) -> &mut [u8] {
        &mut self.data
    }
}

impl Drop for SharedBuf {
    fn drop(&mut self) {
        let v = std::mem::take(&mut self.data);
        if v.capacity() >= POOL_MIN_BYTES {
            let mut pool = BUF_POOL.lock().unwrap();
            if pool.len() < pool_max() {
                pool.push(v);
            }
        }
    }
}

impl std::ops::Deref for SharedBuf {
    type Target = [u8];
    #[inline]
    fn deref(&self) -> &[u8] {
        &self.data
    }
}

impl std::fmt::Debug for SharedBuf {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "SharedBuf({} bytes)", self.data.len())
    }
}

/// Packed byte payload of a quantized tensor: either owned, or a **view into a
/// shared buffer**. The share case lets an expert's `gate`/`up`/`down` weights —
/// contiguous on disk — be read in one shot into a single allocation the three
/// tensors slice into, instead of three separate reads + allocations (the streaming
/// decode bottleneck). The buffer is an `Arc<SharedBuf>` (not `Arc<[u8]>`) for two
/// reasons: `Arc::new` moves only the Vec header so the payload is never copied
/// (`Arc<[u8]>::from(Box<[u8]>)` re-allocates and memcpys), and dropping the last
/// view recycles the allocation. Derefs to `[u8]`, so consumers see a byte slice.
#[derive(Debug, Clone, Default)]
pub enum Bytes {
    #[default]
    Empty,
    Owned(Vec<u8>),
    Shared {
        buf: std::sync::Arc<SharedBuf>,
        off: usize,
        len: usize,
    },
}

impl Bytes {
    #[inline]
    pub fn as_slice(&self) -> &[u8] {
        match self {
            Bytes::Empty => &[],
            Bytes::Owned(v) => v,
            Bytes::Shared { buf, off, len } => &buf[*off..*off + *len],
        }
    }
    #[inline]
    pub fn len(&self) -> usize {
        self.as_slice().len()
    }
    #[inline]
    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }
}

impl std::ops::Deref for Bytes {
    type Target = [u8];
    #[inline]
    fn deref(&self) -> &[u8] {
        self.as_slice()
    }
}

impl From<Vec<u8>> for Bytes {
    fn from(v: Vec<u8>) -> Self {
        Bytes::Owned(v)
    }
}

impl PartialEq for Bytes {
    fn eq(&self, other: &Self) -> bool {
        self.as_slice() == other.as_slice()
    }
}

/// A quantized tensor of logical shape `[O, I]` (rows × cols).
///
/// Exactly one of the payload buffers is populated per `fmt`:
///   - `F32`  → `qf`
///   - `Int8` → `q8` (1 byte/param) + per-row scale `s`
///   - `Int2` → `q4` (4 values/byte, packed) + per-row scale `s`
///
/// The heavy `unsafe`/SIMD matmul kernels that consume this live in
/// `colibri-kernels`; this type is just the container.
#[derive(Debug, Clone, Default)]
pub struct QTensor {
    pub fmt_code: i32,
    pub qf: Vec<f32>,
    pub q8: Vec<i8>,
    pub q4: Bytes,
    /// per-row scales (length `O`), empty for `F32`
    pub s: Vec<f32>,
    /// NVFP4 (`fmt_code == 5`) only: ue4m3 per-16-input block scales, `O × ceil(I/16)`
    /// bytes row-major. The effective scale of the 16-wide block containing column `c`
    /// of row `r` is `f8e4m3_to_f32(bs[r*ceil(I/16) + c/16]) * g`. Empty otherwise.
    pub bs: Bytes,
    /// NVFP4 (`fmt_code == 5`) only: per-tensor global scale (modelopt-style; the block
    /// scales above are multiplied by it). `0.0` / unused for every other format.
    pub g: f32,
    /// rows (output dim)
    pub o: i32,
    /// cols (input dim)
    pub i: i32,
    /// Whether this tensor is stable/resident and may be cached on the GPU. Set
    /// for dense weights and preloaded experts; left `false` for streaming
    /// experts (whose buffers are reused for different ids, so a device cache
    /// keyed by address would go stale). Mirrors the C engine's `cuda_eligible`.
    pub gpu_eligible: bool,
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
            4 => n + self.o as i64 * 4, // e4m3 fp8: 1 byte/weight + scales
            // NVFP4: ceil(I/2) nibbles + ceil(I/16) ue4m3 block scales per row + 1 global.
            5 => {
                self.o as i64 * ((self.i as i64 + 1) / 2)
                    + self.o as i64 * ((self.i as i64 + 15) / 16)
                    + 4
            }
            3 => self.o as i64 * ((self.i as i64 + 3) / 4) + self.o as i64 * 4,
            // Every real format (0,1,3,4,5) has an explicit arm above; an unknown
            // code contributes no resident bytes.
            _ => 0,
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
        // int2: O*ceil(I/4) + O*4
        assert_eq!(qt(3, 10, 21).bytes(), 10 * 6 + 10 * 4);
    }

    #[test]
    fn format_bits() {
        assert_eq!(QFormat::from_code(1), Some(QFormat::Int8));
        assert_eq!(QFormat::Int8.bits(), 8);
        assert_eq!(QFormat::from_code(3), Some(QFormat::Int2));
        assert_eq!(QFormat::Int2.bits(), 2);
        assert_eq!(QFormat::from_code(2), None); // int4 removed
        assert_eq!(QFormat::from_code(9), None);
    }

    #[test]
    fn sharedbuf_pool_recycles_allocation() {
        // Distinctive size so parallel tests can't collide in the global pool.
        const N: usize = (3 << 20) + 4096;
        let mut a = SharedBuf::with_len(N);
        a.as_mut_slice()[0] = 7;
        let ptr = a.as_ptr();
        drop(a); // capacity >= 1 MiB → returned to the pool
        let b = SharedBuf::with_len(N);
        assert_eq!(b.as_ptr(), ptr, "drop should recycle the allocation");
        assert_eq!(b.len(), N);
        drop(b);
        // A smaller request reuses the larger allocation with an exact length.
        let c = SharedBuf::with_len(N - 4096);
        assert_eq!(c.as_ptr(), ptr);
        assert_eq!(c.len(), N - 4096);
    }

    #[test]
    fn sharedbuf_small_is_fresh_and_zeroed() {
        // Below the pool threshold: never recycled, so contents are zeroed.
        let d = SharedBuf::with_len(64);
        assert_eq!(d.len(), 64);
        assert!(d.iter().all(|&b| b == 0));
    }

    #[test]
    fn bytes_shared_views_slice_a_sharedbuf() {
        let mut sb = SharedBuf::with_len(64);
        for (i, b) in sb.as_mut_slice().iter_mut().enumerate() {
            *b = i as u8;
        }
        let arc = std::sync::Arc::new(sb);
        let v = Bytes::Shared { buf: arc.clone(), off: 16, len: 8 };
        assert_eq!(&*v, &[16, 17, 18, 19, 20, 21, 22, 23]);
        assert_eq!(v.len(), 8);
    }
}
