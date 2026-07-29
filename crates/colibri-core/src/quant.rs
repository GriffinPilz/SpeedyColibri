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
    store: Store,
}

/// Where a [`SharedBuf`]'s bytes live.
enum Store {
    /// A pool-recycled heap allocation — the streaming read path writes into it.
    Heap(Vec<u8>),
    /// A read-only *view* into memory owned by something else, kept alive by `_owner`
    /// (in practice a shard's `mmap`). Owns no allocation, so it is never pooled and
    /// costs nothing to drop.
    ///
    /// The owner is a type-erased `Arc` so the mapping itself can live in
    /// `colibri-safetensors` (which has `libc`) rather than dragging platform code
    /// into this crate.
    View { ptr: *const u8, len: usize, _owner: std::sync::Arc<dyn std::any::Any + Send + Sync> },
}

// SAFETY: a `View` is immutable for its whole life and its bytes stay mapped as long as
// `_owner` is alive, which the `Arc` guarantees. The raw pointer is only ever read.
unsafe impl Send for SharedBuf {}
unsafe impl Sync for SharedBuf {}

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
                return SharedBuf { store: Store::Heap(v) };
            }
        }
        SharedBuf { store: Store::Heap(vec![0u8; len]) }
    }

    /// Wrap `len` bytes at `ptr` that are owned by `owner`, without copying.
    ///
    /// # Safety
    /// `ptr` must be valid for reads of `len` bytes, and must stay valid and immutable
    /// for as long as `owner` is alive. Holding `owner` in the returned buffer is what
    /// enforces the lifetime.
    pub unsafe fn from_view(
        owner: std::sync::Arc<dyn std::any::Any + Send + Sync>,
        ptr: *const u8,
        len: usize,
    ) -> SharedBuf {
        SharedBuf { store: Store::View { ptr, len, _owner: owner } }
    }

    /// True when this buffer is a borrowed view rather than an owned allocation.
    #[inline]
    pub fn is_view(&self) -> bool {
        matches!(self.store, Store::View { .. })
    }

    /// Writable bytes. Only meaningful for a heap buffer — a view is read-only by
    /// construction, and the read path only mutates a buffer it just allocated with
    /// [`SharedBuf::with_len`].
    #[inline]
    pub fn as_mut_slice(&mut self) -> &mut [u8] {
        match &mut self.store {
            Store::Heap(v) => v,
            Store::View { .. } => {
                panic!("SharedBuf::as_mut_slice on a mapped view — views are read-only")
            }
        }
    }
}

impl Drop for SharedBuf {
    fn drop(&mut self) {
        // Only heap allocations are recycled. A view owns nothing — pooling its bytes
        // would hand a mapped region out as a scratch buffer for the next read.
        let Store::Heap(v) = &mut self.store else { return };
        let v = std::mem::take(v);
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
        match &self.store {
            Store::Heap(v) => v,
            // SAFETY: the from_view contract — valid for `len` reads while `_owner` lives.
            Store::View { ptr, len, .. } => unsafe { std::slice::from_raw_parts(*ptr, *len) },
        }
    }
}

impl std::fmt::Debug for SharedBuf {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        let kind = if self.is_view() { "view" } else { "heap" };
        write!(f, "SharedBuf({} bytes, {kind})", self.len())
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
    /// Block-scaled FP4 (`fmt_code == 5` NVFP4, `== 6` MXFP4) only: per-block scale
    /// bytes, `O × ceil(I/BLK)` row-major, where `BLK` is 16 for NVFP4 and 32 for MXFP4.
    /// The effective scale of the block containing column `c` of row `r` is
    /// `decode(bs[r*ceil(I/BLK) + c/BLK]) * g`, with `decode` = `f8e4m3_to_f32` for
    /// NVFP4 and `2^(byte - 127)` (OCP E8M0) for MXFP4. Empty otherwise.
    pub bs: Bytes,
    /// Block-scaled FP4 (`fmt_code == 5` / `6`) only: per-tensor global scale, which the
    /// block scales above are multiplied by. `0.0` / unused for every other format.
    ///
    /// NVFP4 uses it modelopt-style (`amax / (6 * 448)`). MXFP4 has no global scale of
    /// its own — the E8M0 byte already spans 2^±127 — so a native MXFP4 tensor carries
    /// `g == 1.0`. It stays in the layout so both formats share one decode signature.
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
            // MXFP4: same e2m1 nibbles, but one E8M0 byte per 32 inputs instead of per 16.
            // That is 4.25 bits/weight vs NVFP4's 4.5 — the reason a natively-MXFP4
            // checkpoint (Kimi-K3) is 5.9% smaller kept as-is than transcoded to NVFP4.
            6 => {
                self.o as i64 * ((self.i as i64 + 1) / 2)
                    + self.o as i64 * ((self.i as i64 + 31) / 32)
                    + 4
            }
            3 => self.o as i64 * ((self.i as i64 + 3) / 4) + self.o as i64 * 4,
            // Every real format (0,1,3,4,5,6) has an explicit arm above; an unknown
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
