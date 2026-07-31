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
    View {
        ptr: *const u8,
        len: usize,
        _owner: std::sync::Arc<dyn std::any::Any + Send + Sync>,
    },
}

// SAFETY: a `View` is immutable for its whole life and its bytes stay mapped as long as
// `_owner` is alive, which the `Arc` guarantees. The raw pointer is only ever read.
unsafe impl Send for SharedBuf {}
unsafe impl Sync for SharedBuf {}

/// Recycled allocations plus the bytes they hold, so the pool can be bounded by size
/// rather than by count. Tracked inside the mutex — no separate atomic to drift.
#[derive(Default)]
struct Pool {
    bufs: Vec<Vec<u8>>,
    bytes: u64,
}

static BUF_POOL: std::sync::Mutex<Pool> = std::sync::Mutex::new(Pool {
    bufs: Vec::new(),
    bytes: 0,
});

/// Don't pool buffers smaller than this — tiny reads don't pay the fault cost.
const POOL_MIN_BYTES: usize = 1 << 20;

/// Page-lock hooks, installed by the CUDA backend once a device exists.
///
/// A GPU copy out of *pageable* host memory is bounced through the driver's own staging
/// buffer; out of *registered* memory it is a straight DMA. That difference is the whole
/// reason the expert staging path has to copy every expert into a pinned buffer before
/// uploading it — a copy that was measured at 10.5 CPU-seconds per M2.7 prefill, spent on a
/// box whose cores were already oversubscribed. Pinning the pool's own allocations lets the
/// upload read them where they already are.
///
/// The pool is the right place for this because it is bounded and it recycles: registration
/// is paid once per *allocation*, not once per expert, and a few dozen buffers then serve
/// every expert for the rest of the run.
///
/// Unset when there is no CUDA device, which is also what makes this free for CPU-only
/// builds — no hook, no registration, and the allocation path is exactly what it was.
type PinHooks = (fn(*mut u8, usize) -> bool, fn(*mut u8));
static HOST_PIN: std::sync::OnceLock<PinHooks> = std::sync::OnceLock::new();

/// Base pointers currently page-locked by [`pin_alloc`].
///
/// Needed because hooks are installed *after* startup: buffers allocated before then are
/// unregistered, and unregistering one would be a driver error on memory it never locked.
/// Tracking what we actually registered is what keeps release symmetric with acquire.
static PINNED: std::sync::Mutex<std::collections::BTreeSet<usize>> =
    std::sync::Mutex::new(std::collections::BTreeSet::new());

/// Page-locks succeeded / failed, and the bytes currently locked.
pub static PIN_OK: std::sync::atomic::AtomicU64 = std::sync::atomic::AtomicU64::new(0);
pub static PIN_FAIL: std::sync::atomic::AtomicU64 = std::sync::atomic::AtomicU64::new(0);
pub static PIN_BYTES: std::sync::atomic::AtomicU64 = std::sync::atomic::AtomicU64::new(0);
/// Allocations left pageable because [`pin_max_bytes`] was already reached. Non-zero is
/// normal and healthy — it is the ceiling doing its job, not a failure.
pub static PIN_CAPPED: std::sync::atomic::AtomicU64 = std::sync::atomic::AtomicU64::new(0);

/// Install the page-lock hooks. Called by the CUDA backend from `init`; idempotent.
pub fn set_host_pin_hooks(reg: fn(*mut u8, usize) -> bool, unreg: fn(*mut u8)) {
    let _ = HOST_PIN.set((reg, unreg));
}

/// Bytes the engine has granted for page-locking, set per model by [`set_pin_budget`].
static PIN_BUDGET: std::sync::atomic::AtomicU64 = std::sync::atomic::AtomicU64::new(0);

/// Grant page-locking a budget, in bytes. **Pass the expert-cache fill target** — the same
/// per-model number the RAM ledger is given for `Class::Experts`.
///
/// This replaced a flat 8 GB default whose justification was wrong. That default claimed
/// page-locking would make memory unreclaimable and endanger the ceiling. What gets locked
/// here are *live, anonymous, dirty* heap allocations held by the expert cache, and with
/// swap off the kernel cannot reclaim those pages whether they are locked or not. Locking
/// them takes nothing from the ceiling that holding them had not already taken.
///
/// A flat cap was also actively harmful: under max residency the cache holds tens of GB, so
/// an 8 GB ceiling would leave most experts on the slow path in exactly the regime we ship.
/// The bound that is real is the one the ledger already computes per model — the expert
/// grant, derived from that model's coverage and headroom — because a buffer inside that
/// grant is memory the process was always going to hold.
///
/// Zero (the default, when no engine has set it) means **do not page-lock at all**, which
/// keeps exactly the behaviour that shipped before any of this. That is the right default
/// for tests, CPU-only builds, and any caller that has not thought about it.
pub fn set_pin_budget(bytes: u64) {
    PIN_BUDGET.store(bytes, std::sync::atomic::Ordering::Relaxed);
}

/// Ceiling on page-locked bytes: `COLI_PIN_MAX_MB` if set (for experiments), else the
/// engine's per-model grant from [`set_pin_budget`], else zero.
fn pin_max_bytes() -> u64 {
    static ENV: std::sync::OnceLock<Option<u64>> = std::sync::OnceLock::new();
    let env = *ENV.get_or_init(|| {
        std::env::var("COLI_PIN_MAX_MB")
            .ok()
            .and_then(|s| s.parse::<u64>().ok())
            .map(|m| m << 20)
    });
    env.unwrap_or_else(|| PIN_BUDGET.load(std::sync::atomic::Ordering::Relaxed))
}

/// Page-lock a freshly allocated pool buffer, if there is a device to pin it for.
fn pin_alloc(v: &mut Vec<u8>) {
    use std::sync::atomic::Ordering::Relaxed;
    let Some((reg, _)) = HOST_PIN.get() else {
        return;
    };
    if v.capacity() < POOL_MIN_BYTES {
        return;
    }
    if PIN_BYTES.load(Relaxed) + v.capacity() as u64 > pin_max_bytes() {
        PIN_CAPPED.fetch_add(1, Relaxed);
        return;
    }
    let p = v.as_mut_ptr();
    // `cudaHostRegister` wants a page-aligned range. At >= 1 MiB malloc serves the
    // allocation via mmap so this holds, but a `false` here is a silent no-pin rather
    // than a driver error, and the counter says which happened.
    if (p as usize) % 4096 != 0 {
        PIN_FAIL.fetch_add(1, Relaxed);
        return;
    }
    if reg(p, v.capacity()) {
        PINNED.lock().unwrap().insert(p as usize);
        PIN_OK.fetch_add(1, Relaxed);
        PIN_BYTES.fetch_add(v.capacity() as u64, Relaxed);
    } else {
        PIN_FAIL.fetch_add(1, Relaxed);
    }
}

/// Release a page-lock before the allocation is freed. A registration that outlives its
/// memory is a dangling page-lock in the driver, so this must run on every free path.
fn unpin_alloc(v: &mut Vec<u8>) {
    use std::sync::atomic::Ordering::Relaxed;
    let Some((_, unreg)) = HOST_PIN.get() else {
        return;
    };
    let p = v.as_mut_ptr();
    if PINNED.lock().unwrap().remove(&(p as usize)) {
        unreg(p);
        PIN_BYTES.fetch_sub(v.capacity() as u64, Relaxed);
    }
}

/// `(page-locks taken, failed, bytes currently locked, capped by the ceiling)`.
pub fn pin_profile() -> (u64, u64, u64, u64) {
    use std::sync::atomic::Ordering::Relaxed;
    (
        PIN_OK.load(Relaxed),
        PIN_FAIL.load(Relaxed),
        PIN_BYTES.load(Relaxed),
        PIN_CAPPED.load(Relaxed),
    )
}

/// Max pooled entries (`COLI_BUF_POOL`; `0` disables recycling).
///
/// **The default was 32 and that cost GLM 1.20x on expert-load.** A rejected return is
/// not a no-op: at >= `POOL_MIN_BYTES` malloc served the buffer via `mmap`, so dropping
/// it is a real `munmap` with page-table teardown, and it happens under the cache lock
/// with the drive idle. Measured on 42b2, interleaved, token-identical:
///
/// | cap | expert-load | evict `free` | peak RSS |
/// |---|---|---|---|
/// | 0 | 18451 / 18259 ms | 5506 / 5270 ms | |
/// | 32 | 14449 / 14411 ms | 2057 / 2045 ms | 59.8 GiB |
/// | **64** | **12078 ms** | **18 ms** | **59.8 GiB** |
/// | 4096 | 12064 / 12083 ms | 1 / 1 ms | |
///
/// 64 captures the whole win and 4096 adds nothing, so the knee is just above 32. RSS is
/// **identical** at 32 and 64: the pool does not retain the extra memory, it only needs
/// headroom not to reject returns. Default set to 2x the measured knee for models with
/// more spans per batch than GLM's ~12.5.
///
/// Verified across the fleet at 32 vs 128, interleaved, **token-identical and
/// RSS-identical for all five**:
///
/// | model | expert-load | where |
/// |---|---|---|
/// | GLM-5.2 | 14110/14274 -> 12079/12088 ms (1.17x) | `evict free` 1804 -> 1 ms |
/// | Kimi-K3 | 28331/28084 -> 26433/26425 ms (1.07x) | `span-setup alloc` 1470 -> 108 ms |
/// | MiniMax-M2.7 | 473 -> 461/473 ms | neutral: all mmap views, 1 pool op |
/// | MiniMax-M3 | 8111/8054 -> 7991/7922 ms | neutral: never overflowed at 32 |
/// | Nemotron-3 | 2 ms | neutral: preloads once, never frees |
///
/// The three neutral models are neutral *by construction* — they reject no returns even
/// at 32 — not by luck.
/// **The count cap is off by default; [`pool_max_bytes`] is the live bound.**
///
/// A count is the wrong unit and the old default of 128 was a constant that helped some
/// models and quietly hurt others. Entries range from ~1 MB (Kimi-K3's MXFP4 spans) to
/// ~21 MB (GLM's coalesced experts), so 128 entries meant **128 MB retained for K3 against
/// 2.7 GB for GLM** — for the model with the *most* spans per batch, the count cap bound at
/// a twentieth of the byte cap it was supposed to sit under, and every return past it was a
/// real `munmap`. That is the same failure the cap was raised from 32 to fix, still present
/// for small-span models.
///
/// The byte cap adapts in exactly the dimension that matters: small spans retain more
/// buffers, large spans retain fewer, and the memory ceiling is protected either way. So
/// the count cap has nothing left to do. Its own documentation already called it "a coarse
/// secondary limit kept so the A/B above stays reproducible", and the measured table says
/// 4096 adds nothing over 64 — i.e. once the cap stops binding, raising it further is
/// neutral. Unlimited is that same plateau.
///
/// `COLI_BUF_POOL` still sets it, so every A/B in the table above reproduces exactly.
///
/// **Measured neutral on speed** (2026-07-31, cap=128 vs unset, one binary, 2 reps ABBA,
/// token-identical): M2.7 wall 52 s unset / 50 s capped, moe 29.0 / 28.1 s; GLM pack
/// 37-40 GB/s unset / 30-36 capped with `cpu ~= elapsed` in both arms. This change is
/// justified by the K3-vs-GLM asymmetry above, **not** by a throughput claim — and in
/// practice it is a no-op for the current fleet, because at these span sizes the byte cap
/// already binds first for GLM and the count cap barely bound for M2.7. K3, the model it
/// exists for, is still unmeasured.
fn pool_max() -> usize {
    static N: std::sync::OnceLock<usize> = std::sync::OnceLock::new();
    *N.get_or_init(|| {
        std::env::var("COLI_BUF_POOL")
            .ok()
            .and_then(|s| s.parse().ok())
            .unwrap_or(usize::MAX)
    })
}

/// Arena mode: the pool is pre-allocated to a fixed size and becomes the **only** source
/// of expert-sized buffers, so expert memory neither grows nor is ever freed.
///
/// `0` (default) = organic pool, the behaviour described above. Non-zero = arena: that
/// many bytes of `slot_bytes` buffers are allocated and touched once at startup, and
/// [`SharedBuf::with_len`] serves every request of that size from them.
///
/// This is what makes "never enter swap" a property of the design. An organic pool is
/// bounded in what it *retains* but not in what it *allocates* — under a burst it will
/// happily allocate past the pool cap and free the excess later. A pre-allocated arena
/// cannot: the memory is claimed up front, recycled by overwriting cold slots, and never
/// returned. Combined with `RamManager`, which sizes this grant from what requests will
/// not need, the process's expert footprint is a constant.
static ARENA: std::sync::Mutex<Option<ArenaCfg>> = std::sync::Mutex::new(None);

#[derive(Clone, Copy)]
struct ArenaCfg {
    slot_bytes: usize,
    slots: usize,
}

/// Pre-allocate the expert arena: `slots` buffers of `slot_bytes` each.
///
/// Touches every buffer so the pages are really resident and accounted — an untouched
/// allocation is a promise the kernel has not yet had to keep, which is exactly the kind
/// of deferred cost that made `MemAvailable` misleading in the first place.
///
/// Returns the bytes actually claimed.
pub fn arena_init(slot_bytes: usize, slots: usize) -> u64 {
    if slot_bytes == 0 || slots == 0 {
        return 0;
    }
    let mut pool = BUF_POOL.lock().unwrap();
    pool.bufs.reserve(slots);
    for _ in 0..slots {
        let mut v = vec![0u8; slot_bytes];
        // Touch one byte per 4 KiB page: `vec![0u8; n]` may be served by a lazily-faulted
        // mmap, and a slot that faults on first *use* would take its cost inside a
        // forward pass rather than at startup.
        for p in (0..slot_bytes).step_by(4096) {
            unsafe { std::ptr::write_volatile(v.as_mut_ptr().add(p), 0) };
        }
        pin_alloc(&mut v); // arena slots live for the process, so this is paid once each
        pool.bytes += v.capacity() as u64;
        pool.bufs.push(v);
    }
    let claimed = pool.bytes;
    drop(pool);
    *ARENA.lock().unwrap() = Some(ArenaCfg { slot_bytes, slots });
    claimed
}

/// `(slot_bytes, slots)` when the arena is active.
pub fn arena_cfg() -> Option<(usize, usize)> {
    ARENA.lock().unwrap().map(|a| (a.slot_bytes, a.slots))
}

/// Max **bytes** retained (`COLI_BUF_POOL_MB`, default 2048).
///
/// A count is the wrong unit here: entries range from 1 MB (K3's MXFP4 spans) to ~21 MB
/// (GLM's coalesced experts), so the same cap means 128 MB for one model and 2.7 GB for
/// another. This is the bound that actually protects the memory ceiling
/// (see `memory-ceiling-is-real`); `pool_max` is a coarse secondary limit kept so the
/// A/B above stays reproducible.
fn pool_max_bytes() -> u64 {
    static N: std::sync::OnceLock<u64> = std::sync::OnceLock::new();
    *N.get_or_init(|| {
        std::env::var("COLI_BUF_POOL_MB")
            .ok()
            .and_then(|s| s.parse().ok())
            .unwrap_or(2048u64)
            << 20
    })
}

/// Pool effectiveness: recycled / freshly allocated / returned / **rejected because the
/// pool was full**.
///
/// These were added because the old 32-entry cap demonstrably bound and no model of the
/// workload explained why. They answered it: at cap 32 GLM rejected **774 returns
/// totalling 16.4 GB** (21.2 MB each — exactly one coalesced expert) and K3 rejected
/// **5431 totalling 95.3 GB**, against 0 and 76 at cap 128.
///
/// They also corrected the mechanism. Missing the pool is not symmetric with being
/// rejected by it, and the two land in *different* profile phases:
///
/// - a **miss** on acquire falls through to `vec![0u8; len]` — an mmap **plus
///   zero-filling** the whole buffer — and shows up in `span-setup / alloc`;
/// - a **rejection** on release drops a >= `POOL_MIN_BYTES` buffer that malloc served via
///   mmap, so it is a real `munmap` with page-table teardown, and shows up in
///   `evict / free`.
///
/// Which side dominates is model-dependent, not size-dependent: GLM's cost was `free`
/// (1804 -> 1 ms) while K3's was `alloc` (1470 -> 108 ms), for near-identical buffer
/// sizes. An early explanation that K3's spans were below `POOL_MIN_BYTES` was simply
/// wrong, and these counters are what disproved it.
pub static POOL_HITS: std::sync::atomic::AtomicU64 = std::sync::atomic::AtomicU64::new(0);
pub static POOL_MISSES: std::sync::atomic::AtomicU64 = std::sync::atomic::AtomicU64::new(0);
pub static POOL_PUSHES: std::sync::atomic::AtomicU64 = std::sync::atomic::AtomicU64::new(0);
pub static POOL_DROPS: std::sync::atomic::AtomicU64 = std::sync::atomic::AtomicU64::new(0);
/// Bytes rejected, so the drops can be read as memory traffic rather than a bare count.
pub static POOL_DROP_BYTES: std::sync::atomic::AtomicU64 = std::sync::atomic::AtomicU64::new(0);
/// Times a caller had to wait for an arena slot (every slot lent out at once).
pub static ARENA_WAITS: std::sync::atomic::AtomicU64 = std::sync::atomic::AtomicU64::new(0);
/// Times the wait gave up and allocated anyway — always a bug, never contention.
pub static ARENA_STARVED: std::sync::atomic::AtomicU64 = std::sync::atomic::AtomicU64::new(0);

/// `(hits, misses, pushes, drops, drop_bytes)`.
pub fn pool_profile() -> (u64, u64, u64, u64, u64) {
    use std::sync::atomic::Ordering::Relaxed;
    (
        POOL_HITS.load(Relaxed),
        POOL_MISSES.load(Relaxed),
        POOL_PUSHES.load(Relaxed),
        POOL_DROPS.load(Relaxed),
        POOL_DROP_BYTES.load(Relaxed),
    )
}

impl SharedBuf {
    /// A buffer of exactly `len` bytes: recycled from the pool when one with
    /// enough capacity is available (contents are stale, caller overwrites),
    /// freshly zero-allocated otherwise.
    pub fn with_len(len: usize) -> SharedBuf {
        use std::sync::atomic::Ordering::Relaxed;
        if len >= POOL_MIN_BYTES {
            let mut pool = BUF_POOL.lock().unwrap();
            if let Some(i) = pool.bufs.iter().position(|v| v.capacity() >= len) {
                let v = pool.bufs.swap_remove(i);
                pool.bytes = pool.bytes.saturating_sub(v.capacity() as u64);
                drop(pool);
                // Stale bytes are fine: previously written, about to be overwritten.
                let mut v = v;
                v.truncate(len);
                v.resize(len, 0);
                POOL_HITS.fetch_add(1, Relaxed);
                return SharedBuf {
                    store: Store::Heap(v),
                };
            }
            POOL_MISSES.fetch_add(1, Relaxed);
            // Arena mode, right-sized request, no slot free: every slot is currently
            // lent out. Falling through to `vec![0u8; len]` here is what would let the
            // expert footprint grow past its grant — the exact overshoot that put
            // MiniMax-M3 into swap. Wait for a slot instead; the cache returns one as
            // soon as any in-flight expert is dropped.
            //
            // Deadlock is not possible while callers hold at most one slot at a time
            // *more* than the arena has: the arena is sized to the cache budget, and the
            // cache evicts before it loads. `ARENA_WAITS` makes a stall visible rather
            // than mysterious if that ever stops being true.
            let arena_slot = ARENA
                .lock()
                .unwrap()
                .map(|a| len <= a.slot_bytes)
                .unwrap_or(false);
            if arena_slot {
                drop(pool);
                ARENA_WAITS.fetch_add(1, Relaxed);
                let mut spins = 0u32;
                loop {
                    std::thread::yield_now();
                    let mut p = BUF_POOL.lock().unwrap();
                    if let Some(i) = p.bufs.iter().position(|v| v.capacity() >= len) {
                        let v = p.bufs.swap_remove(i);
                        p.bytes = p.bytes.saturating_sub(v.capacity() as u64);
                        drop(p);
                        let mut v = v;
                        v.truncate(len);
                        v.resize(len, 0);
                        POOL_HITS.fetch_add(1, Relaxed);
                        return SharedBuf {
                            store: Store::Heap(v),
                        };
                    }
                    drop(p);
                    spins += 1;
                    if spins > 10_000 {
                        // Every slot has been lent out for a long time — a real bug, not
                        // contention. Allocate rather than hang; the ledger will notice
                        // the overshoot and this counter says where it came from.
                        ARENA_STARVED.fetch_add(1, Relaxed);
                        break;
                    }
                    if spins % 64 == 0 {
                        std::thread::sleep(std::time::Duration::from_micros(50));
                    }
                }
            }
        }
        let mut v = vec![0u8; len];
        pin_alloc(&mut v);
        SharedBuf {
            store: Store::Heap(v),
        }
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
        SharedBuf {
            store: Store::View {
                ptr,
                len,
                _owner: owner,
            },
        }
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
        let Store::Heap(v) = &mut self.store else {
            return;
        };
        let v = std::mem::take(v);
        if v.capacity() >= POOL_MIN_BYTES {
            use std::sync::atomic::Ordering::Relaxed;
            let cap = v.capacity() as u64;
            let mut pool = BUF_POOL.lock().unwrap();
            // In arena mode a return is ALWAYS accepted. The arena is the memory — handing
            // a slot back is how it gets recycled, and rejecting one would turn a recycle
            // into a `munmap` and shrink the arena permanently, which is the opposite of
            // what a fixed grant means. The caps below are for the organic pool only.
            let arena = ARENA.lock().unwrap().is_some();
            if arena || (pool.bufs.len() < pool_max() && pool.bytes + cap <= pool_max_bytes()) {
                pool.bytes += cap;
                pool.bufs.push(v);
                POOL_PUSHES.fetch_add(1, Relaxed);
            } else {
                // Rejected: `v` is dropped here, and at >=1 MB malloc served it via mmap,
                // so this is a real munmap with page-table teardown — the 2057 ms. Release
                // the page-lock first: the driver must not be left holding a registration
                // for memory that is about to be handed back to the kernel.
                drop(pool);
                let mut v = v;
                unpin_alloc(&mut v);
                POOL_DROPS.fetch_add(1, Relaxed);
                POOL_DROP_BYTES.fetch_add(cap, Relaxed);
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
        let v = Bytes::Shared {
            buf: arc.clone(),
            off: 16,
            len: 8,
        };
        assert_eq!(&*v, &[16, 17, 18, 19, 20, 21, 22, 23]);
        assert_eq!(v.len(), 8);
    }
}
