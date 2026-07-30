//! Expert-arena behaviour, in its **own test binary**.
//!
//! The arena and its buffer pool are process-global. Testing them from the unit-test
//! binary is unreliable and was: an address-identity check failed under parallel
//! execution and passed under `--test-threads=1`, because other tests allocate into the
//! same pool concurrently. Cleaning up after each case then produced a panic inside a
//! destructor during unwind, which aborts the whole binary.
//!
//! An integration test gets its own process, so the global is genuinely private here.
//! Everything below therefore runs in ONE `#[test]` — a second test in this file would
//! share the process and reintroduce exactly the interference this is avoiding.

use colibri_core::quant::{arena_cfg, arena_init, pool_profile};
use colibri_core::SharedBuf;

const SLOT: usize = 2 << 20; // 2 MiB — above POOL_MIN_BYTES, so it is a pooled size
const SLOTS: usize = 8;

#[test]
fn arena_claims_its_memory_once_and_then_recycles_by_overwrite() {
    // --- the grant is exact ---------------------------------------------------------
    let claimed = arena_init(SLOT, SLOTS);
    assert_eq!(claimed, (SLOT * SLOTS) as u64, "the arena claims exactly what it was asked");
    assert_eq!(arena_cfg(), Some((SLOT, SLOTS)));

    // --- lending out every slot does not allocate ------------------------------------
    {
        let held: Vec<_> = (0..SLOTS).map(|_| SharedBuf::with_len(SLOT)).collect();
        assert_eq!(held.len(), SLOTS);
        // Distinct memory: an arena that handed the same buffer twice would corrupt.
        let mut addrs: Vec<usize> = held.iter().map(|b| b.as_ptr() as usize).collect();
        addrs.sort_unstable();
        addrs.dedup();
        assert_eq!(addrs.len(), SLOTS, "every lent slot must be distinct memory");
    } // all returned here

    // --- and re-taking them costs NO fresh allocation --------------------------------
    // `misses` counts requests the pool could not satisfy, which fall through to
    // `vec![0u8; len]` and grow the footprint. That growth is precisely what put
    // MiniMax-M3 into swap, so zero is the whole contract.
    let (_, misses_before, _, _, _) = pool_profile();
    for _ in 0..5 {
        let held: Vec<_> = (0..SLOTS).map(|_| SharedBuf::with_len(SLOT)).collect();
        assert_eq!(held.len(), SLOTS);
    }
    let (_, misses_after, _, drops, drop_bytes) = pool_profile();
    assert_eq!(
        misses_after, misses_before,
        "the arena allocated fresh memory — its footprint is not fixed"
    );

    // --- and nothing is ever handed back to the allocator ---------------------------
    // A rejected return is a real `munmap` (>=1 MiB allocations are mmap-backed), which
    // would shrink the grant permanently — the opposite of a fixed arena.
    assert_eq!(drops, 0, "arena returns must never be rejected");
    assert_eq!(drop_bytes, 0);

    // --- the organic caps do not apply -----------------------------------------------
    // Re-init past `COLI_BUF_POOL`'s default retention so the cap would reject if it were
    // still consulted. (Same process, so this grows the arena rather than replacing it —
    // which is itself the point: the arena only ever grows at an explicit `arena_init`.)
    let more = 200usize;
    arena_init(SLOT, more);
    let (_, m0, _, d0, _) = pool_profile();
    {
        let held: Vec<_> = (0..(SLOTS + more)).map(|_| SharedBuf::with_len(SLOT)).collect();
        assert_eq!(held.len(), SLOTS + more);
    }
    let (_, m1, _, d1, _) = pool_profile();
    assert_eq!(d1, d0, "a return past the organic cap must still be accepted in arena mode");
    assert_eq!(m1, m0, "and taking every slot must not allocate");
}
