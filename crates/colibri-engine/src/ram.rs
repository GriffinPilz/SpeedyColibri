//! Single authority for host RAM.
//!
//! Every allocation that lands in RAM is committed here **before** it is made: dense
//! weights, the expert arena, per-request KV, prefill activations, GPU host staging and
//! read buffers. The sum of live commitments can never exceed [`RamManager::ceiling`], so
//! the process cannot walk itself into swap.
//!
//! # Why a ledger, not a memory probe
//!
//! The previous design asked `MemAvailable` whether an allocation would fit. That is a
//! *reactive* signal and it fails this job in two ways, both observed:
//!
//! - It counts reclaimable page cache as free, so it reads generously right up until the
//!   kernel starts paging. Measured: 3 GB already swapped while `MemAvailable` still
//!   reported 30 GB — five times the danger floor, so no guard fired.
//! - It only moves *after* memory is consumed. A ledger refuses the commitment that would
//!   have caused the pressure; a probe notices afterwards and tries to undo it, which is a
//!   race the allocator always wins.
//!
//! The failure this exists to prevent: the expert cache filled MiniMax-M3 to its 94 GB
//! headroom — safe in isolation, since nothing else had claimed the memory yet — and then
//! inference allocated activations, KV and staging on top. All 16 GB of swap was exhausted
//! and the model generated zero tokens. No component was individually at fault; there was
//! no arbiter.
//!
//! # Fixed grants, and why the cache is not "evicted"
//!
//! The expert cache takes a **fixed grant** at startup and never asks for more. Internally
//! it is an arena of slots: making room for an expert means marking a cold slot reusable
//! and **overwriting it in place**, not freeing it. That matters for three measured
//! reasons:
//!
//! - Releasing memory does not reliably return it. For a model served through mapped
//!   views, dropping the mapping releases a *view*; the file pages stay in the page cache
//!   until the kernel reclaims them on its own schedule. An eviction-based pressure valve
//!   therefore does not relieve pressure — which is why the swap guard could not save the
//!   M3 run.
//! - Free is expensive. Above ~1 MB, glibc serves allocations by `mmap`, so releasing one
//!   is a real `munmap` with page-table teardown: measured at **1976 ms** of a single GLM
//!   run, 98% of everything its eviction path cost.
//! - A grant that never grows cannot overshoot. Sizing it once, up front, is what makes
//!   "never enter swap" a property of the design rather than something a monitor chases.
//!
//! So [`Class::Experts`] is committed once and held. Requests are bounded by what remains,
//! and a request that does not fit **queues or is refused** — it is never allocated anyway.

use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::Arc;

/// What a commitment is for.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Class {
    /// Dense (non-expert) weights. Committed at load, held for the process's life.
    Dense,
    /// The routed-expert arena. One fixed grant at startup; slots are recycled by
    /// overwrite, never by release.
    Experts,
    /// A request's KV cache, for the life of the request.
    Kv,
    /// Prefill activations, GPU host staging, per-request scratch.
    Scratch,
    /// Pooled expert read buffers.
    ReadBuf,
}

const N_CLASSES: usize = 5;

pub struct RamManager {
    /// Hard ceiling on the sum of live commitments.
    ceiling: u64,
    per_class: [AtomicU64; N_CLASSES],
}

impl RamManager {
    pub fn new(ceiling: u64) -> RamManager {
        RamManager { ceiling, per_class: Default::default() }
    }

    pub fn ceiling(&self) -> u64 {
        self.ceiling
    }

    pub fn committed(&self) -> u64 {
        self.per_class.iter().map(|c| c.load(Ordering::Relaxed)).sum()
    }

    pub fn committed_in(&self, class: Class) -> u64 {
        self.per_class[class as usize].load(Ordering::Relaxed)
    }

    /// What the expert arena may take: everything the rigid classes will not need.
    ///
    /// Called once, before the arena is sized. `rigid_reserve` is the caller's estimate of
    /// peak concurrent KV + scratch; it is held back so a request never finds the box full
    /// of experts. Held back, not merely hoped for — the arena grant is permanent.
    pub fn arena_grant(&self, rigid_reserve: u64) -> u64 {
        self.ceiling
            .saturating_sub(self.committed())
            .saturating_sub(rigid_reserve)
    }

    /// Commit `bytes`, or `None` if it would breach the ceiling.
    ///
    /// `None` means **queue or refuse**. It must never be treated as advisory: allocating
    /// past a refusal is the exact path that exhausted swap on MiniMax-M3.
    pub fn commit(self: &Arc<Self>, class: Class, bytes: u64) -> Option<Commitment> {
        if bytes == 0 {
            return Some(Commitment { mgr: Arc::clone(self), class, bytes: 0 });
        }
        loop {
            let committed = self.committed();
            if bytes > self.ceiling.saturating_sub(committed) {
                return None;
            }
            self.per_class[class as usize].fetch_add(bytes, Ordering::Relaxed);
            // Another thread may have committed between the read and the add. Over-commit
            // is the one state that must never persist, so undo and retry.
            if self.committed() <= self.ceiling {
                return Some(Commitment { mgr: Arc::clone(self), class, bytes });
            }
            self.per_class[class as usize].fetch_sub(bytes, Ordering::Relaxed);
        }
    }
}

/// RAII handle: the commitment is live until this is dropped.
#[must_use = "dropping the Commitment immediately releases the RAM it reserved"]
pub struct Commitment {
    mgr: Arc<RamManager>,
    class: Class,
    bytes: u64,
}

impl Commitment {
    pub fn bytes(&self) -> u64 {
        self.bytes
    }

    /// Hand the commitment to a long-lived owner (the arena, the dense tier) that will
    /// hold it for the process's life. Purely intent-revealing — it forgets the handle so
    /// the ledger keeps counting the bytes.
    pub fn hold_forever(self) {
        std::mem::forget(self);
    }
}

impl Drop for Commitment {
    fn drop(&mut self) {
        self.mgr.per_class[self.class as usize].fetch_sub(self.bytes, Ordering::Relaxed);
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn mgr(ceiling: u64) -> Arc<RamManager> {
        Arc::new(RamManager::new(ceiling))
    }

    #[test]
    fn total_commitments_never_exceed_the_ceiling() {
        let m = mgr(100);
        let _a = m.commit(Class::Dense, 40).expect("fits");
        let _b = m.commit(Class::Kv, 40).expect("fits");
        assert_eq!(m.committed(), 80);
        assert!(m.commit(Class::Scratch, 30).is_none(), "110 > 100 must be refused");
        assert_eq!(m.committed(), 80, "a refused commit leaves the ledger untouched");
        let _c = m.commit(Class::Scratch, 20).expect("exact fit is allowed");
        assert_eq!(m.committed(), 100);
        assert!(m.commit(Class::Scratch, 1).is_none());
    }

    /// The arena is sized from what the rigid classes will *not* need, so a request can
    /// always be served without asking the cache to give anything back. This is the whole
    /// point: the M3 failure was the cache taking room that inference then needed.
    #[test]
    fn the_arena_grant_leaves_room_for_requests() {
        let m = mgr(100);
        m.commit(Class::Dense, 10).unwrap().hold_forever();

        let grant = m.arena_grant(25); // hold back 25 for peak KV + scratch
        assert_eq!(grant, 65, "100 ceiling - 10 dense - 25 rigid reserve");
        m.commit(Class::Experts, grant).unwrap().hold_forever();

        // Requests now fit without touching the arena.
        let kv = m.commit(Class::Kv, 20).expect("KV fits in the held-back room");
        let scratch = m.commit(Class::Scratch, 5).expect("scratch fits too");
        assert_eq!(m.committed(), 100);
        assert_eq!(m.committed_in(Class::Experts), 65, "the arena never gave anything up");

        // And the ceiling still binds beyond that.
        assert!(m.commit(Class::Kv, 1).is_none());
        drop(kv);
        drop(scratch);
        assert!(m.commit(Class::Kv, 25).is_some(), "room returns when requests finish");
    }

    /// A request that cannot fit is refused. It is NOT allocated anyway — that is the
    /// behaviour that exhausted 16 GB of swap and produced zero tokens on MiniMax-M3.
    #[test]
    fn an_oversized_request_is_refused_not_squeezed_in() {
        let m = mgr(100);
        m.commit(Class::Dense, 10).unwrap().hold_forever();
        m.commit(Class::Experts, m.arena_grant(20)).unwrap().hold_forever();
        assert_eq!(m.committed(), 80);

        assert!(m.commit(Class::Kv, 30).is_none(), "30 > the 20 held back");
        assert_eq!(m.committed(), 80, "and nothing was committed on the way to refusing");
        assert!(m.commit(Class::Kv, 20).is_some(), "exactly what was held back still fits");
    }

    #[test]
    fn releasing_a_commitment_returns_the_room() {
        let m = mgr(100);
        {
            let _a = m.commit(Class::Kv, 60).unwrap();
            assert!(m.commit(Class::Kv, 50).is_none());
        }
        assert_eq!(m.committed(), 0, "drop releases");
        assert!(m.commit(Class::Kv, 100).is_some(), "the room came back");
    }

    #[test]
    fn concurrent_commits_cannot_overshoot_the_ceiling() {
        let m = mgr(1000);
        let mut hs = Vec::new();
        for _ in 0..16 {
            let m = Arc::clone(&m);
            hs.push(std::thread::spawn(move || {
                let mut held = Vec::new();
                for _ in 0..200 {
                    if let Some(c) = m.commit(Class::Kv, 7) {
                        held.push(c);
                    }
                    assert!(
                        m.committed() <= m.ceiling(),
                        "ceiling breached: {} > {}",
                        m.committed(),
                        m.ceiling()
                    );
                }
                held
            }));
        }
        let all: Vec<_> = hs.into_iter().map(|h| h.join().unwrap()).collect();
        assert!(m.committed() <= m.ceiling());
        drop(all);
        assert_eq!(m.committed(), 0);
    }
}
