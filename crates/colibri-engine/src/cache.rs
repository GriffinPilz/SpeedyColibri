//! Resident expert cache — port of the per-layer expert LRU (`ecache`) and the
//! pinned hot-store (`pin`) from `c/glm.c`, using the LFRU eviction policy from
//! `colibri-core::tier`.
//!
//! Without this, every routed expert is re-read from disk on every token
//! (`ShardsExpertProvider` alone). With it, an expert loaded once **stays
//! resident in RAM** and is only dropped when the cache exceeds its byte budget,
//! at which point the coldest (lowest LFRU score) unpinned expert is evicted.
//! Pinned experts (the hot-store) are never evicted.
//!
//! On DGX Spark this is what keeps the hot experts off the disk path: a 128 GB
//! node holds a few thousand experts resident (see [`capacity`]); the OS page
//! cache is a free L2 for the rest.

use crate::moe::{Expert, ExpertProvider};
use colibri_core::tier::lfru_score;
use std::collections::{HashMap, HashSet};
use std::io;
use std::sync::{Arc, Mutex};

/// One cached expert plus its LFRU bookkeeping.
struct Entry {
    expert: Arc<Expert>,
    bytes: u64,
    heat: u32,
    last: u32,
}

struct State {
    entries: HashMap<(usize, usize), Entry>,
    pinned: HashSet<(usize, usize)>,
    bytes: u64,
    clock: u32,
    hits: u64,
    misses: u64,
    evictions: u64,
}

/// Cache statistics snapshot.
#[derive(Debug, Clone, Copy, Default)]
pub struct CacheStats {
    pub hits: u64,
    pub misses: u64,
    pub evictions: u64,
    /// experts currently resident
    pub resident: usize,
    /// bytes currently resident
    pub bytes: u64,
    /// byte budget
    pub budget: u64,
}

/// A resident, budget-bounded cache in front of any [`ExpertProvider`].
pub struct ExpertCache<P: ExpertProvider> {
    inner: P,
    budget: u64,
    state: Mutex<State>,
}

impl<P: ExpertProvider> ExpertCache<P> {
    /// Wrap `inner` with a cache holding up to `budget_bytes` of experts. Use
    /// `u64::MAX` to never evict (hold everything that's ever loaded).
    pub fn new(inner: P, budget_bytes: u64) -> ExpertCache<P> {
        ExpertCache {
            inner,
            budget: budget_bytes,
            state: Mutex::new(State {
                entries: HashMap::new(),
                pinned: HashSet::new(),
                bytes: 0,
                clock: 0,
                hits: 0,
                misses: 0,
                evictions: 0,
            }),
        }
    }

    /// Pin `(layer, eid)` into the hot-store: once resident it is never evicted.
    /// Loads it now if absent.
    pub fn pin(&self, layer: usize, eid: usize) -> io::Result<()> {
        self.expert(layer, eid)?; // ensure resident
        self.state.lock().unwrap().pinned.insert((layer, eid));
        Ok(())
    }

    /// Current cache statistics.
    pub fn stats(&self) -> CacheStats {
        let s = self.state.lock().unwrap();
        CacheStats {
            hits: s.hits,
            misses: s.misses,
            evictions: s.evictions,
            resident: s.entries.len(),
            bytes: s.bytes,
            budget: self.budget,
        }
    }
}

impl State {
    /// Evict coldest unpinned experts until at or under `budget`.
    fn evict_to(&mut self, budget: u64) {
        while self.bytes > budget {
            let clock = self.clock;
            let pinned = &self.pinned;
            // coldest unpinned entry by LFRU score
            let victim = self
                .entries
                .iter()
                .filter(|(k, _)| !pinned.contains(*k))
                .min_by_key(|(_, e)| lfru_score(e.heat, e.last, clock))
                .map(|(k, _)| *k);
            match victim {
                Some(k) => {
                    if let Some(e) = self.entries.remove(&k) {
                        self.bytes -= e.bytes;
                        self.evictions += 1;
                    }
                }
                None => break, // everything left is pinned
            }
        }
    }
}

impl<P: ExpertProvider> ExpertProvider for ExpertCache<P> {
    fn expert(&self, layer: usize, eid: usize) -> io::Result<Arc<Expert>> {
        let key = (layer, eid);
        // Fast path: resident hit.
        {
            let mut s = self.state.lock().unwrap();
            s.clock = s.clock.wrapping_add(1);
            let clock = s.clock;
            if let Some(e) = s.entries.get_mut(&key) {
                e.heat = e.heat.saturating_add(1);
                e.last = clock;
                let ex = e.expert.clone(); // ends the borrow of s.entries
                s.hits += 1;
                return Ok(ex);
            }
            s.misses += 1;
        }
        // Miss: load outside the lock (disk I/O), then insert + evict.
        let ex = self.inner.expert(layer, eid)?;
        let bytes = ex.bytes();
        let mut s = self.state.lock().unwrap();
        // Another thread may have inserted it while we loaded.
        if let Some(e) = s.entries.get(&key) {
            return Ok(e.expert.clone());
        }
        let clock = s.clock;
        s.entries.insert(
            key,
            Entry {
                expert: ex.clone(),
                bytes,
                heat: 1,
                last: clock,
            },
        );
        s.bytes += bytes;
        let budget = self.budget;
        s.evict_to(budget);
        Ok(ex)
    }
}

/// Capacity planning for DGX Spark deployments.
pub mod capacity {
    /// Byte size of one `[O, I]` tensor stored at `bits` (matches `QTensor::bytes`
    /// / the `qt_alloc` format selection).
    fn qt_bytes(o: u64, i: u64, bits: u32) -> u64 {
        let n = o * i;
        if bits >= 16 {
            n * 4
        } else if bits >= 5 {
            n + o * 4 // int8
        } else if bits >= 3 {
            o * i.div_ceil(2) + o * 4 // int4
        } else {
            o * i.div_ceil(4) + o * 4 // int2
        }
    }

    /// Resident bytes of one routed expert (gate + up + down) for a model with
    /// the given `hidden`/`moe_inter`, at `bits`.
    pub fn bytes_per_expert(hidden: u64, moe_inter: u64, bits: u32) -> u64 {
        // gate [moe_inter, hidden], up [moe_inter, hidden], down [hidden, moe_inter]
        2 * qt_bytes(moe_inter, hidden, bits) + qt_bytes(hidden, moe_inter, bits)
    }

    /// How many experts of `bytes_per_expert` fit in `budget_bytes`.
    pub fn experts_in_budget(budget_bytes: u64, bytes_per_expert: u64) -> u64 {
        if bytes_per_expert == 0 {
            0
        } else {
            budget_bytes / bytes_per_expert
        }
    }
}

/// Available RAM in bytes, best-effort. Reads `/proc/meminfo` `MemAvailable` on
/// Linux (the DGX Spark target); returns `None` elsewhere (e.g. macOS dev boxes),
/// where the caller should fall back to an explicit budget.
pub fn available_ram_bytes() -> Option<u64> {
    let meminfo = std::fs::read_to_string("/proc/meminfo").ok()?;
    for line in meminfo.lines() {
        if let Some(rest) = line.strip_prefix("MemAvailable:") {
            let kb: u64 = rest.split_whitespace().next()?.parse().ok()?;
            return Some(kb * 1024);
        }
    }
    None
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::quantize::qtensor_from_f32;
    use std::sync::atomic::{AtomicUsize, Ordering};

    // A provider that counts how many times it actually loads (i.e. cache misses
    // that reach disk).
    struct CountingProvider {
        loads: AtomicUsize,
        inter: usize,
        d: usize,
    }
    impl ExpertProvider for CountingProvider {
        fn expert(&self, _layer: usize, eid: usize) -> io::Result<Arc<Expert>> {
            self.loads.fetch_add(1, Ordering::Relaxed);
            let mk = |o: usize, i: usize| {
                let w: Vec<f32> = (0..o * i).map(|k| ((k + eid) % 5) as f32 * 0.1).collect();
                qtensor_from_f32(&w, o, i, 8)
            };
            Ok(Arc::new(Expert {
                gate: mk(self.inter, self.d),
                up: mk(self.inter, self.d),
                down: mk(self.d, self.inter),
            }))
        }
    }

    fn counting() -> CountingProvider {
        CountingProvider {
            loads: AtomicUsize::new(0),
            inter: 4,
            d: 8,
        }
    }

    #[test]
    fn hit_avoids_reload() {
        let cache = ExpertCache::new(counting(), u64::MAX);
        let _ = cache.expert(0, 1).unwrap();
        let _ = cache.expert(0, 1).unwrap();
        let _ = cache.expert(0, 1).unwrap();
        assert_eq!(cache.inner.loads.load(Ordering::Relaxed), 1, "loaded once");
        let s = cache.stats();
        assert_eq!(s.misses, 1);
        assert_eq!(s.hits, 2);
        assert_eq!(s.resident, 1);
    }

    #[test]
    fn evicts_when_over_budget() {
        // budget for ~2 experts; load 3 distinct -> one eviction, stays under budget.
        let one = {
            let c = ExpertCache::new(counting(), u64::MAX);
            c.expert(0, 0).unwrap().bytes()
        };
        let cache = ExpertCache::new(counting(), one * 2);
        cache.expert(0, 0).unwrap();
        cache.expert(0, 1).unwrap();
        // touch expert 0 so it's hotter than 1
        cache.expert(0, 0).unwrap();
        cache.expert(0, 2).unwrap(); // triggers eviction of the coldest (expert 1)
        let s = cache.stats();
        assert!(s.bytes <= one * 2, "over budget: {} > {}", s.bytes, one * 2);
        assert_eq!(s.resident, 2);
        assert!(s.evictions >= 1);
        // expert 1 was coldest -> evicted -> reloading it is a miss again
        let before = cache.inner.loads.load(Ordering::Relaxed);
        cache.expert(0, 1).unwrap();
        assert_eq!(cache.inner.loads.load(Ordering::Relaxed), before + 1);
    }

    #[test]
    fn pinned_survives_eviction() {
        let one = {
            let c = ExpertCache::new(counting(), u64::MAX);
            c.expert(0, 0).unwrap().bytes()
        };
        let cache = ExpertCache::new(counting(), one * 2);
        cache.pin(0, 0).unwrap(); // pin expert 0
        cache.expert(0, 1).unwrap();
        cache.expert(0, 2).unwrap(); // eviction — must not drop pinned expert 0
        cache.expert(0, 3).unwrap();
        // expert 0 still resident (a hit, no new load)
        let before = cache.inner.loads.load(Ordering::Relaxed);
        cache.expert(0, 0).unwrap();
        assert_eq!(cache.inner.loads.load(Ordering::Relaxed), before, "pinned reloaded");
    }

    #[test]
    fn glm52_expert_size_and_capacity() {
        // GLM-5.2: hidden 6144, moe_inter 2048, int4 -> ~18-19 MB/expert.
        let bpe = capacity::bytes_per_expert(6144, 2048, 4);
        let mb = bpe as f64 / (1024.0 * 1024.0);
        assert!((17.0..20.0).contains(&mb), "expert MB = {mb}");
        // ~110 GB budget (a Spark after dense+overhead) -> a few thousand experts.
        let n = capacity::experts_in_budget(110 * (1 << 30), bpe);
        assert!((5_000..7_000).contains(&n), "experts in 110GB = {n}");
    }
}
