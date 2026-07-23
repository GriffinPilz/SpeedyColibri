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
use crate::usage::UsageHistory;
use colibri_core::tier::evict_score;
use std::collections::{HashMap, HashSet};
use std::io;
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::{mpsc, Arc, Mutex, OnceLock};

/// Online next-layer expert predictor for speculative prefetch (`COLI_PREFETCH`).
///
/// As tokens stream past it learns two things: per-layer expert **frequency**, and
/// the adjacent-layer **transition** `layer L-1 expert → layer L expert`
/// co-occurrence. Given a layer's routed experts it predicts the *next* layer's
/// likely experts (transition-scored, frequency-backfilled) so they can be loaded
/// in the background during this layer's compute. `scripts/expert_prefetch_analysis.py`
/// measured this "markov+freq" predictor covering ~68% of cache misses at top-16 in
/// the miss-heavy (working-set > cache) regime — the 1–4 Spark case.
struct Predictor {
    topn: usize,
    freq: HashMap<usize, HashMap<u32, u32>>,
    trans: HashMap<usize, HashMap<u32, HashMap<u32, u32>>>,
    last: Option<(usize, Vec<u32>)>,
}

impl Predictor {
    fn new(topn: usize) -> Predictor {
        Predictor { topn, freq: HashMap::new(), trans: HashMap::new(), last: None }
    }

    /// Record this layer's experts and return the predicted top-N for the *next*
    /// layer.
    fn observe_and_predict(&mut self, layer: usize, eids: &[usize]) -> Vec<usize> {
        let cur: Vec<u32> = eids.iter().map(|&e| e as u32).collect();
        let f = self.freq.entry(layer).or_default();
        for &e in &cur {
            *f.entry(e).or_insert(0) += 1;
        }
        if let Some((ll, le)) = self.last.take() {
            if ll + 1 == layer {
                let t = self.trans.entry(layer).or_default();
                for &pe in &le {
                    let c = t.entry(pe).or_default();
                    for &e in &cur {
                        *c.entry(e).or_insert(0) += 1;
                    }
                }
            }
        }
        let predicted = self.predict(layer + 1, &cur);
        self.last = Some((layer, cur));
        predicted
    }

    /// Top-N predicted experts for `target` given `from` (the previous layer's
    /// experts): sum the learned transitions, then backfill by frequency.
    fn predict(&self, target: usize, from: &[u32]) -> Vec<usize> {
        let mut score: HashMap<u32, u32> = HashMap::new();
        if let Some(t) = self.trans.get(&target) {
            for &e in from {
                if let Some(c) = t.get(&e) {
                    for (&ne, &cnt) in c {
                        *score.entry(ne).or_insert(0) += cnt;
                    }
                }
            }
        }
        let mut ranked: Vec<(u32, u32)> = score.into_iter().collect();
        ranked.sort_unstable_by(|a, b| b.1.cmp(&a.1).then(a.0.cmp(&b.0)));
        let mut out: Vec<usize> = Vec::with_capacity(self.topn);
        for (e, _) in ranked {
            out.push(e as usize);
            if out.len() >= self.topn {
                break;
            }
        }
        if out.len() < self.topn {
            if let Some(f) = self.freq.get(&target) {
                let mut fr: Vec<(u32, u32)> = f.iter().map(|(&e, &c)| (e, c)).collect();
                fr.sort_unstable_by(|a, b| b.1.cmp(&a.1).then(a.0.cmp(&b.0)));
                for (e, _) in fr {
                    let e = e as usize;
                    if !out.contains(&e) {
                        out.push(e);
                        if out.len() >= self.topn {
                            break;
                        }
                    }
                }
            }
        }
        out
    }
}

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
    /// per-(layer,eid) selections this session (feeds the persistent history)
    session_usage: HashMap<(usize, usize), u64>,
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
    /// Cache byte ceiling. Atomic because the adaptive-budget monitor
    /// ([`spawn_adaptive_budget`]) rewrites it live to track free RAM; the static
    /// path just sets it once. Read on every insert's eviction pass.
    budget: AtomicU64,
    state: Mutex<State>,
    /// Speculative-prefetch predictor + background-loader channel, present only
    /// when [`enable_prefetch`](ExpertCache::enable_prefetch) was called.
    predictor: Mutex<Option<Predictor>>,
    prefetch_tx: OnceLock<mpsc::SyncSender<(usize, Vec<usize>)>>,
}

impl<P: ExpertProvider> ExpertCache<P> {
    /// Wrap `inner` with a cache holding up to `budget_bytes` of experts. Use
    /// `u64::MAX` to never evict (hold everything that's ever loaded).
    pub fn new(inner: P, budget_bytes: u64) -> ExpertCache<P> {
        ExpertCache {
            inner,
            budget: AtomicU64::new(budget_bytes),
            state: Mutex::new(State {
                entries: HashMap::new(),
                pinned: HashSet::new(),
                bytes: 0,
                clock: 0,
                hits: 0,
                misses: 0,
                evictions: 0,
                session_usage: HashMap::new(),
            }),
            predictor: Mutex::new(None),
            prefetch_tx: OnceLock::new(),
        }
    }

    /// Pin `(layer, eid)` into the hot-store: once resident it is never evicted.
    /// Loads it now if absent. (Warm-up loads are not counted as usage.)
    pub fn pin(&self, layer: usize, eid: usize) -> io::Result<()> {
        self.fetch(layer, eid, false)?; // ensure resident
        self.state.lock().unwrap().pinned.insert((layer, eid));
        Ok(())
    }

    /// Warm the pinned hot-store from a usage history — the AUTOPIN startup step
    /// (`pin_load` in `c/glm.c`). Pins the globally hottest experts (by
    /// cumulative selection count) until `pin_budget_bytes` is reached. Returns
    /// how many were pinned. Warm-up loads do not count as session usage.
    pub fn warm_pin(&self, history: &UsageHistory, pin_budget_bytes: u64) -> io::Result<usize> {
        Ok(self.pin_ranked(&history.ranked(), pin_budget_bytes, usize::MAX)?.0)
    }

    /// Auto-sized AUTOPIN: pin the hot **head** of the usage curve — as many of the
    /// hottest experts as sit before the coverage curve's knee ([`UsageHistory::knee`])
    /// — instead of a hand-picked GB budget. Capped at ~80% of `cache_budget_bytes`
    /// so the cold tail still has room to stream through the LRU (pinning the whole
    /// cache would leave nothing evictable and thrash every miss). Returns
    /// `(n_pinned, bytes_pinned, coverage)` where `coverage` is the fraction of
    /// historical selections the pinned set accounts for.
    pub fn warm_pin_auto(
        &self,
        history: &UsageHistory,
        cache_budget_bytes: u64,
    ) -> io::Result<(usize, u64, f64)> {
        let ranked = history.ranked();
        let knee = history.knee().min(ranked.len());
        // Leave headroom for the streaming tail; guard against an unbounded budget.
        let byte_cap = (cache_budget_bytes / 5).saturating_mul(4); // 80%, overflow-safe
        let (n, bytes) = self.pin_ranked(&ranked, byte_cap, knee)?;
        Ok((n, bytes, history.coverage_of_top(n)))
    }

    /// Pin the first entries of `ranked` (hottest-first) until either `byte_cap`
    /// bytes or `count_cap` experts is reached, whichever comes first. Always pins
    /// at least the first entry (if any). Returns `(n_pinned, bytes_pinned)`.
    fn pin_ranked(
        &self,
        ranked: &[(usize, usize)],
        byte_cap: u64,
        count_cap: usize,
    ) -> io::Result<(usize, u64)> {
        let mut bytes = 0u64;
        let mut n = 0usize;
        for &(layer, eid) in ranked {
            if n >= count_cap {
                break;
            }
            let ex = self.fetch(layer, eid, false)?; // load resident, not a selection
            let b = ex.bytes();
            if n > 0 && bytes + b > byte_cap {
                break; // budget reached (the just-loaded one stays unpinned/LRU)
            }
            self.state.lock().unwrap().pinned.insert((layer, eid));
            bytes += b;
            n += 1;
        }
        Ok((n, bytes))
    }

    /// Snapshot this session's expert selections as a [`UsageHistory`], to merge
    /// into the persistent `.coli_usage` and save.
    pub fn usage_snapshot(&self) -> UsageHistory {
        let s = self.state.lock().unwrap();
        let mut h = UsageHistory::new();
        for (&(l, e), &c) in &s.session_usage {
            h.add(l, e, c);
        }
        h
    }

    /// Number of currently-pinned experts.
    pub fn pinned_count(&self) -> usize {
        self.state.lock().unwrap().pinned.len()
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
            budget: self.budget.load(Ordering::Relaxed),
        }
    }
}

impl State {
    /// Evict least-recently-used unpinned experts until at or under `budget`.
    ///
    /// Ranks with [`evict_score`] (recency primary) rather than `lfru_score`
    /// (frequency primary): prefill leaves a full cache of `heat = 2` residents and
    /// every decode load enters at `heat = 1`, so a frequency-primary rank evicts
    /// decode's live working set in favour of prefill leftovers that will never be
    /// read again. Measured 5.8% vs 44.8% decode hit rate.
    fn evict_to(&mut self, budget: u64) {
        self.evict_to_protecting(budget, &HashSet::new());
    }

    /// Like [`State::evict_to`] but never evicts a key in `protect` — used when
    /// bulk-inserting a layer's freshly-loaded batch, so the just-loaded experts
    /// (heat = 1, so "cold" to LFRU) survive to the compute loop instead of being
    /// evicted by their own batch and reloaded.
    fn evict_to_protecting(&mut self, budget: u64, protect: &HashSet<(usize, usize)>) {
        while self.bytes > budget {
            let clock = self.clock;
            let pinned = &self.pinned;
            let victim = self
                .entries
                .iter()
                .filter(|(k, _)| !pinned.contains(*k) && !protect.contains(*k))
                .min_by_key(|(_, e)| evict_score(e.heat, e.last, clock))
                .map(|(k, _)| *k);
            match victim {
                Some(k) => {
                    if let Some(e) = self.entries.remove(&k) {
                        self.bytes -= e.bytes;
                        self.evictions += 1;
                    }
                }
                None => break, // everything left is pinned or protected
            }
        }
    }
}

impl<P: ExpertProvider> ExpertCache<P> {
    /// Core cache access. `record` counts the access as a router selection in the
    /// session usage (true for real MoE routing, false for warm-up/pin loads).
    fn fetch(&self, layer: usize, eid: usize, record: bool) -> io::Result<Arc<Expert>> {
        let key = (layer, eid);
        // Fast path: resident hit.
        {
            let mut s = self.state.lock().unwrap();
            s.clock = s.clock.wrapping_add(1);
            let clock = s.clock;
            if record {
                *s.session_usage.entry(key).or_insert(0) += 1;
            }
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
        let budget = self.budget.load(Ordering::Relaxed);
        s.evict_to(budget);
        Ok(ex)
    }
}

impl<P: ExpertProvider + Sync> ExpertProvider for ExpertCache<P> {
    fn expert(&self, layer: usize, eid: usize) -> io::Result<Arc<Expert>> {
        self.fetch(layer, eid, true)
    }

    /// Disk→RAM for a layer's experts — the decode bottleneck once compute is on the
    /// GPU. Experts are loaded **serially**: each `inner.expert` read is chunked
    /// across cores internally (`Shards::pread_chunked`), so even a single-miss layer
    /// saturates the NVMe (which needs ~10 outstanding requests). Loading experts
    /// concurrently would only oversubscribe the already-saturated drive. Loads run
    /// **off the cache lock**; the batch is then inserted under one lock and evicted
    /// once while protecting itself. Preloads aren't router selections — the compute
    /// loop's `expert` call then hits and records the selection.
    fn prefetch(&self, layer: usize, eids: &[usize]) -> io::Result<()> {
        // Hand the *next* layer's experts to the background loader so they stream in
        // during this layer's compute. Two source modes:
        //   - PREFILL prefetch-ahead (COLI_PREFETCH_AHEAD): every layer routes to ~all
        //     experts, so queue exactly this layer's (large) set for layer+1 — an exact,
        //     not predicted, next-layer working set. The pipeline primes on layer 1 and
        //     every later load_batch is a cache hit, so the disk-load never sits on the
        //     critical path (it overlaps the GPU-bound attention + moe compute, when the
        //     NVMe is otherwise idle). Gated to the prefill regime by `eids.len()` so
        //     decode — where speculative loads evict the working set and steal demand
        //     bandwidth (measured net-negative) — is untouched.
        //   - Otherwise the learned predictor (decode / miss-heavy regime), if enabled.
        if let Some(tx) = self.prefetch_tx.get() {
            if prefetch_ahead_enabled() && eids.len() >= PREFETCH_AHEAD_MIN {
                let _ = tx.try_send((layer + 1, eids.to_vec()));
            } else {
                let predicted = self
                    .predictor
                    .lock()
                    .unwrap()
                    .as_mut()
                    .map(|p| p.observe_and_predict(layer, eids));
                if let Some(pred) = predicted {
                    if !pred.is_empty() {
                        let _ = tx.try_send((layer + 1, pred));
                    }
                }
            }
        }
        self.load_batch(layer, eids)
    }
}

/// Minimum routed-expert count for the prefill prefetch-ahead to fire — separates
/// prefill (routes to ~all `n_experts`) from decode (top-k per token, ~8).
const PREFETCH_AHEAD_MIN: usize = 64;

/// Prefill prefetch-ahead: during prefill, unconditionally background-load the next
/// layer's experts (they overlap the current layer's GPU compute). **On by default**
/// — measured token-identical and a clean prefill win on both models (GLM@4096 1.58×,
/// M3@512 1.26×; the hidden fraction grows with context). Set `COLI_PREFETCH_AHEAD=0`
/// to disable. Decode is never affected (gated by [`PREFETCH_AHEAD_MIN`]: a decode
/// step's per-layer union is ~top-k ≪ 64, so the ahead path never fires there).
fn prefetch_ahead_enabled() -> bool {
    static ON: OnceLock<bool> = OnceLock::new();
    *ON.get_or_init(|| std::env::var("COLI_PREFETCH_AHEAD").ok().as_deref() != Some("0"))
}

impl<P: ExpertProvider + Sync> ExpertCache<P> {
    /// Load `eids` for `layer` into the cache if absent (used by both `prefetch`
    /// and the background prefetch loader). Loads run **off the cache lock**; the
    /// batch is inserted under one lock and evicted once while protecting itself.
    /// Loads aren't router selections — the compute loop's `expert` call then hits
    /// and records the selection.
    fn load_batch(&self, layer: usize, eids: &[usize]) -> io::Result<()> {
        let missing: Vec<usize> = {
            let s = self.state.lock().unwrap();
            eids.iter()
                .copied()
                .filter(|&e| !s.entries.contains_key(&(layer, e)))
                .collect()
        };
        if missing.is_empty() {
            return Ok(());
        }

        // Load off the cache lock. The provider pools the whole batch through one
        // continuously-streaming reader by default (COLI_READER_POOL=0 disables);
        // on any batch error fall back to best-effort per-expert loads (a failure
        // otherwise surfaces when the compute loop calls `expert`).
        let loaded: Vec<(usize, Arc<Expert>)> = match self.inner.experts_batch(layer, &missing) {
            Ok(exps) if exps.len() == missing.len() => {
                missing.iter().copied().zip(exps).collect()
            }
            _ => {
                let mut v = Vec::with_capacity(missing.len());
                for &e in &missing {
                    if let Ok(ex) = self.inner.expert(layer, e) {
                        v.push((e, ex));
                    }
                }
                v
            }
        };

        // Serial bookkeeping: insert the batch, then a single protected eviction.
        let batch: HashSet<(usize, usize)> = missing.iter().map(|&e| (layer, e)).collect();
        let mut s = self.state.lock().unwrap();
        let clock = s.clock;
        for (e, ex) in loaded {
            let key = (layer, e);
            if s.entries.contains_key(&key) {
                continue;
            }
            let bytes = ex.bytes();
            s.entries.insert(key, Entry { expert: ex, bytes, heat: 1, last: clock });
            s.bytes += bytes;
            s.misses += 1;
        }
        let budget = self.budget.load(Ordering::Relaxed);
        s.evict_to_protecting(budget, &batch);
        Ok(())
    }
}

impl<P: ExpertProvider + Send + Sync + 'static> ExpertCache<P> {
    /// Turn on **speculative prefetch**: from each layer's routed experts, predict
    /// the next layer's and load them in the background (up to `topn`/layer) during
    /// this layer's compute, so a predicted expert is already resident when its
    /// layer runs. Best-effort — it never blocks the forward pass, only loads
    /// experts that aren't cached, and stops at the byte budget like any other load.
    ///
    /// **Off by default, and it should stay off when experts load from the local
    /// NVMe.** A controlled A/B on a Spark (GLM-5.2 int4, 20 GB cache, miss-heavy
    /// regime) regressed decode throughput at every degree — 1.01 tok/s off vs 0.99
    /// (top-2), 0.93 (top-4), 0.82 (top-16) — because (1) speculative loads evict
    /// working-set experts the model still needs (misses climb from 15k to 37k), and
    /// (2) the background loader steals bandwidth from demand reads on an
    /// already-saturated drive (expert-load time rises 29→34 s). Prediction accuracy
    /// isn't the bottleneck; you can't hide loads behind the drive that *is* the
    /// bottleneck. This machinery earns its keep only when the prefetch **source** is
    /// a peer's RAM over RDMA (multispark) rather than local disk — no drive
    /// contention there — or with a separate staging budget that can't evict the
    /// working set. Kept opt-in for that. See `scripts/expert_prefetch_analysis.py`.
    pub fn enable_prefetch(self: &Arc<Self>, topn: usize) {
        *self.predictor.lock().unwrap() = Some(Predictor::new(topn));
        let (tx, rx) = mpsc::sync_channel::<(usize, Vec<usize>)>(4);
        if self.prefetch_tx.set(tx).is_err() {
            return; // already enabled
        }
        let cache = Arc::clone(self);
        std::thread::spawn(move || {
            for (layer, eids) in rx {
                let _ = cache.load_batch(layer, &eids);
            }
        });
    }

    /// Make the byte ceiling **adaptive**: fill RAM up to `fill_target` and hold it,
    /// ceding memory (evicting LRU experts) only when a *different* tenant creates
    /// genuine, sustained pressure — `MemAvailable` staying under `danger_floor`.
    ///
    /// `fill_target` is the standing budget: `total − reserve`, where the reserve is
    /// sized for the serving process's own peak runtime (KV cache, prefill activations,
    /// GPU host staging, expert read buffers). The cache grows to it through the insert
    /// path; we set it once here.
    ///
    /// Why we do NOT track `MemAvailable` symmetrically (the old `budget = resident +
    /// (avail − floor)` law): once RAM is filled, `avail` sits low *by design*, and every
    /// request transiently allocates activations/staging on top for its whole duration
    /// (seconds). A law that shrinks on any dip below the floor evicts the exact working
    /// set it just made resident — and with fadvise on, each false eviction is a cold
    /// re-read, so throughput *degrades* under sustained diverse load (measured: median
    /// 2.06 vs the static-budget 4.35 tok/s on M2.7). The correct separation: the reserve
    /// (baked into `fill_target`) absorbs our own request runtime so `avail` never reaches
    /// `danger_floor` on our account; only an external tenant pushes it there, and only
    /// then — after `SUSTAIN` ticks, so a momentary blip is ignored — do we evict, down to
    /// just off the swap line. When the pressure clears we reclaim back up to `fill_target`.
    ///
    /// SAFETY: `MemAvailable` is only honest once expert reads don't double-hold in the
    /// page cache (fadvise or O_DIRECT). Enable this only alongside one of those. Off-Linux
    /// (no `/proc/meminfo`) it no-ops after setting the standing budget.
    pub fn spawn_adaptive_budget(self: &Arc<Self>, fill_target: u64, danger_floor: u64) {
        const TICK_MS: u64 = 250;
        const SUSTAIN: u32 = 3; // ~750 ms below the danger floor before we cede memory
        const FLOOR_MIN: u64 = 2 << 30; // never target < 2 GiB resident
        let fill_target = fill_target.max(FLOOR_MIN);
        let danger_floor = danger_floor.max(FLOOR_MIN);
        let cache = Arc::clone(self);
        // Grow to the standing fill target immediately (the insert path enforces it).
        cache.budget.store(fill_target, Ordering::Relaxed);
        std::thread::spawn(move || {
            let mut low_ticks: u32 = 0;
            loop {
                std::thread::sleep(std::time::Duration::from_millis(TICK_MS));
                let avail = match available_ram_bytes() {
                    Some(a) => a,
                    None => return, // non-Linux: no live signal, keep the standing budget
                };
                low_ticks = if avail < danger_floor {
                    low_ticks.saturating_add(1)
                } else {
                    0
                };
                let resident = cache.state.lock().unwrap().bytes;
                let new_budget = if low_ticks >= SUSTAIN {
                    // Sustained external pressure: cede RAM to just clear the swap line.
                    resident.saturating_sub(danger_floor - avail).max(FLOOR_MIN)
                } else {
                    // Normal (incl. our own transient request spikes): hold the target.
                    fill_target
                };
                cache.budget.store(new_budget, Ordering::Relaxed);
                if new_budget < resident {
                    cache.state.lock().unwrap().evict_to(new_budget);
                }
            }
        });
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

    /// Compressed MLA KV-cache bytes per token — exactly what `KvCache`
    /// allocates: every one of `n_layers` attention layers caches a normalized
    /// latent (`kv_lora` floats) and a roped key (`qk_rope` floats) per token.
    /// (The DSA indexer, if enabled, adds a little more; not counted here.)
    pub fn kv_bytes_per_token(kv_lora: u64, qk_rope: u64, n_layers: u64) -> u64 {
        (kv_lora + qk_rope) * 4 * n_layers
    }

    /// Max context (tokens) whose KV cache fits in `budget_bytes`.
    pub fn context_in_kv_budget(budget_bytes: u64, kv_bytes_per_token: u64) -> u64 {
        if kv_bytes_per_token == 0 {
            0
        } else {
            budget_bytes / kv_bytes_per_token
        }
    }
}

/// Available RAM in bytes, best-effort. Reads `/proc/meminfo` `MemAvailable` on
/// Linux (the DGX Spark target); returns `None` elsewhere (e.g. macOS dev boxes),
/// where the caller should fall back to an explicit budget.
pub fn available_ram_bytes() -> Option<u64> {
    meminfo_field("MemAvailable:")
}

/// Total RAM in bytes, best-effort (`/proc/meminfo` `MemTotal`).
///
/// Distinct from [`available_ram_bytes`] on purpose: `MemAvailable` counts reclaimable
/// page cache as free, so budgeting from it hands the expert cache memory the kernel
/// is *already using* to cache the model file — and the cache then pages itself out.
/// The safe ceiling scales with the size of the machine, which only `MemTotal` knows.
pub fn total_ram_bytes() -> Option<u64> {
    meminfo_field("MemTotal:")
}

fn meminfo_field(key: &str) -> Option<u64> {
    let meminfo = std::fs::read_to_string("/proc/meminfo").ok()?;
    for line in meminfo.lines() {
        if let Some(rest) = line.strip_prefix(key) {
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

    #[test]
    fn predictor_learns_layer_transition() {
        let mut p = Predictor::new(4);
        // Teach it twice: at layer 1 expert 10 is followed by expert 20 at layer 2.
        for _ in 0..2 {
            p.observe_and_predict(1, &[10]);
            p.observe_and_predict(2, &[20]);
        }
        // Now, seeing expert 10 at layer 1, it should predict 20 for layer 2.
        let pred = p.observe_and_predict(1, &[10]);
        assert_eq!(pred.first(), Some(&20), "predicted {pred:?}");
    }

    #[test]
    fn predictor_backfills_with_frequency() {
        let mut p = Predictor::new(3);
        // No transitions into layer 5 learned, but layer 5 saw expert 7 often.
        for _ in 0..3 {
            p.observe_and_predict(5, &[7, 8]);
        }
        // Predicting layer 5 from an unknown context falls back to frequency (7, 8).
        let pred = p.predict(5, &[999]);
        assert!(pred.contains(&7) && pred.contains(&8), "predicted {pred:?}");
    }

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
    fn warm_pin_pins_hottest_within_budget() {
        // History: expert (0,2) hottest, then (0,1), then (0,0).
        let mut h = UsageHistory::new();
        h.add(0, 0, 1);
        h.add(0, 1, 10);
        h.add(0, 2, 100);
        let one = {
            let c = ExpertCache::new(counting(), u64::MAX);
            c.expert(0, 0).unwrap().bytes()
        };
        let cache = ExpertCache::new(counting(), u64::MAX);
        // budget for exactly 2 experts -> pin the two hottest: (0,2) and (0,1).
        let pinned = cache.warm_pin(&h, one * 2).unwrap();
        assert_eq!(pinned, 2);
        assert_eq!(cache.pinned_count(), 2);
        // warm-up loads must NOT count as session usage
        assert_eq!(cache.usage_snapshot().total(), 0);

        // now churn other experts under a tight budget; the pinned two survive.
        let cache = ExpertCache::new(counting(), one * 3);
        cache.warm_pin(&h, one * 2).unwrap(); // pin (0,2),(0,1)
        for e in 3..8 {
            cache.expert(0, e).unwrap(); // real selections, evictable
        }
        // pinned experts still resident: accessing them is a hit (no reload)
        let before = cache.inner.loads.load(Ordering::Relaxed);
        cache.expert(0, 2).unwrap();
        cache.expert(0, 1).unwrap();
        assert_eq!(cache.inner.loads.load(Ordering::Relaxed), before, "pinned reloaded");
    }

    #[test]
    fn warm_pin_auto_pins_the_hot_head() {
        // 4 hot experts then a flat tail: auto should pin ~the head, not the tail,
        // and report a coverage well above the pinned fraction.
        let mut h = UsageHistory::new();
        for e in 0..4 {
            h.add(0, e, 100);
        }
        for e in 4..60 {
            h.add(0, e, 1);
        }
        let cache = ExpertCache::new(counting(), u64::MAX);
        let (n, bytes, cov) = cache.warm_pin_auto(&h, u64::MAX).unwrap();
        assert_eq!(cache.pinned_count(), n);
        assert!((4..=12).contains(&n), "auto pinned {n}, expected the ~4 hot head");
        assert!(bytes > 0);
        assert!(cov > 0.8, "coverage {cov} should capture the hot head's traffic");
        assert_eq!(cache.usage_snapshot().total(), 0, "warm-up isn't session usage");
    }

    #[test]
    fn warm_pin_auto_respects_cache_headroom() {
        // With a tiny cache budget, auto must not pin the whole thing — it caps at
        // ~80% so the streaming tail keeps room. Budget for 5 experts -> <=4 pinned.
        let mut h = UsageHistory::new();
        for e in 0..20 {
            h.add(0, e, 100 - e as u64); // gently decreasing, knee is late
        }
        let one = {
            let c = ExpertCache::new(counting(), u64::MAX);
            c.expert(0, 0).unwrap().bytes()
        };
        let cache = ExpertCache::new(counting(), one * 5);
        let (n, bytes, _cov) = cache.warm_pin_auto(&h, one * 5).unwrap();
        assert!(n <= 4, "pinned {n}, must leave headroom below the 5-expert budget");
        assert!(bytes <= one * 4);
    }

    #[test]
    fn session_usage_tracks_selections() {
        let cache = ExpertCache::new(counting(), u64::MAX);
        cache.expert(3, 5).unwrap();
        cache.expert(3, 5).unwrap();
        cache.expert(3, 7).unwrap();
        let u = cache.usage_snapshot();
        assert_eq!(u.get(3, 5), 2);
        assert_eq!(u.get(3, 7), 1);
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
