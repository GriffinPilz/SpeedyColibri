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
        Predictor {
            topn,
            freq: HashMap::new(),
            trans: HashMap::new(),
            last: None,
        }
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
        // Bound-check BEFORE pushing. Push-then-check emitted one expert even at
        // `topn == 0` — and `topn == 0` is exactly what `prefetch_topn()` passes to wire
        // the background loader with what it documents as "a no-op predictor" (the
        // prefill prefetch-ahead needs the loader thread even when COLI_PREFETCH is off).
        // So the no-op predictor was speculatively loading one expert per layer: measured
        // on K3 as 80 wasted expert loads (~1.4 GB) per run against an otherwise
        // identical arm.
        let mut out: Vec<usize> = Vec::with_capacity(self.topn);
        for (e, _) in ranked {
            if out.len() >= self.topn {
                break;
            }
            out.push(e as usize);
        }
        if out.len() < self.topn {
            if let Some(f) = self.freq.get(&target) {
                let mut fr: Vec<(u32, u32)> = f.iter().map(|(&e, &c)| (e, c)).collect();
                fr.sort_unstable_by(|a, b| b.1.cmp(&a.1).then(a.0.cmp(&b.0)));
                for (e, _) in fr {
                    if out.len() >= self.topn {
                        break;
                    }
                    let e = e as usize;
                    if !out.contains(&e) {
                        out.push(e);
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
    /// Standing fill target the monitor grows toward (`0` = unmanaged). Held so
    /// [`reserve_ram`](ExpertCache::reserve_ram) and the monitor agree on the ceiling.
    fill_target: AtomicU64,
    /// Hard OOM-guard line (`MemAvailable` must stay above this); shared with
    /// [`reserve_ram`](ExpertCache::reserve_ram) so a KV reservation leaves the same margin.
    hard_floor: AtomicU64,
    /// RAM (bytes) reserved by callers for non-expert allocations about to happen —
    /// chiefly a request's KV cache. The monitor holds experts to `fill_target − reserved`,
    /// and [`reserve_ram`](ExpertCache::reserve_ram) evicts to free it up front.
    reserved: AtomicU64,
    state: Mutex<State>,
    /// Speculative-prefetch predictor + background-loader channel, present only
    /// when [`enable_prefetch`](ExpertCache::enable_prefetch) was called.
    predictor: Mutex<Option<Predictor>>,
    prefetch_tx: OnceLock<mpsc::SyncSender<(usize, Vec<usize>)>>,
    /// The model's routed-expert count, so the prefill prefetch-ahead can judge what
    /// *fraction* of the experts a layer touched. `0` = not supplied (legacy behaviour:
    /// the absolute [`PREFETCH_AHEAD_MIN`] gate alone). See [`ahead_predicts_next_layer`].
    n_experts: AtomicU64,
}

impl<P: ExpertProvider> ExpertCache<P> {
    /// Wrap `inner` with a cache holding up to `budget_bytes` of experts. Use
    /// `u64::MAX` to never evict (hold everything that's ever loaded).
    pub fn new(inner: P, budget_bytes: u64) -> ExpertCache<P> {
        ExpertCache {
            inner,
            budget: AtomicU64::new(budget_bytes),
            fill_target: AtomicU64::new(0),
            hard_floor: AtomicU64::new(0),
            reserved: AtomicU64::new(0),
            n_experts: AtomicU64::new(0),
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
        Ok(self
            .pin_ranked(&history.ranked(), pin_budget_bytes, usize::MAX)?
            .0)
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
        if self.bytes <= budget {
            self.publish_ram();
            return;
        }
        // Rank once, then evict down the list — rather than re-scanning every entry to
        // find each successive victim.
        //
        // This is not an approximation. `evict_score(heat, last, clock)` reads only fields
        // of the entry itself, and eviction mutates none of them: removing one entry
        // leaves every other entry's score unchanged, and `pinned`/`protect`/`clock` are
        // fixed for the call. So "repeatedly take the minimum" and "take them in ascending
        // score order" select the same victims, in the same order. Only ties can resolve
        // differently, and those were already decided by `HashMap` iteration order, i.e.
        // arbitrary. Verified against the old implementation in
        // `batch_eviction_picks_the_same_victims_as_repeated_min`.
        //
        // The old shape was O(entries) per victim. Measured on GLM (COLI_PROFILE=1): 3488
        // evictions against ~2051 resident entries cost **1839 ms**, 87% of everything
        // expert-load spent outside the reader and 13% of expert-load itself — with the
        // drive idle throughout, because this runs under `state.lock()` between batches.
        //
        // This is O(N log N) against the old O(M*N), so for **one** victim against a huge
        // cache the asymptotics favour the old shape — worth noting, because Nemotron
        // preloads 20480 experts. The constants should still favour this: the old loop
        // measured ~237 ns per entry visited (1698 ms / 3488 / 2051 — HashMap iteration,
        // pointer-chasing into Arc'd entries, two tuple hashes for pinned/protect),
        // whereas this collects and sorts a flat Vec of Copy tuples. That comparison is
        // *reasoned, not measured*: the small-M/large-N regime has not been profiled, and
        // Nemotron in practice preloads to fit and evicts ~nothing, so it is not currently
        // exercised. Measure before assuming it holds if that changes.
        let t_select = crate::forward::profile_on().then(std::time::Instant::now);
        let clock = self.clock;
        let pinned = &self.pinned;
        let mut victims: Vec<((usize, usize), u64, u64)> = self
            .entries
            .iter()
            .filter(|(k, _)| !pinned.contains(*k) && !protect.contains(*k))
            .map(|(k, e)| (*k, evict_score(e.heat, e.last, clock), e.bytes))
            .collect();
        victims.sort_unstable_by_key(|&(_, score, _)| score);
        if let Some(t) = t_select {
            EVICT_SELECT_US.fetch_add(t.elapsed().as_micros() as u64, Ordering::Relaxed);
        }
        // Move victims out of the map WITHOUT dropping them here, so the ranking cost and
        // the deallocation cost are separable. Rewriting the O(entries)-per-victim scan as
        // a single ranked pass did not move GLM's `evict` at all (1698-1839 ms before,
        // 2003 ms after) — so the scan was never what this function spends its time on,
        // and the next candidate is freeing ~21 MB of expert buffers per victim. Measure
        // which, rather than guess a third time.
        let t_drop = crate::forward::profile_on().then(std::time::Instant::now);
        let mut freed: Vec<Entry> = Vec::new();
        for (k, _, bytes) in victims {
            if self.bytes <= budget {
                break;
            }
            if let Some(e) = self.entries.remove(&k) {
                self.bytes -= bytes;
                self.evictions += 1;
                freed.push(e);
            }
        }
        drop(freed);
        if let Some(t) = t_drop {
            EVICT_DROP_US.fetch_add(t.elapsed().as_micros() as u64, Ordering::Relaxed);
        }
        self.publish_ram();
        // Falling off the end means everything left is pinned or protected — the same
        // outcome the old `None => break` arm produced.
    }

    /// Publish resident expert bytes to the RAM ledger.
    ///
    /// Both insert paths call `evict_to`/`evict_to_protecting` immediately after adding to
    /// `self.bytes`, and the adaptive monitor calls them directly, so publishing on **every**
    /// exit of `evict_to_protecting` — including the early "already fits" return, which is
    /// the common case after an insert — covers every mutation of `self.bytes` without
    /// scattering calls across the file.
    ///
    /// Before this, `Class::Experts` was never committed. The cache bounded itself against
    /// its own `budget` while `serve`'s KV admission computed headroom as
    /// `ceiling - dense - experts` with `experts` stuck at 0 — so the two largest consumers
    /// on the box each sized themselves as if the other were absent. That is the accounting
    /// hole behind "RAM must never pass 96%", not the ceiling arithmetic, which was correct.
    fn publish_ram(&self) {
        crate::ram::set_usage(crate::ram::Class::Experts, self.bytes);
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
            if prefetch_ahead_enabled()
                && ahead_predicts_next_layer(eids.len(), self.n_experts.load(Ordering::Relaxed))
            {
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

/// Phases of [`ExpertCache::load_batch`], µs, under `COLI_PROFILE=1`. Together with the
/// reader's own span-setup/drain/post these fully account for `expert-load`, so the
/// drive-idle portion of that window can be attributed to a phase instead of inferred.
///
/// `FETCH` brackets `experts_batch` — it *contains* the reader's time, so the
/// interesting quantity is `FETCH - (setup+drain+post)`: per-expert construction on top
/// of the bytes. `EVICT` is the one to watch structurally: it is O(entries) per victim.
/// `EVICT` split in two: ranking the victims versus actually freeing them.
///
/// Needed because rewriting the ranking from O(entries)-per-victim to a single sorted
/// pass changed GLM's `evict` by nothing at all (1698-1839 ms before, 2003 ms after).
/// A rewrite that provably picks the same victims and provably does asymptotically less
/// work, with zero effect, means the work was never in the part that was rewritten.
pub(crate) static EVICT_SELECT_US: AtomicU64 = AtomicU64::new(0);
pub(crate) static EVICT_DROP_US: AtomicU64 = AtomicU64::new(0);

pub(crate) static CACHE_FILTER_US: AtomicU64 = AtomicU64::new(0);
pub(crate) static CACHE_FETCH_US: AtomicU64 = AtomicU64::new(0);
pub(crate) static CACHE_INSERT_US: AtomicU64 = AtomicU64::new(0);
pub(crate) static CACHE_EVICT_US: AtomicU64 = AtomicU64::new(0);

/// Minimum routed-expert count for the prefill prefetch-ahead to fire — separates
/// prefill (routes to ~all `n_experts`) from decode (top-k per token, ~8).
const PREFETCH_AHEAD_MIN: usize = 64;

/// Minimum *percentage* of the model's experts a layer must touch before
/// prefetch-ahead will fire. See [`ahead_predicts_next_layer`].
const PREFETCH_AHEAD_MIN_PCT: u64 = 50;

/// Should prefetch-ahead queue this layer's expert set as the prediction for the next
/// layer?
///
/// Prefetch-ahead predicts layer L+1's working set by reusing layer L's expert **ids**.
/// That is only sound when a layer routes to ~*all* experts — then "the same ids" and
/// "everything" are the same set and the prediction cannot miss. An absolute count
/// (`>= 64`) is a poor proxy for that, because what matters is the count relative to
/// `n_experts`:
///
/// | model | experts | union/layer | fraction | prefetch-ahead |
/// |---|---|---|---|---|
/// | GLM @4096 | 160 | ~all | ~100% | **1.58× win** |
/// | Kimi-K3 @5 tok | 896 | ~78 | 8.7% | **1.26× loss** |
///
/// K3 cleared the old absolute gate (78 ≥ 64) while predicting almost nothing: measured
/// ABBA on 42b2, prefetch-ahead cost **2293 extra expert loads (~50 GiB)** and bought
/// **2 additional cache hits** (7147→7149), for expert-load 12926→16277 ms. The
/// fraction gate rejects that case and still admits GLM's. It is also context-adaptive
/// in the right direction: the same K3 at long context routes to ~all 896 experts, and
/// there the prediction becomes correct and the gate re-opens on its own.
///
/// The 50% threshold is bounded by those two measured anchors (8.7% harmful, ~100%
/// helpful); the crossover between them is **not** measured.
fn ahead_predicts_next_layer(n_routed: usize, n_experts: u64) -> bool {
    // Decode (per-token top-k ≪ 64) never prefetches ahead: speculative loads evict the
    // working set and steal demand bandwidth (measured net-negative).
    if n_routed < PREFETCH_AHEAD_MIN {
        return false;
    }
    // Expert count not supplied — keep the historical absolute-gate behaviour.
    if n_experts == 0 {
        return true;
    }
    n_routed as u64 * 100 >= n_experts * PREFETCH_AHEAD_MIN_PCT
}

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
        // Phase timers (COLI_PROFILE=1 only). `expert-load` minus the reader's own
        // setup/drain/post leaves ~2.1 s unaccounted on a 14.4 s GLM run — time with
        // the drive completely idle. Guessing put that on eviction; measure it instead.
        let prof = crate::forward::profile_on();
        let t = std::time::Instant::now();
        let missing: Vec<usize> = {
            let s = self.state.lock().unwrap();
            eids.iter()
                .copied()
                .filter(|&e| !s.entries.contains_key(&(layer, e)))
                .collect()
        };
        if prof {
            CACHE_FILTER_US.fetch_add(t.elapsed().as_micros() as u64, Ordering::Relaxed);
        }
        if missing.is_empty() {
            return Ok(());
        }

        // Load off the cache lock. The provider pools the whole batch through one
        // continuously-streaming reader by default (COLI_READER_POOL=0 disables);
        // on any batch error fall back to best-effort per-expert loads (a failure
        // otherwise surfaces when the compute loop calls `expert`).
        let t = std::time::Instant::now();
        let loaded: Vec<(usize, Arc<Expert>)> = match self.inner.experts_batch(layer, &missing) {
            Ok(exps) if exps.len() == missing.len() => missing.iter().copied().zip(exps).collect(),
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
        if prof {
            CACHE_FETCH_US.fetch_add(t.elapsed().as_micros() as u64, Ordering::Relaxed);
        }

        // Serial bookkeeping: insert the batch, then a single protected eviction.
        let t = std::time::Instant::now();
        let batch: HashSet<(usize, usize)> = missing.iter().map(|&e| (layer, e)).collect();
        let mut s = self.state.lock().unwrap();
        let clock = s.clock;
        for (e, ex) in loaded {
            let key = (layer, e);
            if s.entries.contains_key(&key) {
                continue;
            }
            let bytes = ex.bytes();
            s.entries.insert(
                key,
                Entry {
                    expert: ex,
                    bytes,
                    heat: 1,
                    last: clock,
                },
            );
            s.bytes += bytes;
            s.misses += 1;
        }
        if prof {
            CACHE_INSERT_US.fetch_add(t.elapsed().as_micros() as u64, Ordering::Relaxed);
        }
        let t = std::time::Instant::now();
        let budget = self.budget.load(Ordering::Relaxed);
        s.evict_to_protecting(budget, &batch);
        if prof {
            CACHE_EVICT_US.fetch_add(t.elapsed().as_micros() as u64, Ordering::Relaxed);
        }
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
    /// `n_experts` is the model's routed-expert count, used by the prefill
    /// prefetch-ahead to judge what fraction of the experts a layer touched
    /// ([`ahead_predicts_next_layer`]). Pass `0` if unknown.
    pub fn enable_prefetch(self: &Arc<Self>, topn: usize, n_experts: u64) {
        self.n_experts.store(n_experts, Ordering::Relaxed);
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

    /// Manage the byte ceiling to **fill RAM safely**: grow toward `fill_target`, but
    /// continuously evict LRU experts so `MemAvailable` never crosses `hard_floor`. This
    /// runs for **every** model and every budget (near-fit *or* ≫ RAM) — the eviction is
    /// what makes filling RAM safe: a cache that gives memory back under pressure cannot
    /// OOM, so there is no model too large to point it at. `fill_target` is aspirational
    /// (fill RAM); `hard_floor` is the real guarantee (never touch the last few GB).
    ///
    /// Two thresholds, deliberately separated:
    /// - **`hard_floor`** — the OOM guard. Checked every tick with **no hysteresis**: if
    ///   `MemAvailable` is below it (our own growth *or* another tenant, incl. the GPU on
    ///   GB10's unified pool), evict immediately, down to a few GB of slack above it. This
    ///   is what a fixed manual budget lacked — a static number with no feedback grows into
    ///   the wall (measured: forcing 100 GB on the 216 GB M3 drove avail→0 and OOM-killed
    ///   the server). With this, that same budget just caps itself where the box stays safe.
    /// - **`danger_floor`** (< `hard_floor` is wrong; it sits *above* `hard_floor`) — the
    ///   soft line for a *sustained external* tenant. Only after `SUSTAIN` ticks below it do
    ///   we cede gradually, so a momentary dip (our own request's activation/staging spike)
    ///   is ignored and we don't churn the resident near-fit working set (that symmetric
    ///   `budget = resident + (avail − floor)` law regressed M2.7 to 2.06 vs 4.35 tok/s).
    ///
    /// The insert path also enforces `budget`, so between ticks the cache never grows past
    /// the last value we set. Fast `TICK_MS` keeps the reaction window small. Off-Linux
    /// (no `/proc/meminfo`) it no-ops after setting the standing budget.
    pub fn spawn_adaptive_budget(
        self: &Arc<Self>,
        fill_target: u64,
        danger_floor: u64,
        hard_floor: u64,
    ) {
        const TICK_MS: u64 = 100; // poll ~2.5× faster than before — react before OOM, not after
        const SUSTAIN: u32 = 6; // ~600 ms below the danger floor before we cede memory
        const HARD_SLACK: u64 = 3 << 30; // when the OOM guard fires, evict back to this much headroom
        const FLOOR_MIN: u64 = 2 << 30; // never target < 2 GiB resident
                                        // Slack above the startup swap level before the guard fires: absorbs the kernel
                                        // paging out genuinely cold pages (a long-idle allocation) without treating it as
                                        // expert-cache pressure. Anything beyond this is us.
        const SWAP_TOLERANCE: u64 = 256 << 20;
        let fill_target = fill_target.max(FLOOR_MIN);
        // hard_floor is the emergency line; danger_floor the softer one above it.
        let hard_floor = hard_floor.max(FLOOR_MIN);
        let danger_floor = danger_floor.max(hard_floor);
        // Publish the ceiling + floor so `reserve_ram` agrees with the monitor.
        self.fill_target.store(fill_target, Ordering::Relaxed);
        self.hard_floor.store(hard_floor, Ordering::Relaxed);
        let cache = Arc::clone(self);
        // Grow to the standing fill target immediately (the insert path enforces it).
        cache.budget.store(fill_target, Ordering::Relaxed);
        // Swap in use at startup is someone else's; only growth beyond it is ours.
        let swap_baseline = swap_used_bytes().unwrap_or(0);
        std::thread::spawn(move || {
            let mut low_ticks: u32 = 0;
            // Last tick's `MemAvailable`, so the guard can see how fast memory is going.
            // Seeded with the current value: the first tick then measures a drop of 0 and
            // the floor starts at `hard_floor`, which is what a still cache deserves.
            let mut prev_avail = available_ram_bytes().unwrap_or(u64::MAX);
            // Wall-clock of the previous tick. The monitor competes with ~20 reader threads
            // and the prefill for 20 cores, so "the guard did not brake" has two very
            // different causes — it decided not to, or it never ran. `COLI_GUARD_TRACE=1`
            // prints both, with the actual gap between ticks, so they can be told apart.
            let trace = std::env::var("COLI_GUARD_TRACE").map(|v| v != "0").unwrap_or(false);
            let mut prev_tick = std::time::Instant::now();
            // What we have LEARNED this box will actually hold, which is not what the
            // load-time plan predicted. `fill_target` is an estimate built from the dense
            // tier and a flat runtime reserve; on K3 the dense tier alone came in 4 GB over
            // its estimate. This is the measured answer, and it only ever moves in response
            // to real memory pressure. AIMD: the guard cuts it hard, the tick below earns
            // it back slowly.
            //
            // SEEDED FROM MEASUREMENT, NOT FROM THE PLAN. `fill_target` is computed before
            // the dense tier is loaded, from an *estimate* of it; by the time this thread
            // starts, the real number is observable. On K3 that estimate was 4 GB light
            // (54 GiB planned, 62.06 GB actual), and seeding the cap at the planned 56.10 GB
            // let the cache reach it — RSS 121.6 GB on a 130.66 GB box — before the guard
            // fired even once, so the AIMD below never got to engage at all. Whatever is
            // genuinely free right now, less room to brake in, is the honest ceiling; the
            // plan remains the upper bound, so a model with real headroom is unaffected.
            let mut learned_cap = match available_ram_bytes() {
                Some(a) => fill_target.min(supported_cap(0, a, hard_floor).max(FLOOR_MIN)),
                None => fill_target,
            };
            loop {
                std::thread::sleep(std::time::Duration::from_millis(TICK_MS));
                let avail = match available_ram_bytes() {
                    Some(a) => a,
                    None => return, // non-Linux: no live signal, keep the standing budget
                };
                let gap_ms = prev_tick.elapsed().as_millis() as u64;
                prev_tick = std::time::Instant::now();
                // Sample the consumption rate HERE, before any early `continue`. The swap
                // guard below returns to the top of the loop without reaching the bottom,
                // so updating `prev_avail` down there left it stale across every
                // swap-guard tick: the next `drop_per_tick` then spanned two or more ticks
                // and overstated the rate, inflating `braking_floor` — and with the swap
                // guard firing repeatedly it would pin the brake at MAX_BRAKE and evict far
                // more than needed. Conservative direction, so it could never have caused
                // an OOM, which is exactly why it would have gone unnoticed.
                let drop_per_tick = prev_avail.saturating_sub(avail);
                prev_avail = avail;
                // Keep the ledger's read-buffer figure current. It was charged in exactly
                // one place — inside forward.rs's `[profile]` print — i.e. once, at the end
                // of a run, so for the whole run the ledger believed ReadBuf was 0 while it
                // was really GBs (GLM 5.9, M2.7 1.7). Every consumer of `headroom()` was
                // that much optimistic, notably gpu.rs sizing its staging chunk as
                // `headroom() / 4`. This tick already runs every 100 ms and already has the
                // dependency, so currency costs one atomic swap.
                crate::ram::set_usage(crate::ram::Class::ReadBuf, colibri_core::pool_live_bytes());
                // Same treatment for `Class::Scratch`, which was worse off: measured 0.0 GB
                // on ALL FIVE models. Its only charge site (gpu.rs) is a *prediction* made
                // on the grouped NVFP4 path, and that path does not fire on a plain `gen`,
                // so the class was pure dead accounting while the CUDA context held
                // 0.17-2.30 GB of real LPDDR5X (M3 largest). On GB10 "VRAM" is the same
                // pool the ledger is budgeting, so this is not a bookkeeping nicety.
                //
                // NOTE this does NOT change the cache ceiling: `supported_cap` works off
                // MemAvailable, which already reflects every cudaMalloc. What it fixes is
                // `headroom()`, whose real consumer is gpu.rs sizing its staging chunk as
                // `headroom() / 4` — that was optimistic by up to 2.30 GB.
                //
                // The gpu.rs prediction is left in place: `set_usage` is absolute, so this
                // supersedes it within one tick, which is strictly better than a prediction
                // that is never cleared.
                #[cfg(feature = "cuda")]
                crate::ram::set_usage(
                    crate::ram::Class::Scratch,
                    colibri_backend::cuda::scratch_bytes(),
                );
                let resident = cache.state.lock().unwrap().bytes;
                if trace {
                    eprintln!(
                        "[guard] gap={gap_ms}ms avail={:.2} GB drop={:.2} GB cache={:.2} GB \
                         budget={:.2} GB cap={:.2} GB reserved={:.2} GB supported={:.2} GB",
                        avail as f64 / 1e9,
                        // `drop_per_tick`, not `prev_avail - avail` — `prev_avail` is
                        // updated above this point now, so recomputing it here would print
                        // 0.00 every tick and quietly blind the one instrument that showed
                        // the fill rate in the first place.
                        drop_per_tick as f64 / 1e9,
                        resident as f64 / 1e9,
                        cache.budget.load(Ordering::Relaxed) as f64 / 1e9,
                        learned_cap as f64 / 1e9,
                        // `reserved` and `supported` are the two terms the serve regression
                        // hypothesis is about: if the same KV is subtracted twice, `cap`
                        // sits ~`reserved` below `supported` while memory is not tight.
                        cache.reserved.load(Ordering::Relaxed) as f64 / 1e9,
                        supported_cap(resident, avail, hard_floor) as f64 / 1e9,
                    );
                }

                // SWAP GUARD — checked before the MemAvailable floors, because it fires
                // first. Measured on 42b2: a 110 GB fill put 3 GB into swap while
                // MemAvailable still read 30 GB, so neither floor below ever tripped and
                // throughput fell to 0.06-0.24 tok/s. Once the kernel is paging us out,
                // resident experts are actively harmful — a faulted page is a 4 KiB disk
                // read where a cache miss would have been one coalesced expert read.
                // Give memory back hard and let the standing budget re-grow only if the
                // pressure was transient.
                if let Some(sw) = swap_used_bytes() {
                    if sw > swap_baseline + SWAP_TOLERANCE {
                        let new_budget = resident
                            .saturating_sub(resident / 4)
                            .min(cache.budget.load(Ordering::Relaxed))
                            .max(FLOOR_MIN);
                        cache.budget.store(new_budget, Ordering::Relaxed);
                        cache.state.lock().unwrap().evict_to(new_budget);
                        low_ticks = 0;
                        continue;
                    }
                }

                // OOM guard (immediate, no hysteresis): never let avail cross the floor,
                // whatever ate the memory. Evict back to a few GB of slack above it.
                //
                // The floor is *rate-aware* — see `braking_floor`. A static `hard_floor`
                // is only safe if the guard can stop within the margin between it and
                // whatever kills us; on Kimi-K3 one tick of fill was larger than that
                // entire margin, so the guard lost every time.
                let floor_now = braking_floor(hard_floor, drop_per_tick);
                if avail < floor_now {
                    let reclaim = (floor_now - avail).saturating_add(HARD_SLACK);
                    let new_budget = resident.saturating_sub(reclaim).max(FLOOR_MIN);
                    cache.budget.store(new_budget, Ordering::Relaxed);
                    let after = {
                        let mut s = cache.state.lock().unwrap();
                        s.evict_to(new_budget);
                        s.bytes
                    };
                    if trace {
                        eprintln!(
                            "[guard] FIRE avail={:.2} floor={:.2} (base {:.2} + brake {:.2}) \
                             cache {:.2} -> {:.2} GB (budget {:.2}, freed {:.2})",
                            avail as f64 / 1e9,
                            floor_now as f64 / 1e9,
                            hard_floor as f64 / 1e9,
                            (floor_now - hard_floor) as f64 / 1e9,
                            resident as f64 / 1e9,
                            after as f64 / 1e9,
                            new_budget as f64 / 1e9,
                            resident.saturating_sub(after) as f64 / 1e9,
                        );
                    }
                    // MULTIPLICATIVE DECREASE. Without this the correction was thrown away
                    // on the very next tick: the soft path below used to reset the budget
                    // straight back to `fill_target`, so the cache refilled at full read
                    // bandwidth and was back at the wall ~2 s later. Measured on K3 as a
                    // 56 -> 37 -> 56 GB sawtooth against the OOM line, every cycle a coin
                    // flip against a scheduling gap. The guard is supposed to be a
                    // backstop; resetting its cut made it the operating point.
                    learned_cap = new_budget;
                    low_ticks = 0;
                    continue;
                }

                // Sustained external pressure (soft): cede gradually toward the danger line.
                low_ticks = if avail < danger_floor {
                    low_ticks.saturating_add(1)
                } else {
                    0
                };
                // ADDITIVE INCREASE. The guard's cut above is multiplicative and permanent
                // until earned back, and it is earned back only while there is real room —
                // slowly, so the *cap's* growth rate is what bounds our consumption rate.
                // That is the property that makes this loop stable: the cache can never
                // again approach the floor faster than `CAP_RECOVER_PER_TICK`, which is
                // ~0.64 GB/s against the 8.24 GB/s an unbounded refill managed. A 461 ms
                // scheduling gap then costs ~0.3 GB instead of 3.12 GB.
                // Never promise more than the memory that exists right now supports. Applied
                // every tick, so the approach is bounded by observation rather than by a
                // one-shot estimate made before the runtime revealed its own footprint.
                learned_cap = learned_cap
                    .min(supported_cap(resident, avail, hard_floor))
                    .max(FLOOR_MIN);
                // "Binding" needs a tolerance. The insert path stops *just under* the
                // budget, so the cache is never exactly at the cap and a `>=` test is
                // essentially never true: measured on K3 as cache=25.44 against cap=25.46,
                // pinned there with 38.92 GB free and a supported cap of ~51 GB. Within one
                // recovery step counts as binding.
                if avail > danger_floor
                    && resident.saturating_add(CAP_RECOVER_PER_TICK) >= learned_cap
                {
                    learned_cap = recover_cap(learned_cap, fill_target);
                }
                // Hold the learned cap minus whatever callers have reserved (e.g. live KV
                // caches), so the monitor never refills experts into space a request needs.
                let held = learned_cap.saturating_sub(cache.reserved.load(Ordering::Relaxed));
                let new_budget = if low_ticks >= SUSTAIN {
                    resident.saturating_sub(danger_floor - avail).max(FLOOR_MIN)
                } else {
                    held.max(FLOOR_MIN) // hold; our own transient spikes don't evict
                };
                cache.budget.store(new_budget, Ordering::Relaxed);
                if new_budget < resident {
                    cache.state.lock().unwrap().evict_to(new_budget);
                }
            }
        });
    }
}

impl<P: ExpertProvider> ExpertCache<P> {
    /// Reserve `bytes` of RAM for a non-expert allocation about to happen — a request's
    /// KV cache, sized to *that request's* prompt + completion (not the worst-case window).
    /// Evicts LRU experts **now** so the allocation has room, instead of pre-reserving the
    /// full context statically or racing the async monitor when the KV is allocated eagerly
    /// (a large-`COLI_CTX` request allocs its whole KV in one shot). Balance with
    /// [`release_ram`](ExpertCache::release_ram) once the request's KV is dropped.
    ///
    /// Returns `true` if the room now exists (or the cache is unmanaged and can't tell),
    /// `false` if even evicting every expert down to the floor cannot free enough — the
    /// caller must then **not** allocate (reject the request) rather than OOM the box. On
    /// `false` the reservation is rolled back.
    #[must_use]
    pub fn reserve_ram(&self, bytes: u64) -> bool {
        const FLOOR_MIN: u64 = 2 << 30;
        self.reserved.fetch_add(bytes, Ordering::Relaxed);
        if self.fill_target.load(Ordering::Relaxed) == 0 {
            return true; // unmanaged: no live signal; the static budget left headroom
        }
        let hard = self.hard_floor.load(Ordering::Relaxed);
        let avail = match available_ram_bytes() {
            Some(a) => a,
            None => return true, // no /proc/meminfo: can't evict-to-fit, assume OK
        };
        // Want `bytes` free for the KV *and* still clear the hard floor afterward.
        let need = bytes.saturating_add(hard);
        if avail >= need {
            return true; // already enough headroom
        }
        // Evict, re-measure, evict the REMAINDER. A single pass structurally undershoots:
        // it targets exactly the deficit, but freeing N bytes of cache does not raise
        // `MemAvailable` by N. Measured 2026-08-02 on m2.7 — evicting 64.4 GB moved avail
        // 40.1 -> 102.5 GB, returning 62.4 GB (~97%) and missing `need` by 2.0 GB. The
        // request was refused for a 2 GB shortfall after correctly freeing 64 GB.
        //
        // A fixed slack constant would be the wrong fix: the shortfall is a ratio, not a
        // size, and any constant is either too small for a big eviction or over-evicts a
        // small one. Re-measuring costs one /proc/meminfo read per pass and is self-scaling.
        // Bounded so a pathological case cannot spin; it stops early at the floor anyway.
        const MAX_PASSES: usize = 4;
        let before = self.state.lock().unwrap().bytes;
        let mut after = before;
        let mut avail_after = Some(avail);
        let mut passes = 0usize;
        for _ in 0..MAX_PASSES {
            let now = match available_ram_bytes() {
                Some(a) => a,
                None => break, // no signal; the post-loop check treats this as OK
            };
            avail_after = Some(now);
            if now >= need {
                break;
            }
            let mut s = self.state.lock().unwrap();
            let target = s.bytes.saturating_sub(need - now).max(FLOOR_MIN);
            if target >= s.bytes {
                after = s.bytes;
                break; // already at the floor — evicting more is not possible
            }
            s.evict_to(target);
            self.budget.store(target, Ordering::Relaxed);
            after = s.bytes;
            drop(s);
            passes += 1;
        }
        // Decide on a FRESH reading, not the one sampled at the top of the last pass — if the
        // loop exits by exhausting MAX_PASSES that value predates its own eviction and would
        // reject a request that now fits.
        avail_after = available_ram_bytes().or(avail_after);
        let ok = avail_after.map(|a| a >= need).unwrap_or(true);
        // Always log this path. Evicting the expert cache to admit one request is a rare,
        // operationally significant event, and its outcome was previously invisible: a
        // rejection here and a rejection from the ledger downstream produced indistinguishable
        // symptoms. `slack` is the number that matters — a small negative value means the
        // eviction undershot rather than the request being genuinely too big, which is exactly
        // the 2.0 GB miss that motivated the multi-pass loop above.
        let g = |b: u64| b as f64 / 1e9;
        eprintln!(
            "[cache] reserve_ram {}: need {:.1} GB (kv {:.1} + floor {:.1}) | avail {:.1} -> \
             {:.1} GB | experts {:.1} -> {:.1} GB (floor_min {:.1}) | {} pass(es) | slack {:.1} GB",
            if ok { "OK" } else { "FAILED" },
            g(need),
            g(bytes),
            g(hard),
            g(avail),
            avail_after.map(g).unwrap_or(f64::NAN),
            g(before),
            g(after),
            g(FLOOR_MIN),
            passes,
            avail_after.map(|a| a as f64 - need as f64).unwrap_or(f64::NAN) / 1e9,
        );
        if !ok {
            self.release_ram(bytes); // roll back — the caller will reject
        }
        ok
    }

    /// Release a prior [`reserve_ram`](ExpertCache::reserve_ram). Only drops the counter;
    /// the monitor grows experts back into the freed room on its next tick (avoiding a
    /// thundering refill race between concurrent requests).
    pub fn release_ram(&self, bytes: u64) {
        let prev = self.reserved.load(Ordering::Relaxed);
        self.reserved
            .store(prev.saturating_sub(bytes), Ordering::Relaxed);
    }
}

/// Capacity planning for DGX Spark deployments.
pub mod capacity {
    use colibri_core::Config;

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

    /// KV-cache bytes per token. Delegates to [`crate::KvCache::bytes_per_token`],
    /// which lives beside the allocation and is the single source of truth.
    ///
    /// This used to be its own MLA-only formula — `(kv_lora + qk_rope) * 4 * n_layers`
    /// — which silently under-reported every model that is not GLM: it omitted the GQA
    /// `k_full`/`v_full` (the dominant term for M3 / M2.7), the CUDA device shadow, and
    /// for a hybrid stack it charged all layers rather than just the attention ones.
    /// `coli capacity` therefore quoted context limits far larger than RAM can hold.
    /// Keep this a delegate: a fourth private copy of this arithmetic is how each of
    /// the previous errors happened.
    pub fn kv_bytes_per_token(cfg: &Config) -> u64 {
        crate::KvCache::bytes_per_token(cfg) as u64
    }

    /// Per-sequence KV bytes independent of context length (Mamba2 recurrent state);
    /// 0 for non-hybrid models. See [`crate::KvCache::fixed_bytes`].
    pub fn kv_fixed_bytes(cfg: &Config) -> u64 {
        crate::KvCache::fixed_bytes(cfg) as u64
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

/// Ticks of stopping distance the OOM guard keeps in hand. The guard can brake at most
/// once per `TICK_MS`, so it needs the floor to cover the fill it cannot yet see, plus
/// slack for a poll that lands late — the monitor thread competes with 20 reader threads
/// and the prefill on 20 cores, so ticks are not evenly spaced.
const BRAKE_TICKS: u64 = 4;

/// Cap on the brake term. A single pathological tick (an external tenant taking tens of GB
/// at once) must not translate into "evict the entire cache" — the reclaim arithmetic below
/// the floor already handles a genuine emergency, and this term is only the *anticipatory*
/// part.
const MAX_BRAKE: u64 = 8 << 30;

/// The floor the OOM guard must actually defend, given how fast memory is disappearing.
///
/// **A static floor is a bug when the fill is fast.** The guard is a control loop: it
/// samples `MemAvailable` every `TICK_MS` and can only evict at a tick. So it is safe only
/// while
///
/// ```text
///     consumption_per_tick  <  hard_floor − (whatever kills us)
/// ```
///
/// Measured on Kimi-K3 (2026-07-31, 512-token prefill, `/tmp/k3_trace2.rss`): the expert
/// cache fills at **8.24 GB/s** — 0.82 GB per 100 ms tick — because K3's set is 1347 GiB
/// against 3% coverage, so it streams at full read bandwidth and never reaches its fill
/// target. The margin from `ADAPTIVE_HARD_FLOOR` (3 GiB = 3.22 GB) down to earlyoom's
/// `-m 2` line (2% of MemTotal = 2.61 GB) is **0.61 GB — 0.74 of one tick**. The trace
/// caught exactly that: the last sample read `avail=3.19 GB`, already under the floor, and
/// the process was SIGTERMed before the next poll could evict anything.
///
/// So the stopping distance has to scale with the speed. This adds `BRAKE_TICKS` worth of
/// the observed per-tick drop to the floor, which is a no-op for a cache in steady state
/// (drop ≈ 0 ⇒ floor unchanged) and only bites while something is consuming fast — i.e. it
/// cannot regress the four models that were already passing.
///
/// This was *not* an accounting shortfall. Two earlier hypotheses were wrong: that the
/// buffer-pool byte cap killed K3 (it dies with the cap reverted), and that
/// `RUNTIME_RESERVE` was simply too small (the plan does fit — dense 62.06 GB observed plus
/// a 55.8 GB fill target leaves 8.6 GB, comfortably above a 3.22 GB floor). The plan was
/// fine; the guard could not hold it.
fn braking_floor(hard_floor: u64, drop_per_tick: u64) -> u64 {
    hard_floor.saturating_add(drop_per_tick.saturating_mul(BRAKE_TICKS).min(MAX_BRAKE))
}

/// Bytes the learned cap earns back per tick once memory pressure has passed. See
/// [`recover_cap`].
const CAP_RECOVER_PER_TICK: u64 = 64 << 20;

/// One additive-increase step on the learned cap.
///
/// The cap is the loop's memory: the guard cuts it multiplicatively the moment it has to
/// evict, and this earns it back at a fixed, deliberately slow rate. Two things fall out,
/// and both were failures observed on K3 before this existed:
///
/// 1. **The guard's correction survives.** The soft path used to reset the budget straight
///    back to `fill_target` on the next non-firing tick, so the cache refilled at full read
///    bandwidth and was back at the OOM line ~2 s later — a 56 -> 37 -> 56 GB sawtooth with
///    the guard as the operating point rather than a backstop.
/// 2. **Consumption becomes rate-limited by us, not by the drive.** The cache can only grow
///    into headroom the cap has already granted, so approach speed is `CAP_RECOVER_PER_TICK`
///    per tick (~0.64 GB/s) rather than the ~8.24 GB/s an unbounded refill managed. A late
///    poll then costs ~0.3 GB instead of the 3.12 GB one measured 461 ms gap swallowed.
///
/// The cap never exceeds the load-time plan, so a model that never trips the guard — every
/// model that already passed the ceiling check — behaves exactly as before.
///
/// **Only called while the cap is actually binding** (`resident >= learned_cap`). Raising a
/// cap the cache has not yet reached buys nothing and actively destroys information: the
/// first version recovered on every comfortable tick, so on K3 the seeded 53.66 GB cap
/// climbed back to the planned 56.10 GB in 38 ticks (3.8 s) while the initial fill was still
/// in progress (~7 s). The measurement was erased before it could take effect and the run
/// died at the same 122 GB as with no seed at all. Additive increase has to be conditioned
/// on the constraint being active, or it is just a slow way back to the number that failed.
/// Free bytes the cap must leave unspoken-for once the cache is full.
///
/// Because `supported_cap` is a fixed point (the cache growing by D raises `resident` and
/// lowers `avail` by the same D), **this margin IS the free memory at steady state** — not
/// a transient allowance. It has to cover the line we must never cross, plus room for
/// non-cache memory to grow after the cache has settled.
///
/// It was `danger_floor + MAX_BRAKE` = 12.88 GB, and neither term was chosen for this job.
/// `danger_floor` is the *soft* line for a sustained external tenant, deliberately above
/// the real one; `MAX_BRAKE` is a CAP on the *anticipatory* brake term in
/// [`braking_floor`], which already scales with the observed consumption rate — reserving
/// its maximum permanently pays for a worst case that the rate-aware floor handles when it
/// actually happens. I reused both because they were to hand.
///
/// So: the real floor (now earlyoom-derived, see `oom_guard_floor`) plus half a brake for
/// non-cache growth. On gx10-42b2 that is 3.69 + 4.29 = 7.98 GB rather than 12.88 GB,
/// returning ~4.9 GB to the expert cache.
///
/// This matters more than its size suggests: the #41 sweep measured the margin moving
/// prefill by up to 17% in EITHER direction, per model — M2.7 (near-fit, ~86% coverage)
/// gained 17.1% on prefill from more free memory while losing 11.9% on decode from less
/// cache, and M3/GLM went the other way. It is the largest single lever measured.
fn cap_margin(hard_floor: u64) -> u64 {
    hard_floor.saturating_add(MAX_BRAKE / 2)
}

/// The largest cap the memory that exists *right now* can support.
///
/// The seeded cap fixed t=0 only, and t=0 is not where the danger is. K3's first approach
/// ran at full read bandwidth into a cap that was correct when it was chosen and stale a
/// second later, because the cache is not the only thing consuming: read buffers, GPU
/// staging and the CUDA context draw on the same pool, so `MemAvailable` falls *faster*
/// than the cache grows. The guard still caught it — but only at avail 6.59 GB, and the
/// dip that followed reached **2.45 GB, under earlyoom's 2.61 GB trigger**. It lived.
///
/// So the seed becomes a per-tick invariant: whatever the cache already holds, plus what
/// is free, less the margin we refuse to spend. That is self-correcting in the one way the
/// seed was not — as runtime overhead reveals itself, the ceiling comes down with it,
/// without waiting for an eviction to teach it.
///
/// This only ever *lowers* the cap. Earning it back stays the job of the deliberately slow
/// [`recover_cap`], so a transient dip cannot be converted into an instant refill.
fn supported_cap(resident: u64, avail: u64, hard_floor: u64) -> u64 {
    resident
        .saturating_add(avail)
        .saturating_sub(cap_margin(hard_floor))
}

fn recover_cap(learned_cap: u64, fill_target: u64) -> u64 {
    learned_cap
        .saturating_add(CAP_RECOVER_PER_TICK)
        .min(fill_target)
}

/// Available RAM in bytes, best-effort. Reads `/proc/meminfo` `MemAvailable` on
/// Linux (the DGX Spark target); returns `None` elsewhere (e.g. macOS dev boxes),
/// where the caller should fall back to an explicit budget.
pub fn available_ram_bytes() -> Option<u64> {
    meminfo_field("MemAvailable:")
}

/// Bytes currently paged out (`SwapTotal - SwapFree`).
///
/// `MemAvailable` is not sufficient to keep the box off swap. The kernel begins paging
/// while `MemAvailable` still reads comfortably above any floor — measured on 42b2 with a
/// 110 GB expert fill: the serve process reached 108.7 GiB RSS and **3 GB of swap was in
/// use while `MemAvailable` sat at 30 GB**, five times the danger floor and ten times the
/// hard floor, so neither guard ever fired. Throughput collapsed to 0.06-0.24 tok/s.
///
/// Swap-in-use is the unambiguous signal that residency has gone too far: a cache that
/// has to be paged back in is worse than no cache at all, since a page fault costs a
/// 4 KiB disk read where a miss would have cost one coalesced expert read.
pub fn swap_used_bytes() -> Option<u64> {
    let total = meminfo_field("SwapTotal:")?;
    let free = meminfo_field("SwapFree:")?;
    Some(total.saturating_sub(free))
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

    /// A still cache must not pay for a brake it does not need: with nothing consuming,
    /// the guard's floor is exactly `hard_floor`. This is what keeps the rate-aware floor
    /// from regressing the four models that already passed the ceiling check.
    #[test]
    fn braking_floor_is_inert_in_steady_state() {
        let hard = 3u64 << 30;
        assert_eq!(braking_floor(hard, 0), hard);
    }

    /// The Kimi-K3 case, in the units it was measured in.
    ///
    /// Fill 8.24 GB/s at `TICK_MS` = 100 ⇒ 0.82 GB per tick. earlyoom runs `-m 2` on a
    /// 130.66 GB box ⇒ it SIGTERMs at 2.61 GB. The floor the guard defends has to leave
    /// more than one tick of fill above that line, or it brakes into a process that is
    /// already dead — which is exactly what the trace caught (`avail=3.19 GB` at the last
    /// sample, under the old static 3.22 GB floor, killed before the next poll).
    #[test]
    fn braking_floor_clears_earlyoom_at_the_measured_k3_fill_rate() {
        let hard = 3u64 << 30; // ADAPTIVE_HARD_FLOOR = 3.22 GB
        let earlyoom = 2_613_000_000u64; // 2% of 130.66 GB MemTotal
        let per_tick = 824_000_000u64; // 8.24 GB/s x 100 ms

        // The old static floor: less than one tick of margin. This is the bug.
        assert!(
            hard - earlyoom < per_tick,
            "a static floor only failed because one tick of fill ({per_tick}) exceeded its \
             whole margin ({}); if that stops being true, re-derive BRAKE_TICKS",
            hard - earlyoom
        );

        // The rate-aware floor: strictly more than one tick of stopping distance.
        let floor = braking_floor(hard, per_tick);
        assert!(
            floor - earlyoom > per_tick,
            "guard must be able to brake before earlyoom: floor {floor} leaves {} above the \
             kill line, which is less than the {per_tick} consumed per tick",
            floor - earlyoom
        );
    }

    /// The stability invariant of the whole loop, in the units it was measured in.
    ///
    /// K3's unbounded refill reached 8.24 GB/s. Once the guard has cut the cap, the cache
    /// can only grow into headroom the cap has granted, so approach speed is the cap's
    /// recovery rate. That has to stay far enough under the read bandwidth that a late poll
    /// is survivable — the worst gap observed was 461 ms.
    #[test]
    fn cap_recovery_is_slow_enough_to_survive_a_late_poll() {
        const TICK_MS: u64 = 100; // must track spawn_adaptive_budget's TICK_MS
        let recover_per_s = CAP_RECOVER_PER_TICK * (1000 / TICK_MS);
        let measured_unbounded_fill = 8_240_000_000u64; // K3, before the cap existed
        assert!(
            recover_per_s * 8 < measured_unbounded_fill,
            "cap recovery {recover_per_s} B/s must be well under the {measured_unbounded_fill} \
             B/s an unbounded refill reached, or the guard is back to racing the drive"
        );
        // The worst scheduling gap seen was 461 ms. What can vanish in it must stay small
        // next to the margin the braking floor buys.
        let worst_gap_ms = 461u64;
        let lost = recover_per_s * worst_gap_ms / 1000;
        assert!(
            lost < 1 << 30,
            "a {worst_gap_ms} ms blind window may cost at most ~1 GB, got {lost}"
        );
    }

    /// The per-tick ceiling must bind exactly when free memory is about to run out, and be
    /// inert while there is real room. Numbers are the two measured K3 states.
    #[test]
    fn supported_cap_binds_on_approach_and_not_in_steady_state() {
        let hard = 3u64 << 30; // the real floor, not the soft danger line
        // Converged steady state: cache 33.32 GB with 29.12 GB free. Nothing to clamp —
        // the cap it earned (33.33) is comfortably under what memory supports.
        let steady = supported_cap(33_320_000_000, 29_120_000_000, hard);
        assert!(
            steady > 33_330_000_000,
            "must not claw back a converged cache: {steady}"
        );
        // The approach, at the moment the guard first fired: cache 53.62 GB, 6.59 GB free.
        // Here the ceiling has to bite, because the next second took it to 2.45 GB — under
        // earlyoom's 2.61 GB line.
        let approach = supported_cap(53_620_000_000, 6_590_000_000, hard);
        assert!(
            approach < 53_620_000_000,
            "must cut the cap below the cache during the approach: {approach}"
        );
    }

    /// The cap earns headroom back but never past the load-time plan — so a model that
    /// never trips the guard is unaffected by any of this.
    #[test]
    fn cap_recovery_is_bounded_by_the_plan() {
        let target = 50u64 << 30;
        assert_eq!(recover_cap(target, target), target, "never exceeds the plan");
        assert_eq!(recover_cap(target - 1, target), target, "clamps, not overshoots");
        assert_eq!(recover_cap(0, target), CAP_RECOVER_PER_TICK);
        assert_eq!(recover_cap(u64::MAX, target), target, "must not overflow");
    }

    /// One pathological tick must not demand evicting the world; the brake term is only
    /// the anticipatory part and the reclaim below the floor handles a real emergency.
    #[test]
    fn braking_floor_is_capped() {
        let hard = 3u64 << 30;
        assert_eq!(braking_floor(hard, 40 << 30), hard + MAX_BRAKE);
        assert_eq!(braking_floor(u64::MAX, 1 << 30), u64::MAX, "must not overflow");
    }

    /// The pre-rewrite eviction loop, kept verbatim as the oracle.
    fn evict_by_repeated_min(
        entries: &mut HashMap<(usize, usize), Entry>,
        bytes: &mut u64,
        pinned: &HashSet<(usize, usize)>,
        clock: u32,
        budget: u64,
        protect: &HashSet<(usize, usize)>,
    ) -> Vec<(usize, usize)> {
        let mut order = Vec::new();
        while *bytes > budget {
            let victim = entries
                .iter()
                .filter(|(k, _)| !pinned.contains(*k) && !protect.contains(*k))
                .min_by_key(|(_, e)| evict_score(e.heat, e.last, clock))
                .map(|(k, _)| *k);
            match victim {
                Some(k) => {
                    if let Some(e) = entries.remove(&k) {
                        *bytes -= e.bytes;
                        order.push(k);
                    }
                }
                None => break,
            }
        }
        order
    }

    #[test]
    fn batch_eviction_picks_the_same_victims_as_repeated_min() {
        // `evict_to_protecting` was O(entries) per victim and cost 1839 ms on a profiled
        // GLM run. Replacing it with a single ranked pass is only legitimate if it evicts
        // *exactly* what the old loop did — scores do not change as entries are removed,
        // so it should. Assert that rather than trust the argument.
        //
        // Distinct scores throughout: ties were always broken by HashMap iteration order,
        // which is arbitrary, so equality there is neither expected nor meaningful.
        // `bytes` is tracked on the Entry, so an empty Expert is enough here.
        let mk = |bytes: u64| Entry {
            expert: Arc::new(Expert::default()),
            bytes,
            heat: 1,
            last: 0,
        };
        let build = || {
            let mut m: HashMap<(usize, usize), Entry> = HashMap::new();
            let mut total = 0u64;
            for i in 0..64usize {
                let mut e = mk(1000 + i as u64);
                // `evict_score` is recency-primary; distinct `last` gives a total order.
                e.last = i as u32;
                e.heat = 1 + (i % 3) as u32;
                total += e.bytes;
                m.insert((i % 4, i), e);
            }
            (m, total)
        };

        let pinned: HashSet<(usize, usize)> = [(0usize, 0usize), (1, 5)].into_iter().collect();
        let protect: HashSet<(usize, usize)> = [(2usize, 6usize), (3, 11)].into_iter().collect();
        let clock = 100u32;

        for budget_frac in [0u64, 1, 2, 3] {
            let (mut a_entries, mut a_bytes) = build();
            let budget = a_bytes * budget_frac / 4;
            let expected = evict_by_repeated_min(
                &mut a_entries,
                &mut a_bytes,
                &pinned,
                clock,
                budget,
                &protect,
            );

            let (b_entries, b_bytes) = build();
            let mut s = State {
                entries: b_entries,
                pinned: pinned.clone(),
                bytes: b_bytes,
                clock,
                hits: 0,
                misses: 0,
                evictions: 0,
                session_usage: HashMap::new(),
            };
            s.evict_to_protecting(budget, &protect);

            // Guard against a vacuous pass: if nothing is ever evicted, every assertion
            // below holds trivially and the test proves nothing about the rewrite.
            assert!(
                !expected.is_empty(),
                "budget={budget}: oracle evicted nothing — test is not exercising eviction"
            );
            assert_eq!(
                s.evictions as usize,
                expected.len(),
                "budget={budget}: eviction count differs"
            );
            assert_eq!(s.bytes, a_bytes, "budget={budget}: resident bytes differ");
            let mut survivors: Vec<_> = s.entries.keys().copied().collect();
            let mut oracle: Vec<_> = a_entries.keys().copied().collect();
            survivors.sort_unstable();
            oracle.sort_unstable();
            assert_eq!(
                survivors, oracle,
                "budget={budget}: different victims chosen"
            );
            // Whatever else happened, the invariants the caller relies on must hold.
            for k in &pinned {
                assert!(s.entries.contains_key(k), "evicted a pinned entry {k:?}");
            }
            for k in &protect {
                assert!(s.entries.contains_key(k), "evicted a protected entry {k:?}");
            }
        }
    }

    #[test]
    fn topn_zero_predictor_really_predicts_nothing() {
        // `prefetch_topn()` returns Some(0) to wire the background loader for the prefill
        // prefetch-ahead while leaving the learned predictor inert. A push-then-check
        // loop broke that: it emitted one expert per layer even at topn=0, which on K3
        // showed up as 80 speculative loads per run that nothing asked for.
        // The predictor must actually HAVE something to predict, or this proves nothing:
        // teach the 1->2 transition first, exactly as `predictor_learns_layer_transition`
        // does. With a cold predictor `ranked` is empty and any bound looks correct.
        let mut p = Predictor::new(0);
        for _ in 0..2 {
            p.observe_and_predict(1, &[10]);
            p.observe_and_predict(2, &[20]);
        }
        let pred = p.observe_and_predict(1, &[10]);
        assert!(pred.is_empty(), "topn=0 must predict nothing, got {pred:?}");

        // Same teaching with a real bound still yields the learned expert, so the fix
        // did not simply disable prediction.
        let mut q = Predictor::new(4);
        for _ in 0..2 {
            q.observe_and_predict(1, &[10]);
            q.observe_and_predict(2, &[20]);
        }
        let qp = q.observe_and_predict(1, &[10]);
        assert_eq!(
            qp.first(),
            Some(&20),
            "topn=4 should still predict, got {qp:?}"
        );
        assert!(qp.len() <= 4, "bound must be exact, got {qp:?}");
    }

    #[test]
    fn prefetch_ahead_gate_tracks_fraction_not_absolute_count() {
        // The two measured anchors. GLM @4096: 160 experts, a layer routes to ~all →
        // prefetch-ahead is a 1.58× win and must stay on.
        assert!(
            ahead_predicts_next_layer(160, 160),
            "GLM full-union prefill must prefetch"
        );
        assert!(
            ahead_predicts_next_layer(120, 160),
            "75% of experts still predicts well"
        );

        // Kimi-K3 @5 tokens: 896 experts, union ~78. This CLEARS the old absolute gate
        // (78 ≥ 64) but predicts almost nothing — measured 2293 wasted loads (~50 GiB)
        // for 2 extra hits, expert-load 12926→16277 ms. It must now be rejected.
        assert!(
            !ahead_predicts_next_layer(78, 896),
            "K3's 8.7% union must NOT prefetch ahead — this is the regression under test"
        );

        // Same model at long context routes to ~all experts; the gate reopens on its own.
        assert!(
            ahead_predicts_next_layer(896, 896),
            "K3 at full union should prefetch again"
        );

        // Decode is excluded on the absolute floor regardless of fraction: a tiny model
        // where top-k IS most of the experts must still not speculate per token.
        assert!(
            !ahead_predicts_next_layer(8, 8),
            "decode-sized unions never prefetch ahead"
        );

        // Unsupplied expert count keeps the historical absolute-gate behaviour.
        assert!(
            ahead_predicts_next_layer(78, 0),
            "n_experts=0 falls back to the old gate"
        );
        assert!(
            !ahead_predicts_next_layer(63, 0),
            "the absolute floor still applies"
        );
    }

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
        assert_eq!(
            cache.inner.loads.load(Ordering::Relaxed),
            before,
            "pinned reloaded"
        );
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
        assert_eq!(
            cache.inner.loads.load(Ordering::Relaxed),
            before,
            "pinned reloaded"
        );
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
        assert!(
            (4..=12).contains(&n),
            "auto pinned {n}, expected the ~4 hot head"
        );
        assert!(bytes > 0);
        assert!(
            cov > 0.8,
            "coverage {cov} should capture the hot head's traffic"
        );
        assert_eq!(
            cache.usage_snapshot().total(),
            0,
            "warm-up isn't session usage"
        );
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
        assert!(
            n <= 4,
            "pinned {n}, must leave headroom below the 5-expert budget"
        );
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
