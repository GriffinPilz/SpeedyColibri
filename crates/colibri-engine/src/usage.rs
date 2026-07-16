//! Persistent expert-usage history — port of the `.coli_usage` learning cache
//! (`usage_load`/`usage_save`/`stats_dump_q` in `c/glm.c`).
//!
//! A cross-session histogram of how often each `(layer, expert)` was selected by
//! the router. It drives the pinned hot-store warm-up (AUTOPIN): at startup the
//! hottest experts are pinned resident so they never hit the disk path.
//!
//! The on-disk format is `"<layer> <expert> <count>\n"` per line — **identical to
//! the C engine's `.coli_usage`**, so the two share the file.

use std::collections::HashMap;
use std::fs::File;
use std::io::{self, BufRead, BufReader, Write};
use std::path::Path;

/// A `(layer, expert) -> selection count` histogram.
#[derive(Debug, Clone, Default)]
pub struct UsageHistory {
    counts: HashMap<(usize, usize), u64>,
}

impl UsageHistory {
    pub fn new() -> UsageHistory {
        UsageHistory::default()
    }

    /// Record one selection of `(layer, eid)`.
    pub fn record(&mut self, layer: usize, eid: usize) {
        *self.counts.entry((layer, eid)).or_insert(0) += 1;
    }

    /// Add `count` selections of `(layer, eid)`.
    pub fn add(&mut self, layer: usize, eid: usize, count: u64) {
        *self.counts.entry((layer, eid)).or_insert(0) += count;
    }

    pub fn get(&self, layer: usize, eid: usize) -> u64 {
        self.counts.get(&(layer, eid)).copied().unwrap_or(0)
    }

    pub fn len(&self) -> usize {
        self.counts.len()
    }

    pub fn is_empty(&self) -> bool {
        self.counts.is_empty()
    }

    /// Total selections across all experts.
    pub fn total(&self) -> u64 {
        self.counts.values().sum()
    }

    /// Per-expert selection weight summed across all layers: `w[e] = Σ_layer
    /// count(layer, e)`, length `n_experts`. Feeds [`ExpertSharding::balanced`] —
    /// expert ownership is layer-independent, so we balance the aggregate traffic.
    pub fn expert_weights(&self, n_experts: usize) -> Vec<u64> {
        let mut w = vec![0u64; n_experts];
        for (&(_layer, e), &c) in &self.counts {
            if e < n_experts {
                w[e] += c;
            }
        }
        w
    }

    /// `(layer, eid)` ranked by count descending; ties broken by `(layer, eid)`
    /// ascending for determinism. This is the pin order (global, all layers
    /// pooled — matching the C `pin_load` ranking).
    pub fn ranked(&self) -> Vec<(usize, usize)> {
        let mut v: Vec<((usize, usize), u64)> =
            self.counts.iter().map(|(&k, &c)| (k, c)).collect();
        v.sort_by(|a, b| b.1.cmp(&a.1).then(a.0.cmp(&b.0)));
        v.into_iter().map(|(k, _)| k).collect()
    }

    /// Descending selection counts (the `ranked()` order), for coverage/knee math.
    fn sorted_counts(&self) -> Vec<u64> {
        let mut v: Vec<u64> = self.counts.values().copied().collect();
        v.sort_unstable_by(|a, b| b.cmp(a));
        v
    }

    /// Fraction of all selections covered by the `n` hottest experts (the pin
    /// coverage if the top-`n` were pinned). `n == 0` → 0; `n >= len` → 1.
    pub fn coverage_of_top(&self, n: usize) -> f64 {
        let total = self.total();
        if total == 0 || n == 0 {
            return 0.0;
        }
        let covered: u64 = self.sorted_counts().iter().take(n).sum();
        covered as f64 / total as f64
    }

    /// The **knee** of the cumulative-coverage curve: how many of the hottest
    /// experts to pin before returns flatten. Auto-sizing for AUTOPIN — instead of
    /// a hand-picked `COLI_PIN_GB`, pin exactly the hot head and stream the tail.
    ///
    /// The curve `cum[i] = Σ_{j≤i} count[j] / total` (experts hottest-first) is
    /// concave-increasing, so it has a well-defined elbow. We take the Kneedle
    /// point: the index of maximum vertical distance above the chord joining the
    /// first and last points. Returns a **count of experts** (≥1 when non-empty).
    pub fn knee(&self) -> usize {
        let counts = self.sorted_counts();
        let n = counts.len();
        if n <= 2 {
            return n;
        }
        let total: u64 = counts.iter().sum();
        if total == 0 {
            return n;
        }
        // Cumulative coverage in [0,1], then farthest point above the endpoints' chord.
        let inv_total = 1.0 / total as f64;
        let inv_span = 1.0 / (n - 1) as f64;
        let mut cum = 0u64;
        let y0 = counts[0] as f64 * inv_total; // cum[0]
        let mut best_i = 0usize;
        let mut best_d = f64::NEG_INFINITY;
        for (i, &c) in counts.iter().enumerate() {
            cum += c;
            let y = cum as f64 * inv_total;
            let x = i as f64 * inv_span; // 0..1
            let chord = y0 + x * (1.0 - y0); // line from (0,y0) to (1,1)
            let d = y - chord;
            if d > best_d {
                best_d = d;
                best_i = i;
            }
        }
        best_i + 1 // count = index + 1
    }

    /// Merge another history into this one (summing counts).
    pub fn merge(&mut self, other: &UsageHistory) {
        for (&(l, e), &c) in &other.counts {
            self.add(l, e, c);
        }
    }

    /// Load from a `.coli_usage`-format file. Missing file → empty history.
    pub fn load(path: impl AsRef<Path>) -> io::Result<UsageHistory> {
        let mut h = UsageHistory::new();
        let f = match File::open(path) {
            Ok(f) => f,
            Err(e) if e.kind() == io::ErrorKind::NotFound => return Ok(h),
            Err(e) => return Err(e),
        };
        for line in BufReader::new(f).lines() {
            let line = line?;
            let mut it = line.split_whitespace();
            if let (Some(l), Some(e), Some(c)) = (it.next(), it.next(), it.next()) {
                if let (Ok(l), Ok(e), Ok(c)) =
                    (l.parse::<usize>(), e.parse::<usize>(), c.parse::<u64>())
                {
                    h.add(l, e, c);
                }
            }
        }
        Ok(h)
    }

    /// Write to a `.coli_usage`-format file (nonzero counts, sorted for a stable,
    /// diffable file).
    pub fn save(&self, path: impl AsRef<Path>) -> io::Result<()> {
        let mut keys: Vec<(usize, usize)> = self.counts.keys().copied().collect();
        keys.sort();
        let mut f = File::create(path)?;
        for (l, e) in keys {
            let c = self.counts[&(l, e)];
            if c > 0 {
                writeln!(f, "{l} {e} {c}")?;
            }
        }
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn ranks_hottest_first() {
        let mut h = UsageHistory::new();
        h.add(3, 10, 5);
        h.add(3, 20, 100);
        h.add(4, 7, 50);
        assert_eq!(h.ranked(), vec![(3, 20), (4, 7), (3, 10)]);
        assert_eq!(h.total(), 155);
    }

    #[test]
    fn save_load_roundtrip_c_format() {
        let dir = std::env::var("TMPDIR").unwrap_or_else(|_| "/tmp".into());
        let path = std::path::Path::new(&dir).join(format!("coli-usage-{}.txt", std::process::id()));
        let mut h = UsageHistory::new();
        h.add(3, 1, 7);
        h.add(5, 200, 3);
        h.save(&path).unwrap();
        // file is exactly "<layer> <eid> <count>" lines
        let text = std::fs::read_to_string(&path).unwrap();
        assert!(text.lines().any(|l| l == "3 1 7"));
        assert!(text.lines().any(|l| l == "5 200 3"));
        let h2 = UsageHistory::load(&path).unwrap();
        assert_eq!(h2.get(3, 1), 7);
        assert_eq!(h2.get(5, 200), 3);
        std::fs::remove_file(&path).ok();
    }

    #[test]
    fn missing_file_is_empty() {
        let h = UsageHistory::load("/no/such/coli_usage").unwrap();
        assert!(h.is_empty());
    }

    #[test]
    fn coverage_of_top_matches_hand_count() {
        let mut h = UsageHistory::new();
        h.add(0, 0, 70);
        h.add(0, 1, 20);
        h.add(0, 2, 10); // total 100
        assert_eq!(h.total(), 100);
        assert!((h.coverage_of_top(0) - 0.0).abs() < 1e-9);
        assert!((h.coverage_of_top(1) - 0.70).abs() < 1e-9);
        assert!((h.coverage_of_top(2) - 0.90).abs() < 1e-9);
        assert!((h.coverage_of_top(3) - 1.00).abs() < 1e-9);
        assert!((h.coverage_of_top(99) - 1.00).abs() < 1e-9);
    }

    #[test]
    fn knee_finds_elbow_of_skewed_curve() {
        // 5 hot experts (count 100) then a long flat tail (count 1). The elbow of
        // the cumulative-coverage curve should land right at the head, not the tail.
        let mut h = UsageHistory::new();
        for e in 0..5 {
            h.add(0, e, 100);
        }
        for e in 5..200 {
            h.add(0, e, 1);
        }
        let k = h.knee();
        assert!((5..=15).contains(&k), "knee {k} should sit near the 5-expert head");
        // Pinning the knee captures the bulk of the traffic.
        assert!(h.coverage_of_top(k) > 0.7, "knee coverage {}", h.coverage_of_top(k));
    }

    #[test]
    fn knee_of_uniform_is_not_degenerate() {
        // Uniform usage has no elbow; knee must stay in-range and never panic.
        let mut h = UsageHistory::new();
        for e in 0..50 {
            h.add(0, e, 10);
        }
        let k = h.knee();
        assert!((1..=50).contains(&k));
    }

    #[test]
    fn knee_tiny_histories() {
        assert_eq!(UsageHistory::new().knee(), 0);
        let mut one = UsageHistory::new();
        one.add(0, 0, 5);
        assert_eq!(one.knee(), 1);
    }

    #[test]
    fn merge_sums() {
        let mut a = UsageHistory::new();
        a.add(1, 1, 2);
        let mut b = UsageHistory::new();
        b.add(1, 1, 3);
        b.add(2, 2, 1);
        a.merge(&b);
        assert_eq!(a.get(1, 1), 5);
        assert_eq!(a.get(2, 2), 1);
    }
}
