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

    /// `(layer, eid)` ranked by count descending; ties broken by `(layer, eid)`
    /// ascending for determinism. This is the pin order (global, all layers
    /// pooled — matching the C `pin_load` ranking).
    pub fn ranked(&self) -> Vec<(usize, usize)> {
        let mut v: Vec<((usize, usize), u64)> =
            self.counts.iter().map(|(&k, &c)| (k, c)).collect();
        v.sort_by(|a, b| b.1.cmp(&a.1).then(a.0.cmp(&b.0)));
        v.into_iter().map(|(k, _)| k).collect()
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
