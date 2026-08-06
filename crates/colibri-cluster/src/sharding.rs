//! Expert-parallel sharding across DGX Spark nodes.
//!
//! The 744B model activates ~40B params/token but only ~11 GB of routed experts
//! change per token. In a multi-node deployment we split the **experts** across
//! nodes: each node owns a contiguous block of the `n_experts` per layer, streams
//! and computes only its block, and the router dispatches each token's chosen
//! experts to their owning node. The dense part (attention, shared expert,
//! embeddings — ~10 GB int4) is replicated on every node so attention runs
//! locally and only expert I/O crosses the wire.
//!
//! With one node this collapses to "everything is local", which is the current
//! single-node target; the mapping is written so the engine's MoE block calls
//! `owner()`/`is_local()` unconditionally and the single-node case is just
//! `owner() == self`.
//!
//! # Hot-aware assignment
//!
//! Contiguous blocks balance expert *count* but not *traffic*: routing is heavily
//! skewed (a handful of experts per layer take most selections), so whichever block
//! happens to hold the popular experts does more work. [`ExpertSharding::balanced`]
//! instead assigns experts to nodes by a weighted longest-processing-time greedy so
//! each node's total selection weight is near-equal — spreading the hot experts.
//!
//! **Measured: no throughput gain on 2 nodes.** Warm 32-token decode over 6 repeats
//! came out at ~1.95 tok/s contiguous vs ~1.96 hot-aware — indistinguishable. Expert
//! popularity is uncorrelated with expert *id*, so a contiguous half already draws
//! ~half the traffic by the law of large numbers; LPT tightens the worst case, not
//! the mean. It may still earn its keep with few experts per node, a pathological
//! workload, or many nodes (where each block is small enough for the split to be
//! lumpy) — but it is not a default, and it costs a usage history that every node
//! must replicate byte-for-byte or the handshake (rightly) refuses to run.
//!
//! **Every node must build the identical map**, or the activation exchange in
//! `moe_sharded` misroutes (node A ships expert `e` to the node it thinks owns it,
//! which may differ from where B computes it). The balanced map is therefore a pure
//! deterministic function of `(num_nodes, n_experts, weights)`; the weights come
//! from a *shared* usage history that the deployment must replicate across nodes.
//! [`ExpertSharding::fingerprint`] hashes the resulting map so callers can log it
//! and an operator (or a future handshake) can confirm all nodes agree before trusting
//! results. When in doubt, use [`ExpertSharding::new`] (contiguous) — it needs no
//! shared state and is agreement-free.

use std::sync::Arc;

/// A node in the cluster, identified by ordinal `0..num_nodes`.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct NodeId(pub u32);

/// Expert→node assignment for expert-parallel MoE.
///
/// Default (`new`) is contiguous near-equal blocks: expert `e` → node
/// `e * num_nodes / n_experts`, no shared state needed. `balanced` instead holds an
/// explicit per-expert owner table produced by weighted load-balancing. The same
/// mapping is used for every layer.
#[derive(Debug, Clone)]
pub struct ExpertSharding {
    num_nodes: u32,
    n_experts: u32,
    /// `Some(table)` ⇒ hot-aware: `owner(e) == table[e]`. `None` ⇒ closed-form
    /// contiguous blocks. `Arc` keeps clones cheap (it's held in `ClusterCtx`).
    table: Option<Arc<Vec<u32>>>,
}

impl ExpertSharding {
    /// Build a **contiguous** sharding for `num_nodes` (≥1) over `n_experts` (≥1).
    pub fn new(num_nodes: u32, n_experts: u32) -> ExpertSharding {
        assert!(num_nodes >= 1, "num_nodes must be >= 1");
        assert!(n_experts >= 1, "n_experts must be >= 1");
        ExpertSharding { num_nodes, n_experts, table: None }
    }

    /// Single-node sharding: every expert is local.
    pub fn single(n_experts: u32) -> ExpertSharding {
        ExpertSharding::new(1, n_experts)
    }

    /// **Hot-aware** sharding: assign each expert to a node so that the per-node sum
    /// of `weights` is as balanced as possible (spreading the popular experts).
    /// `weights[e]` is expert `e`'s aggregate selection count from the shared usage
    /// history; missing/short entries count as 0.
    ///
    /// Uses the LPT (longest-processing-time-first) greedy: experts are placed
    /// heaviest-first onto the currently-lightest node — a 4/3-approximation to the
    /// optimal makespan, and fully deterministic (ties broken by expert/node id), so
    /// every node given the same weights produces the same table. Falls back to the
    /// contiguous map for a single node.
    pub fn balanced(num_nodes: u32, n_experts: u32, weights: &[u64]) -> ExpertSharding {
        assert!(num_nodes >= 1, "num_nodes must be >= 1");
        assert!(n_experts >= 1, "n_experts must be >= 1");
        if num_nodes == 1 {
            return ExpertSharding::single(n_experts);
        }
        let w = |e: u32| weights.get(e as usize).copied().unwrap_or(0);
        // Heaviest expert first; ties by id ascending for determinism.
        let mut order: Vec<u32> = (0..n_experts).collect();
        order.sort_by(|&a, &b| w(b).cmp(&w(a)).then(a.cmp(&b)));

        let mut load = vec![0u64; num_nodes as usize];
        let mut table = vec![0u32; n_experts as usize];
        for e in order {
            // Lightest node; ties → lowest node id (explicit, deterministic).
            let node = (0..num_nodes as usize)
                .min_by(|&i, &j| load[i].cmp(&load[j]).then(i.cmp(&j)))
                .unwrap();
            table[e as usize] = node as u32;
            load[node] += w(e);
        }
        ExpertSharding { num_nodes, n_experts, table: Some(Arc::new(table)) }
    }

    /// **Capacity-weighted** sharding: node `i` owns a share of the experts proportional
    /// to `capacity[i]`, and within that quota the hot experts are still spread by load.
    ///
    /// [`balanced`](Self::balanced) equalises *load* and so hands every node roughly the
    /// same NUMBER of experts. That is right when the nodes are alike and wrong when they
    /// are not: a cluster whose members have very different free space cannot store an even
    /// split at all, and the small node ends up streaming its half from a disk that cannot
    /// hold it. Kimi-K3 is the live case — 1446 GB of experts against two boxes with a few
    /// hundred GB free each — where the point of sharding is to AGGREGATE NVMe bandwidth,
    /// which requires each node to actually hold what it owns.
    ///
    /// Quotas use the largest-remainder method, so they sum to exactly `n_experts` and no
    /// expert is dropped. Assignment is then LPT (as in `balanced`) restricted to nodes
    /// still under quota: heaviest expert onto the lightest node that can still take one.
    /// Deterministic — ties break by node id — so every node computes the same table.
    ///
    /// A node with zero capacity owns zero experts and simply serves none. If the total
    /// capacity is zero (or a single node), this falls back to `balanced`, since there is
    /// no capacity signal to weight by.
    pub fn capacity_weighted(
        num_nodes: u32,
        n_experts: u32,
        capacity: &[u64],
        weights: &[u64],
    ) -> ExpertSharding {
        assert!(num_nodes >= 1, "num_nodes must be >= 1");
        assert!(n_experts >= 1, "n_experts must be >= 1");
        let cap = |n: usize| capacity.get(n).copied().unwrap_or(0);
        let total: u128 = (0..num_nodes as usize).map(|n| cap(n) as u128).sum();
        if num_nodes == 1 || total == 0 {
            return ExpertSharding::balanced(num_nodes, n_experts, weights);
        }
        // Largest-remainder quotas: floor first, then hand the leftovers to the biggest
        // fractional parts. Guarantees sum == n_experts without a rounding drift that
        // would silently orphan an expert.
        let mut quota = vec![0u32; num_nodes as usize];
        let mut rem: Vec<(u128, usize)> = Vec::with_capacity(num_nodes as usize);
        let mut assigned = 0u32;
        for n in 0..num_nodes as usize {
            let exact = cap(n) as u128 * n_experts as u128;
            let q = (exact / total) as u32;
            quota[n] = q;
            assigned += q;
            rem.push((exact % total, n));
        }
        // Biggest remainder first; ties by node id for determinism.
        rem.sort_by(|a, b| b.0.cmp(&a.0).then(a.1.cmp(&b.1)));
        for &(_, n) in rem.iter().take((n_experts - assigned) as usize) {
            quota[n] += 1;
        }

        let w = |e: u32| weights.get(e as usize).copied().unwrap_or(0);
        let mut order: Vec<u32> = (0..n_experts).collect();
        order.sort_by(|&a, &b| w(b).cmp(&w(a)).then(a.cmp(&b)));

        let mut load = vec![0u64; num_nodes as usize];
        let mut used = vec![0u32; num_nodes as usize];
        let mut table = vec![0u32; n_experts as usize];
        for e in order {
            // Lightest node that still has quota left. One always exists: the quotas sum
            // to `n_experts` and we place exactly that many.
            let node = (0..num_nodes as usize)
                .filter(|&i| used[i] < quota[i])
                .min_by(|&i, &j| load[i].cmp(&load[j]).then(i.cmp(&j)))
                .expect("quotas sum to n_experts, so one node is always under quota");
            table[e as usize] = node as u32;
            used[node] += 1;
            load[node] += w(e);
        }
        ExpertSharding { num_nodes, n_experts, table: Some(Arc::new(table)) }
    }

    pub fn num_nodes(&self) -> u32 {
        self.num_nodes
    }

    pub fn n_experts(&self) -> u32 {
        self.n_experts
    }

    /// Whether this is a hot-aware (balanced) map vs a plain contiguous one.
    pub fn is_hot_aware(&self) -> bool {
        self.table.is_some()
    }

    /// The node that owns `expert`.
    pub fn owner(&self, expert: u32) -> NodeId {
        debug_assert!(expert < self.n_experts);
        match &self.table {
            Some(t) => NodeId(t[expert as usize]),
            // Contiguous blocks: e * N / E. Balanced to within one expert per node.
            None => NodeId((expert as u64 * self.num_nodes as u64 / self.n_experts as u64) as u32),
        }
    }

    /// Whether `expert` is owned by `node`.
    pub fn is_local(&self, node: NodeId, expert: u32) -> bool {
        self.owner(expert) == node
    }

    /// Half-open contiguous expert range `[start, end)` owned by `node`.
    ///
    /// **Contiguous maps only** — meaningless for a `balanced` map, whose experts
    /// are not contiguous. Use [`local_experts`](Self::local_experts) /
    /// [`count_for`](Self::count_for) for the general case.
    pub fn range_for(&self, node: NodeId) -> (u32, u32) {
        debug_assert!(self.table.is_none(), "range_for is contiguous-only");
        // Invert the block mapping: start = ceil(node * E / N), end = ceil((node+1) * E / N).
        let e = self.n_experts as u64;
        let n = self.num_nodes as u64;
        let node = node.0 as u64;
        let start = (node * e).div_ceil(n) as u32;
        let end = ((node + 1) * e).div_ceil(n) as u32;
        (start, end)
    }

    /// Number of experts owned by `node` (works for both map kinds).
    pub fn count_for(&self, node: NodeId) -> u32 {
        (0..self.n_experts).filter(|&e| self.owner(e) == node).count() as u32
    }

    /// Experts owned by `node`, ascending (works for both map kinds).
    pub fn local_experts(&self, node: NodeId) -> impl Iterator<Item = u32> + '_ {
        (0..self.n_experts).filter(move |&e| self.owner(e) == node)
    }

    /// Per-node total of `weights` under this map — the balance the assignment
    /// achieves. `node_weights()[n]` is the summed selection weight node `n` serves.
    pub fn node_weights(&self, weights: &[u64]) -> Vec<u64> {
        let mut out = vec![0u64; self.num_nodes as usize];
        for e in 0..self.n_experts {
            out[self.owner(e).0 as usize] += weights.get(e as usize).copied().unwrap_or(0);
        }
        out
    }

    /// FNV-1a hash of the full `owner(0..n_experts)` sequence (plus node/expert
    /// counts). Two nodes with matching fingerprints hold the identical map — log it
    /// on startup so cross-node disagreement (which would silently corrupt the
    /// activation exchange) is visible.
    pub fn fingerprint(&self) -> u64 {
        let mut h: u64 = 0xcbf29ce484222325;
        let mut mix = |x: u32| {
            for b in x.to_le_bytes() {
                h ^= b as u64;
                h = h.wrapping_mul(0x0000_0100_0000_01b3);
            }
        };
        mix(self.num_nodes);
        mix(self.n_experts);
        for e in 0..self.n_experts {
            mix(self.owner(e).0);
        }
        h
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn single_node_everything_local() {
        let sh = ExpertSharding::single(256);
        for e in 0..256 {
            assert_eq!(sh.owner(e), NodeId(0));
            assert!(sh.is_local(NodeId(0), e));
        }
        assert_eq!(sh.count_for(NodeId(0)), 256);
        assert_eq!(sh.range_for(NodeId(0)), (0, 256));
    }

    #[test]
    fn two_node_even_split() {
        // GLM-5.2: 256 experts across 2 DGX Sparks -> 128 each, contiguous.
        let sh = ExpertSharding::new(2, 256);
        assert_eq!(sh.range_for(NodeId(0)), (0, 128));
        assert_eq!(sh.range_for(NodeId(1)), (128, 256));
        assert_eq!(sh.owner(0), NodeId(0));
        assert_eq!(sh.owner(127), NodeId(0));
        assert_eq!(sh.owner(128), NodeId(1));
        assert_eq!(sh.owner(255), NodeId(1));
        assert_eq!(sh.count_for(NodeId(0)), 128);
        assert_eq!(sh.count_for(NodeId(1)), 128);
    }

    #[test]
    fn ranges_partition_all_experts() {
        // For any node count, the per-node ranges must tile [0, n_experts) with
        // no gaps or overlaps, and every expert's owner must fall in its range.
        for nodes in 1..=8u32 {
            for n_experts in [1u32, 7, 8, 100, 256, 257] {
                let sh = ExpertSharding::new(nodes, n_experts);
                let mut covered = 0u32;
                let mut prev_end = 0u32;
                for node in 0..nodes {
                    let (s, e) = sh.range_for(NodeId(node));
                    assert_eq!(s, prev_end, "gap/overlap at node {node}");
                    assert!(e >= s);
                    for expert in s..e {
                        assert_eq!(sh.owner(expert), NodeId(node));
                    }
                    covered += e - s;
                    prev_end = e;
                }
                assert_eq!(prev_end, n_experts);
                assert_eq!(covered, n_experts);
            }
        }
    }

    #[test]
    fn balanced_within_one() {
        // No node should own more than one extra expert vs any other.
        let sh = ExpertSharding::new(3, 256);
        let counts: Vec<u32> = (0..3).map(|n| sh.count_for(NodeId(n))).collect();
        let min = *counts.iter().min().unwrap();
        let max = *counts.iter().max().unwrap();
        assert!(max - min <= 1, "unbalanced: {counts:?}");
        assert_eq!(counts.iter().sum::<u32>(), 256);
    }

    #[test]
    fn hot_aware_beats_contiguous_on_skew() {
        // Skewed weights concentrated in a contiguous ID block: experts 0..8 are
        // hot (weight 1000), the rest cold (weight 1). Contiguous puts all 8 hot
        // experts on node 0 -> badly imbalanced traffic. Hot-aware spreads them.
        let e = 64u32;
        let mut w = vec![1u64; e as usize];
        for h in 0..8 {
            w[h] = 1000;
        }
        let contig = ExpertSharding::new(2, e);
        let cw = contig.node_weights(&w);
        // node 0 owns experts 0..32 (all 8 hot) -> ~8000 vs ~56.
        assert!(cw[0] as f64 / cw[1] as f64 > 10.0, "contiguous should be skewed: {cw:?}");

        let hot = ExpertSharding::balanced(2, e, &w);
        assert!(hot.is_hot_aware());
        let hw = hot.node_weights(&w);
        let (min, max) = (*hw.iter().min().unwrap(), *hw.iter().max().unwrap());
        // Balanced to within one expert's weight (1000): 4 hot each side.
        assert!(max - min <= 1000, "hot-aware should balance traffic: {hw:?}");
        // And it still partitions every expert exactly once.
        let owned: u32 = (0..2).map(|n| hot.count_for(NodeId(n))).sum();
        assert_eq!(owned, e);
    }

    #[test]
    fn balanced_is_deterministic_across_nodes() {
        // Two independently-built maps from the same weights must be byte-identical
        // (same fingerprint) — this is what keeps the activation exchange correct.
        let w: Vec<u64> = (0..256).map(|e| ((e * 37 + 11) % 100) as u64).collect();
        let a = ExpertSharding::balanced(4, 256, &w);
        let b = ExpertSharding::balanced(4, 256, &w);
        assert_eq!(a.fingerprint(), b.fingerprint());
        for e in 0..256 {
            assert_eq!(a.owner(e), b.owner(e));
        }
        // A different weight vector yields a different map (fingerprint changes).
        let mut w2 = w.clone();
        w2[0] += 5000;
        let c = ExpertSharding::balanced(4, 256, &w2);
        assert_ne!(a.fingerprint(), c.fingerprint());
    }

    #[test]
    fn balanced_single_node_is_all_local() {
        let sh = ExpertSharding::balanced(1, 256, &[5; 256]);
        assert!(!sh.is_hot_aware(), "1 node collapses to contiguous");
        for e in 0..256 {
            assert_eq!(sh.owner(e), NodeId(0));
        }
    }

    #[test]
    fn local_experts_and_count_agree_for_hot_aware() {
        let w: Vec<u64> = (0..100).map(|e| (e % 7) as u64).collect();
        let sh = ExpertSharding::balanced(3, 100, &w);
        let mut seen = 0u32;
        for n in 0..3 {
            let listed: Vec<u32> = sh.local_experts(NodeId(n)).collect();
            assert_eq!(listed.len() as u32, sh.count_for(NodeId(n)));
            for &e in &listed {
                assert_eq!(sh.owner(e), NodeId(n));
            }
            seen += listed.len() as u32;
        }
        assert_eq!(seen, 100, "every expert owned exactly once");
    }

    #[test]
    fn contiguous_fingerprint_is_stable() {
        // Two contiguous maps with the same params agree; different node counts differ.
        assert_eq!(
            ExpertSharding::new(2, 256).fingerprint(),
            ExpertSharding::new(2, 256).fingerprint()
        );
        assert_ne!(
            ExpertSharding::new(2, 256).fingerprint(),
            ExpertSharding::new(3, 256).fingerprint()
        );
    }

    /// Expert counts track free space, not node count. The K3 case: two boxes with very
    /// different room, where an even split cannot be stored at all.
    #[test]
    fn capacity_weighted_counts_are_proportional_to_capacity() {
        const E: u32 = 896; // K3's experts per layer
        // 296 GB vs 243 GB free, the actual figures on 42b2 and 5a4f.
        let cap = [296u64 << 30, 243u64 << 30];
        let sh = ExpertSharding::capacity_weighted(2, E, &cap, &[]);
        let (a, b) = (sh.count_for(NodeId(0)), sh.count_for(NodeId(1)));
        assert_eq!(a + b, E, "every expert must be owned exactly once");
        // Within one expert of the exact proportion — largest-remainder's guarantee.
        let want_a = (E as u64 * cap[0] / (cap[0] + cap[1])) as u32;
        assert!(
            a.abs_diff(want_a) <= 1,
            "node 0 got {a}, proportional share is {want_a}"
        );
        assert!(a > b, "the roomier node must own more");
    }

    /// Quotas sum exactly, for awkward ratios that expose a floor-and-hope rounding.
    #[test]
    fn capacity_weighted_never_drops_or_duplicates_an_expert() {
        for (nodes, e, cap) in [
            (3u32, 10u32, vec![1u64, 1, 1]),   // 10/3 — no exact split
            (3, 7, vec![5u64, 3, 1]),
            (4, 5, vec![7u64, 0, 2, 1]),       // a zero-capacity node
            (2, 897, vec![u64::MAX / 2, 1]),   // extreme skew
        ] {
            let sh = ExpertSharding::capacity_weighted(nodes, e, &cap, &[]);
            let total: u32 = (0..nodes).map(|n| sh.count_for(NodeId(n))).sum();
            assert_eq!(total, e, "nodes={nodes} e={e}: counts must sum to n_experts");
            for x in 0..e {
                assert!(sh.owner(x).0 < nodes, "owner out of range");
            }
        }
    }

    /// The leftover expert goes to the node the floor short-changed MOST (largest
    /// remainder), not to whichever node happens to be checked first.
    ///
    /// Pinned because it is the one part of the quota rule that "sums correctly and is
    /// within one of proportional" does NOT determine: capacities 2:1 over 4 experts give
    /// exact shares 2.67 and 1.33, so the spare belongs to node 0. Handing it to node 1
    /// instead yields 2:2 — still exact, still within one, and wrong. A mutation run found
    /// that gap; this closes it.
    #[test]
    fn the_leftover_expert_goes_to_the_largest_remainder() {
        let sh = ExpertSharding::capacity_weighted(2, 4, &[2, 1], &[]);
        assert_eq!(
            (sh.count_for(NodeId(0)), sh.count_for(NodeId(1))),
            (3, 1),
            "2:1 capacity over 4 experts must be 3:1, not 2:2"
        );
    }

    /// A zero-capacity node stores nothing — it cannot serve what it cannot hold.
    #[test]
    fn a_node_with_no_space_owns_no_experts() {
        let sh = ExpertSharding::capacity_weighted(3, 64, &[10, 0, 6], &[]);
        assert_eq!(sh.count_for(NodeId(1)), 0);
        assert_eq!(sh.count_for(NodeId(0)) + sh.count_for(NodeId(2)), 64);
    }

    /// Within its quota it still spreads the hot experts — otherwise the roomier node
    /// would own the whole hot set and the split would balance bytes while unbalancing
    /// work. Compared against the worst case (hottest experts contiguous on one node).
    #[test]
    fn capacity_weighted_still_spreads_load_within_quota() {
        const E: u32 = 64;
        // A steep popularity curve: expert 0 is by far the hottest.
        let w: Vec<u64> = (0..E).map(|e| 1u64 << (31 - (e % 32))).collect();
        let sh = ExpertSharding::capacity_weighted(2, E, &[3, 1], &w);
        let nw = sh.node_weights(&w);
        assert_eq!(sh.count_for(NodeId(0)), 48, "3:1 capacity → 3:1 counts");
        assert_eq!(sh.count_for(NodeId(1)), 16);
        // The light node is only a quarter of the storage but carries real load — far
        // more than the ~0 it would get if experts were handed out coldest-first.
        assert!(
            nw[1] > 0 && nw[1] * 20 > nw[0],
            "load {nw:?} is too lopsided for a 3:1 storage split"
        );
    }

    /// No capacity signal ⇒ this is exactly `balanced`, not a silently different map.
    #[test]
    fn capacity_weighted_falls_back_to_balanced_without_capacities() {
        let w: Vec<u64> = (0..32u32).map(|e| (e as u64 * 7) % 13).collect();
        for cap in [vec![0u64, 0], vec![]] {
            let a = ExpertSharding::capacity_weighted(2, 32, &cap, &w);
            let b = ExpertSharding::balanced(2, 32, &w);
            for e in 0..32 {
                assert_eq!(a.owner(e), b.owner(e), "cap={cap:?} expert {e}");
            }
        }
    }
}
