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
}
