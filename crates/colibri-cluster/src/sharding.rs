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

/// A node in the cluster, identified by ordinal `0..num_nodes`.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct NodeId(pub u32);

/// Expert→node assignment for expert-parallel MoE.
///
/// Experts are split into contiguous, near-equal blocks: expert `e` is owned by
/// node `e * num_nodes / n_experts`. The same mapping is used for every layer
/// (simplest, and keeps each node's on-disk expert shard contiguous). A per-layer
/// interleave could be added later for finer load balancing.
#[derive(Debug, Clone, Copy)]
pub struct ExpertSharding {
    num_nodes: u32,
    n_experts: u32,
}

impl ExpertSharding {
    /// Build a sharding for `num_nodes` (≥1) over `n_experts` (≥1) experts.
    pub fn new(num_nodes: u32, n_experts: u32) -> ExpertSharding {
        assert!(num_nodes >= 1, "num_nodes must be >= 1");
        assert!(n_experts >= 1, "n_experts must be >= 1");
        ExpertSharding {
            num_nodes,
            n_experts,
        }
    }

    /// Single-node sharding: every expert is local.
    pub fn single(n_experts: u32) -> ExpertSharding {
        ExpertSharding::new(1, n_experts)
    }

    pub fn num_nodes(&self) -> u32 {
        self.num_nodes
    }

    pub fn n_experts(&self) -> u32 {
        self.n_experts
    }

    /// The node that owns `expert`.
    pub fn owner(&self, expert: u32) -> NodeId {
        debug_assert!(expert < self.n_experts);
        // Contiguous blocks: e * N / E. Balanced to within one expert per node.
        NodeId((expert as u64 * self.num_nodes as u64 / self.n_experts as u64) as u32)
    }

    /// Whether `expert` is owned by `node`.
    pub fn is_local(&self, node: NodeId, expert: u32) -> bool {
        self.owner(expert) == node
    }

    /// Half-open expert range `[start, end)` owned by `node`.
    pub fn range_for(&self, node: NodeId) -> (u32, u32) {
        // Invert the block mapping: start = ceil(node * E / N), end = ceil((node+1) * E / N).
        let e = self.n_experts as u64;
        let n = self.num_nodes as u64;
        let node = node.0 as u64;
        let start = (node * e).div_ceil(n) as u32;
        let end = ((node + 1) * e).div_ceil(n) as u32;
        (start, end)
    }

    /// Number of experts owned by `node`.
    pub fn count_for(&self, node: NodeId) -> u32 {
        let (s, e) = self.range_for(node);
        e - s
    }

    /// Iterator over the experts owned by `node`.
    pub fn local_experts(&self, node: NodeId) -> impl Iterator<Item = u32> {
        let (s, e) = self.range_for(node);
        s..e
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
}
