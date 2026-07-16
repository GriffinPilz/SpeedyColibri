//! Multi-node distribution for the colibrì engine (DGX Spark).
//!
//! Deployment target: a Docker container on NVIDIA DGX Spark (GB10 Grace
//! Blackwell, aarch64 + CUDA, 128 GB unified memory per node). The eventual
//! cluster is **expert-parallel** — the MoE experts are sharded across nodes and
//! exchanged over an RDMA/RoCE link (ConnectX-7 200 GbE).
//!
//! Current topology is **single node**: [`ExpertSharding::single`] +
//! [`LocalTransport`] make every expert local, so the engine runs unmodified on
//! one box while the split points are already designed in. Flip to a real
//! `num_nodes > 1` sharding and an RDMA transport to distribute.
//!
//! See DEPLOYMENT.md for the container and topology details.

pub mod discovery;
pub mod net;
pub mod sharding;
pub mod transport;

pub use discovery::{connectx_links, discover, ConnectXLink, Discovery, Peer, PeerKind};
pub use net::{serve_experts, TcpTransport};
pub use sharding::{ExpertSharding, NodeId};
pub use transport::{ExpertRequest, ExpertResponse, LocalTransport, Transport, TransportError};

/// Cluster-wide configuration derived from the runtime environment.
#[derive(Debug, Clone)]
pub struct ClusterConfig {
    pub num_nodes: u32,
    pub this_node: NodeId,
}

impl ClusterConfig {
    /// The single-node default: this process is the whole cluster.
    pub fn single() -> ClusterConfig {
        ClusterConfig {
            num_nodes: 1,
            this_node: NodeId(0),
        }
    }

    /// Read the cluster layout from the environment, mirroring how the container
    /// entrypoint will inject it: `COLI_NUM_NODES` and `COLI_NODE_RANK`. Absent or
    /// unparseable values fall back to single-node.
    pub fn from_env() -> ClusterConfig {
        let num_nodes = std::env::var("COLI_NUM_NODES")
            .ok()
            .and_then(|s| s.parse::<u32>().ok())
            .filter(|&n| n >= 1)
            .unwrap_or(1);
        let rank = std::env::var("COLI_NODE_RANK")
            .ok()
            .and_then(|s| s.parse::<u32>().ok())
            .filter(|&r| r < num_nodes)
            .unwrap_or(0);
        ClusterConfig {
            num_nodes,
            this_node: NodeId(rank),
        }
    }

    pub fn is_single_node(&self) -> bool {
        self.num_nodes == 1
    }

    /// Build the expert sharding for this cluster over `n_experts`.
    pub fn expert_sharding(&self, n_experts: u32) -> ExpertSharding {
        ExpertSharding::new(self.num_nodes, n_experts)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn single_node_config() {
        let c = ClusterConfig::single();
        assert!(c.is_single_node());
        assert_eq!(c.this_node, NodeId(0));
        let sh = c.expert_sharding(256);
        assert_eq!(sh.count_for(NodeId(0)), 256);
    }
}
