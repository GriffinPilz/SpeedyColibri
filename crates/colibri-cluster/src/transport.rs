//! Inter-node transport for expert-parallel MoE.
//!
//! When a token routes to an expert owned by a remote node, the owning node must
//! compute that expert over the token's activation and return the result. This
//! module defines the transport that carries those activations and expert
//! outputs between DGX Spark nodes.
//!
//! # Target: RDMA/RoCE over ConnectX-7
//!
//! Two DGX Sparks are joined by a 200 GbE ConnectX-7 link. The intended
//! implementation uses RDMA verbs (`libibverbs`) directly:
//!   - register the activation and expert-output buffers as memory regions (MRs),
//!     ideally GPU memory via GPUDirect RDMA so the Blackwell GPU DMAs straight to
//!     the wire with no host bounce;
//!   - a queue pair (QP) per peer; `post_send`/`post_recv` for the activation
//!     round-trip; completion queue polling for latency.
//!
//! # Status
//!
//! Topology is **single-node first**, so [`LocalTransport`] (everything in-process)
//! is the only live implementation and the engine's MoE block uses it
//! unconditionally. [`RdmaTransport`] is the API-shape placeholder for the
//! multi-node build, behind the `rdma` feature; its methods are `todo!`-free
//! stubs that return [`TransportError::NotConnected`] until wired.

use crate::sharding::NodeId;

/// A batch of activations to compute experts over, addressed to a remote node.
///
/// Layout is deliberately flat (row-major `[n_tokens, hidden]`) so it maps to a
/// single registered memory region for RDMA.
#[derive(Debug, Clone, PartialEq)]
pub struct ExpertRequest {
    /// experts (global ids) the peer should apply, all owned by the target node
    pub experts: Vec<u32>,
    /// routing weight for each expert in `experts` (per the requester's router).
    /// The owner returns `sum_e weight[e] * expert_e(x)` so the response is one
    /// `[n_tokens, hidden]` partial sum regardless of how many experts it owns.
    /// For the batched (`n_tokens > 1`) case, weights are `[n_tokens * n_experts]`
    /// row-major (per token, per expert); for `n_tokens == 1` it is just per expert.
    pub weights: Vec<f32>,
    /// token activations, `[n_tokens * hidden]` row-major f32
    pub activations: Vec<f32>,
    pub n_tokens: usize,
    pub hidden: usize,
    /// which transformer layer these experts belong to
    pub layer: u32,
}

/// The peer's expert outputs, to be scattered back into the local MoE sum.
#[derive(Debug, Clone, PartialEq)]
pub struct ExpertResponse {
    /// `[n_tokens * hidden]` row-major f32, already weighted+summed over the
    /// peer's experts for each token (partial MoE contribution).
    pub outputs: Vec<f32>,
    pub n_tokens: usize,
    pub hidden: usize,
}

/// A batch of layer-input activations for a peer to run **its head slice** of MLA
/// attention over (tensor-parallel attention — the analogue of [`ExpertRequest`] for
/// the attention block). The peer computes the projections, KV, and the DSA-sparse
/// core for heads `[h_start, h_start+h_count)`, then its o-projection, and returns the
/// partial `[n_tokens, hidden]` output. The driver adds the partials from every head
/// slice to reconstruct full attention.
///
/// Prefill-only: `pos_base == 0` and the peer builds a fresh KV from these activations
/// (no cross-request state), so the handler is a pure function.
#[derive(Debug, Clone, PartialEq)]
pub struct AttnRequest {
    /// post-`in_ln` layer input, `[n_tokens * hidden]` row-major f32
    pub activations: Vec<f32>,
    /// the driver's DSA selection: `sel[q]` = the cached positions query `q` attends
    /// to (empty = dense/causal for that query). Shipped so every node's sparse
    /// attention uses the *identical* selection — the peer never runs the indexer, so
    /// there is no selection to diverge or carry across layers. Empty outer vec = no
    /// selection (fully dense attention).
    pub sel: Vec<Vec<u32>>,
    pub n_tokens: usize,
    pub hidden: usize,
    /// position of the first token (0 for single-shot prefill).
    pub pos_base: u32,
    /// first head this node computes, and how many.
    pub h_start: u32,
    pub h_count: u32,
    /// which transformer layer.
    pub layer: u32,
}

/// The peer's attention output for its head slice: a partial `[n_tokens, hidden]` to
/// be summed into the full attention output on the driver.
#[derive(Debug, Clone, PartialEq)]
pub struct AttnResponse {
    /// `[n_tokens * hidden]` row-major f32 — this head slice's o-projected partial.
    pub outputs: Vec<f32>,
    pub n_tokens: usize,
    pub hidden: usize,
}

/// Transport errors.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum TransportError {
    /// No connection to the peer (single-node build, or not yet wired).
    NotConnected,
    /// The RDMA transport is not compiled in (`rdma` feature off).
    RdmaUnavailable,
    /// The peer's expert→node map differs from ours. Fatal and **not retryable**:
    /// with disagreeing maps each side ships experts to the node it *thinks* owns
    /// them, so some experts are computed twice and others never — silently wrong
    /// output. Refuse the connection instead.
    FingerprintMismatch { node: u32, local: u64, remote: u64 },
    Io(String),
}

impl std::fmt::Display for TransportError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            TransportError::NotConnected => write!(f, "transport: no peer connection"),
            TransportError::RdmaUnavailable => {
                write!(f, "transport: RDMA support not compiled in (enable feature `rdma`)")
            }
            TransportError::FingerprintMismatch { node, local, remote } => write!(
                f,
                "transport: node {node} has a different expert sharding map \
                 (ours {local:#018x}, theirs {remote:#018x}). Every node must build the \
                 identical map — check that COLI_SHARD and the usage history (.coli_usage) \
                 match on all nodes. Refusing to run: mismatched maps corrupt results silently."
            ),
            TransportError::Io(s) => write!(f, "transport io error: {s}"),
        }
    }
}

impl std::error::Error for TransportError {}

/// Carries expert work to/from peer nodes.
///
/// The engine calls [`Transport::is_local`] first; only non-local experts go
/// through [`Transport::exchange`]. On a single node everything is local and
/// `exchange` is never called.
pub trait Transport: Send + Sync {
    /// Whether `node` is this process (so its experts run in-process).
    fn is_local(&self, node: NodeId) -> bool;

    /// This node's own id.
    fn this_node(&self) -> NodeId;

    /// Send an expert request to `node` and block for its response.
    ///
    /// For [`LocalTransport`] this is unreachable (the engine never routes local
    /// experts through the transport); it errors rather than pretending.
    fn exchange(
        &self,
        node: NodeId,
        req: &ExpertRequest,
    ) -> Result<ExpertResponse, TransportError>;

    /// Send an attention request (this node's head slice) to `node` and block for its
    /// partial. Same connection/transport as [`Transport::exchange`], different payload.
    ///
    /// Default errors — only the real multi-node transports implement it.
    fn exchange_attn(
        &self,
        _node: NodeId,
        _req: &AttnRequest,
    ) -> Result<AttnResponse, TransportError> {
        Err(TransportError::NotConnected)
    }

    /// Handshake with every peer up front and confirm they agree on the expert
    /// sharding map, so a misconfigured cluster fails at startup rather than
    /// silently producing wrong tokens. Implementations that carry a fingerprint
    /// should return [`TransportError::FingerprintMismatch`] on disagreement.
    ///
    /// Default: no peers to verify (single-node), so `Ok`.
    fn verify_peers(&self) -> Result<(), TransportError> {
        Ok(())
    }
}

/// Single-node transport: this process is the whole cluster.
#[derive(Debug, Clone, Copy)]
pub struct LocalTransport;

impl Transport for LocalTransport {
    fn is_local(&self, node: NodeId) -> bool {
        node == NodeId(0)
    }

    fn this_node(&self) -> NodeId {
        NodeId(0)
    }

    fn exchange(
        &self,
        _node: NodeId,
        _req: &ExpertRequest,
    ) -> Result<ExpertResponse, TransportError> {
        // On a single node no expert is remote, so this path is never taken. If
        // it is, that's a routing bug — surface it instead of fabricating output.
        Err(TransportError::NotConnected)
    }
}

/// RDMA/RoCE transport over ConnectX-7 — multi-node target.
///
/// Placeholder for the `rdma`-feature build. Real implementation will hold a
/// queue pair per peer plus registered (ideally GPU) memory regions; see the
/// module docs. Until wired, `exchange` reports `RdmaUnavailable`.
#[cfg(feature = "rdma")]
pub struct RdmaTransport {
    this: NodeId,
    // TODO(rdma): ibverbs context, per-peer QPs, CQ, registered MRs.
}

#[cfg(feature = "rdma")]
impl RdmaTransport {
    /// Establish RDMA connections to all peers. TODO: verbs setup + QP handshake.
    pub fn connect(this: NodeId) -> Result<RdmaTransport, TransportError> {
        let _ = this; // TODO(rdma): store on the constructed transport
        Err(TransportError::RdmaUnavailable)
    }
}

#[cfg(feature = "rdma")]
impl Transport for RdmaTransport {
    fn is_local(&self, node: NodeId) -> bool {
        node == self.this
    }
    fn this_node(&self) -> NodeId {
        self.this
    }
    fn exchange(
        &self,
        _node: NodeId,
        _req: &ExpertRequest,
    ) -> Result<ExpertResponse, TransportError> {
        // TODO(rdma): post_send(req MR) -> poll CQ -> post_recv(resp MR).
        Err(TransportError::RdmaUnavailable)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn local_transport_is_single_node() {
        let t = LocalTransport;
        assert_eq!(t.this_node(), NodeId(0));
        assert!(t.is_local(NodeId(0)));
        assert!(!t.is_local(NodeId(1)));
    }

    #[test]
    fn local_exchange_is_never_valid() {
        // Routing a "remote" expert on a single node is a bug; the transport must
        // not silently return zeros.
        let t = LocalTransport;
        let req = ExpertRequest {
            experts: vec![0],
            weights: vec![1.0],
            activations: vec![0.0; 4],
            n_tokens: 1,
            hidden: 4,
            layer: 0,
        };
        assert_eq!(t.exchange(NodeId(1), &req), Err(TransportError::NotConnected));
    }
}
