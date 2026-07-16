//! TCP transport for expert-parallel MoE over the ConnectX/RoCE **Ethernet**.
//!
//! RoCE is Ethernet, so plain TCP over the 192.168.100.0/24 fabric already gets
//! the ConnectX-7's 200 GbE bandwidth — just at higher latency than RDMA verbs.
//! This is the first *working* transport: it validates the whole expert-parallel
//! data plane end-to-end with no `libibverbs` dependency. [`RdmaTransport`] can
//! later replace it behind the same [`Transport`] trait to cut the round-trip
//! latency (and add GPUDirect); the engine above does not change.
//!
//! Wire protocol: each message is a `u32` little-endian length prefix followed by
//! that many payload bytes. A request is `sum_e weight_e * expert_e(x)` work for
//! the peer; the response is the peer's partial MoE sum. The server computes via a
//! caller-supplied handler (the engine, which owns the expert weights) — this crate
//! stays free of any dependency on the compute path.

use crate::sharding::NodeId;
use crate::transport::{ExpertRequest, ExpertResponse, Transport, TransportError};
use std::collections::HashMap;
use std::io::{self, Read, Write};
use std::net::{SocketAddr, TcpListener, TcpStream};
use std::sync::{Arc, Mutex};

const REQ_MAGIC: u32 = 0x4352_4551; // "CREQ"
const RSP_MAGIC: u32 = 0x4352_5350; // "CRSP"
const HELLO_MAGIC: u32 = 0x4348_454c; // "CHEL" — connect-time sharding handshake
const HACK_MAGIC: u32 = 0x4348_4143; // "CHAC" — handshake ack
/// Reject frames larger than this (guards against a bad length prefix -> OOM).
const MAX_FRAME: usize = 1 << 30; // 1 GiB

// ---- wire encode/decode ----------------------------------------------------

fn put_u32(b: &mut Vec<u8>, v: u32) {
    b.extend_from_slice(&v.to_le_bytes());
}
fn put_u64(b: &mut Vec<u8>, v: u64) {
    b.extend_from_slice(&v.to_le_bytes());
}
fn put_f32s(b: &mut Vec<u8>, v: &[f32]) {
    b.reserve(v.len() * 4);
    for &x in v {
        b.extend_from_slice(&x.to_le_bytes());
    }
}

struct Cur<'a> {
    b: &'a [u8],
    i: usize,
}
impl<'a> Cur<'a> {
    fn u32(&mut self) -> Option<u32> {
        let e = self.i + 4;
        let v = u32::from_le_bytes(self.b.get(self.i..e)?.try_into().ok()?);
        self.i = e;
        Some(v)
    }
    fn u64(&mut self) -> Option<u64> {
        let e = self.i + 8;
        let v = u64::from_le_bytes(self.b.get(self.i..e)?.try_into().ok()?);
        self.i = e;
        Some(v)
    }
    fn f32s(&mut self, n: usize) -> Option<Vec<f32>> {
        let e = self.i + n * 4;
        let s = self.b.get(self.i..e)?;
        self.i = e;
        Some(s.chunks_exact(4).map(|c| f32::from_le_bytes(c.try_into().unwrap())).collect())
    }
    fn u32s(&mut self, n: usize) -> Option<Vec<u32>> {
        let e = self.i + n * 4;
        let s = self.b.get(self.i..e)?;
        self.i = e;
        Some(s.chunks_exact(4).map(|c| u32::from_le_bytes(c.try_into().unwrap())).collect())
    }
}

/// Encode the connect-time hello: who we are and the fingerprint of our expert
/// sharding map. Sent as the first frame on every new connection, before any
/// activations, so a disagreeing peer is rejected before it can corrupt results.
pub fn encode_hello(node: NodeId, fingerprint: u64) -> Vec<u8> {
    let mut b = Vec::with_capacity(16);
    put_u32(&mut b, HELLO_MAGIC);
    put_u32(&mut b, node.0);
    put_u64(&mut b, fingerprint);
    b
}

/// Decode a hello → `(peer node, peer fingerprint)`.
pub fn decode_hello(b: &[u8]) -> Option<(NodeId, u64)> {
    let mut c = Cur { b, i: 0 };
    if c.u32()? != HELLO_MAGIC {
        return None;
    }
    let node = NodeId(c.u32()?);
    Some((node, c.u64()?))
}

/// Encode the hello ack: whether the peer accepted us, plus its own fingerprint
/// (so a rejected client can report both sides of the mismatch).
pub fn encode_hello_ack(ok: bool, fingerprint: u64) -> Vec<u8> {
    let mut b = Vec::with_capacity(16);
    put_u32(&mut b, HACK_MAGIC);
    put_u32(&mut b, ok as u32);
    put_u64(&mut b, fingerprint);
    b
}

/// Decode a hello ack → `(accepted, responder fingerprint)`.
pub fn decode_hello_ack(b: &[u8]) -> Option<(bool, u64)> {
    let mut c = Cur { b, i: 0 };
    if c.u32()? != HACK_MAGIC {
        return None;
    }
    let ok = c.u32()? != 0;
    Some((ok, c.u64()?))
}

/// Encode a request payload (without the length prefix).
pub fn encode_request(r: &ExpertRequest) -> Vec<u8> {
    let mut b = Vec::with_capacity(24 + r.experts.len() * 4 + (r.weights.len() + r.activations.len()) * 4);
    put_u32(&mut b, REQ_MAGIC);
    put_u32(&mut b, r.layer);
    put_u32(&mut b, r.n_tokens as u32);
    put_u32(&mut b, r.hidden as u32);
    put_u32(&mut b, r.experts.len() as u32);
    put_u32(&mut b, r.weights.len() as u32);
    for &e in &r.experts {
        put_u32(&mut b, e);
    }
    put_f32s(&mut b, &r.weights);
    put_f32s(&mut b, &r.activations);
    b
}

/// Decode a request payload.
pub fn decode_request(b: &[u8]) -> Option<ExpertRequest> {
    let mut c = Cur { b, i: 0 };
    if c.u32()? != REQ_MAGIC {
        return None;
    }
    let layer = c.u32()?;
    let n_tokens = c.u32()? as usize;
    let hidden = c.u32()? as usize;
    let n_experts = c.u32()? as usize;
    let n_weights = c.u32()? as usize;
    let experts = c.u32s(n_experts)?;
    let weights = c.f32s(n_weights)?;
    let activations = c.f32s(n_tokens.checked_mul(hidden)?)?;
    Some(ExpertRequest { experts, weights, activations, n_tokens, hidden, layer })
}

/// Encode a response payload (without the length prefix).
pub fn encode_response(r: &ExpertResponse) -> Vec<u8> {
    let mut b = Vec::with_capacity(12 + r.outputs.len() * 4);
    put_u32(&mut b, RSP_MAGIC);
    put_u32(&mut b, r.n_tokens as u32);
    put_u32(&mut b, r.hidden as u32);
    put_f32s(&mut b, &r.outputs);
    b
}

/// Decode a response payload.
pub fn decode_response(b: &[u8]) -> Option<ExpertResponse> {
    let mut c = Cur { b, i: 0 };
    if c.u32()? != RSP_MAGIC {
        return None;
    }
    let n_tokens = c.u32()? as usize;
    let hidden = c.u32()? as usize;
    let outputs = c.f32s(n_tokens.checked_mul(hidden)?)?;
    Some(ExpertResponse { outputs, n_tokens, hidden })
}

// ---- framing ---------------------------------------------------------------

fn write_frame(s: &mut impl Write, payload: &[u8]) -> io::Result<()> {
    s.write_all(&(payload.len() as u32).to_le_bytes())?;
    s.write_all(payload)?;
    s.flush()
}

fn read_frame(s: &mut impl Read) -> io::Result<Vec<u8>> {
    let mut len = [0u8; 4];
    s.read_exact(&mut len)?;
    let n = u32::from_le_bytes(len) as usize;
    if n > MAX_FRAME {
        return Err(io::Error::new(io::ErrorKind::InvalidData, "frame too large"));
    }
    let mut buf = vec![0u8; n];
    s.read_exact(&mut buf)?;
    Ok(buf)
}

// ---- server ----------------------------------------------------------------

/// Serve expert-compute requests on `listen`, computing each via `handler` (the
/// engine, which owns the resident expert weights). Spawns an accept loop on its
/// own thread and a thread per connection; returns immediately with the listener's
/// bound address so the caller can advertise it.
///
/// `fingerprint` is this node's [`ExpertSharding::fingerprint`](crate::ExpertSharding::fingerprint).
/// Every connection must open with a matching hello or it is refused before a single
/// activation is read — see [`serve_conn`]-level docs and
/// [`TransportError::FingerprintMismatch`](crate::TransportError::FingerprintMismatch).
pub fn serve_experts<F>(listen: SocketAddr, fingerprint: u64, handler: F) -> io::Result<SocketAddr>
where
    F: Fn(&ExpertRequest) -> ExpertResponse + Send + Sync + 'static,
{
    let listener = TcpListener::bind(listen)?;
    let addr = listener.local_addr()?;
    let handler = Arc::new(handler);
    std::thread::spawn(move || {
        for conn in listener.incoming() {
            match conn {
                Ok(stream) => {
                    let h = handler.clone();
                    std::thread::spawn(move || {
                        let _ = serve_conn(stream, fingerprint, &*h);
                    });
                }
                Err(_) => continue,
            }
        }
    });
    Ok(addr)
}

/// One connection: a mandatory sharding handshake, then request/response until EOF.
///
/// The first frame **must** be a hello whose fingerprint equals ours. A mismatch
/// means the peer built a different expert→node map, which would silently corrupt
/// results (experts computed twice or not at all), so we ack the rejection — telling
/// the peer our fingerprint so it can report both sides — and drop the connection
/// without ever calling the handler.
fn serve_conn<F>(mut stream: TcpStream, fingerprint: u64, handler: &F) -> io::Result<()>
where
    F: Fn(&ExpertRequest) -> ExpertResponse,
{
    let _ = stream.set_nodelay(true);

    let hello = read_frame(&mut stream)?;
    let (peer, peer_fp) = match decode_hello(&hello) {
        Some(h) => h,
        None => {
            return Err(io::Error::new(
                io::ErrorKind::InvalidData,
                "expected a sharding hello as the first frame",
            ))
        }
    };
    if peer_fp != fingerprint {
        let _ = write_frame(&mut stream, &encode_hello_ack(false, fingerprint));
        eprintln!(
            "[expert-server] REFUSED node {}: sharding fingerprint {:#018x} != ours {:#018x}. \
             All nodes must build the identical expert map (check COLI_SHARD and .coli_usage).",
            peer.0, peer_fp, fingerprint
        );
        return Err(io::Error::new(io::ErrorKind::InvalidData, "sharding fingerprint mismatch"));
    }
    write_frame(&mut stream, &encode_hello_ack(true, fingerprint))?;

    loop {
        let frame = read_frame(&mut stream)?; // Err on clean EOF ends the loop
        let req = match decode_request(&frame) {
            Some(r) => r,
            None => return Err(io::Error::new(io::ErrorKind::InvalidData, "bad request frame")),
        };
        let resp = handler(&req);
        write_frame(&mut stream, &encode_response(&resp))?;
    }
}

// ---- client ----------------------------------------------------------------

/// TCP client transport: reaches each peer node at a known socket address, with a
/// pooled connection per peer (reconnected on error). Every connection opens with a
/// sharding handshake, so a peer with a different expert map is refused before any
/// activations are sent.
pub struct TcpTransport {
    this: NodeId,
    peers: HashMap<NodeId, SocketAddr>,
    /// Our [`ExpertSharding::fingerprint`](crate::ExpertSharding::fingerprint), sent
    /// in the hello and checked against the peer's.
    fingerprint: u64,
    conns: Mutex<HashMap<NodeId, TcpStream>>,
}

impl TcpTransport {
    /// `this` is our node id; `peers` maps every *other* node to its
    /// `serve_experts` address; `fingerprint` is our expert-sharding map hash, which
    /// every peer must match.
    pub fn new(this: NodeId, peers: HashMap<NodeId, SocketAddr>, fingerprint: u64) -> TcpTransport {
        TcpTransport { this, peers, fingerprint, conns: Mutex::new(HashMap::new()) }
    }

    /// Open a connection to `node` and complete the sharding handshake. A refused
    /// hello surfaces as [`TransportError::FingerprintMismatch`], which callers must
    /// treat as fatal — the peer disagrees about who owns which expert.
    fn connect_and_hello(&self, node: NodeId, addr: SocketAddr) -> Result<TcpStream, TransportError> {
        let mut s = TcpStream::connect(addr).map_err(|e| TransportError::Io(e.to_string()))?;
        let _ = s.set_nodelay(true);
        write_frame(&mut s, &encode_hello(self.this, self.fingerprint))
            .map_err(|e| TransportError::Io(e.to_string()))?;
        let ack = read_frame(&mut s).map_err(|e| TransportError::Io(e.to_string()))?;
        let (ok, peer_fp) =
            decode_hello_ack(&ack).ok_or_else(|| TransportError::Io("bad handshake ack".into()))?;
        if !ok {
            return Err(TransportError::FingerprintMismatch {
                node: node.0,
                local: self.fingerprint,
                remote: peer_fp,
            });
        }
        Ok(s)
    }
}

impl Transport for TcpTransport {
    fn is_local(&self, node: NodeId) -> bool {
        node == self.this
    }

    fn this_node(&self) -> NodeId {
        self.this
    }

    /// Connect to every peer and complete the handshake now, so a cluster whose nodes
    /// disagree about the expert map dies at startup instead of mid-generation. Also
    /// surfaces unreachable peers — start the workers before the driver.
    fn verify_peers(&self) -> Result<(), TransportError> {
        let mut targets: Vec<(NodeId, SocketAddr)> =
            self.peers.iter().map(|(&n, &a)| (n, a)).collect();
        targets.sort_by_key(|(n, _)| n.0); // deterministic error order
        let mut conns = self.conns.lock().unwrap();
        for (node, addr) in targets {
            if conns.contains_key(&node) {
                continue;
            }
            let s = self.connect_and_hello(node, addr)?;
            conns.insert(node, s);
        }
        Ok(())
    }

    fn exchange(&self, node: NodeId, req: &ExpertRequest) -> Result<ExpertResponse, TransportError> {
        let addr = *self
            .peers
            .get(&node)
            .ok_or_else(|| TransportError::Io(format!("no address for node {}", node.0)))?;
        let payload = encode_request(req);

        // One round-trip on a pooled connection; reconnect once on a stale socket.
        let mut conns = self.conns.lock().unwrap();
        for attempt in 0..2 {
            if !conns.contains_key(&node) {
                match self.connect_and_hello(node, addr) {
                    Ok(s) => {
                        conns.insert(node, s);
                    }
                    // A map disagreement is fatal, not transient: never retry it and
                    // never fall through to sending activations to a peer that would
                    // compute the wrong experts.
                    Err(e @ TransportError::FingerprintMismatch { .. }) => return Err(e),
                    Err(e) => {
                        if attempt == 1 {
                            return Err(e);
                        }
                        continue;
                    }
                }
            }
            let stream = conns.get_mut(&node).unwrap();
            let r = write_frame(stream, &payload).and_then(|_| read_frame(stream));
            match r {
                Ok(frame) => {
                    return decode_response(&frame)
                        .ok_or_else(|| TransportError::Io("bad response frame".into()));
                }
                Err(e) => {
                    conns.remove(&node); // drop the broken connection, retry fresh
                    if attempt == 1 {
                        return Err(TransportError::Io(e.to_string()));
                    }
                }
            }
        }
        unreachable!()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn sample_req() -> ExpertRequest {
        ExpertRequest {
            experts: vec![130, 200, 255],
            weights: vec![0.5, 0.25, 0.25],
            activations: (0..8).map(|i| i as f32 * 0.1).collect(),
            n_tokens: 2,
            hidden: 4,
            layer: 7,
        }
    }

    #[test]
    fn request_roundtrip() {
        let r = sample_req();
        assert_eq!(decode_request(&encode_request(&r)).unwrap(), r);
    }

    #[test]
    fn response_roundtrip() {
        let r = ExpertResponse { outputs: vec![1.0, -2.0, 3.5, 0.0], n_tokens: 1, hidden: 4 };
        assert_eq!(decode_response(&encode_response(&r)).unwrap(), r);
    }

    #[test]
    fn decode_rejects_garbage() {
        assert!(decode_request(&[0, 1, 2]).is_none());
        assert!(decode_request(&[9, 9, 9, 9]).is_none()); // wrong magic
    }

    #[test]
    fn hello_roundtrip() {
        let b = encode_hello(NodeId(3), 0xdead_beef_cafe_f00d);
        assert_eq!(decode_hello(&b).unwrap(), (NodeId(3), 0xdead_beef_cafe_f00d));
        let a = encode_hello_ack(true, 0x1234_5678_9abc_def0);
        assert_eq!(decode_hello_ack(&a).unwrap(), (true, 0x1234_5678_9abc_def0));
        let r = encode_hello_ack(false, 7);
        assert_eq!(decode_hello_ack(&r).unwrap(), (false, 7));
        // Cross-decoding must fail on the magic, not silently succeed.
        assert!(decode_hello(&a).is_none());
        assert!(decode_hello_ack(&b).is_none());
        assert!(decode_hello(&[1, 2, 3]).is_none());
    }

    const FP: u64 = 0xa5a5_1234_5678_9abc;

    fn doubling_server(fingerprint: u64) -> SocketAddr {
        serve_experts("127.0.0.1:0".parse().unwrap(), fingerprint, |req| {
            let mut outputs: Vec<f32> = req.activations.iter().map(|x| x * 2.0).collect();
            outputs[0] = req.experts.len() as f32;
            ExpertResponse { outputs, n_tokens: req.n_tokens, hidden: req.hidden }
        })
        .unwrap()
    }

    #[test]
    fn tcp_exchange_end_to_end() {
        // Server doubles the activations and reports the expert count in outputs[0];
        // the client should get exactly that back over a real TCP socket.
        let addr = doubling_server(FP);

        let mut peers = HashMap::new();
        peers.insert(NodeId(1), addr);
        let t = TcpTransport::new(NodeId(0), peers, FP);
        assert!(t.is_local(NodeId(0)));
        assert!(!t.is_local(NodeId(1)));

        let req = sample_req();
        let resp = t.exchange(NodeId(1), &req).unwrap();
        assert_eq!(resp.n_tokens, 2);
        assert_eq!(resp.hidden, 4);
        assert_eq!(resp.outputs[0], 3.0); // 3 experts
        assert_eq!(resp.outputs[1], req.activations[1] * 2.0);

        // A second exchange reuses the pooled connection (handshake not repeated).
        let resp2 = t.exchange(NodeId(1), &req).unwrap();
        assert_eq!(resp2.outputs[0], 3.0);
    }

    #[test]
    fn matching_fingerprints_verify() {
        let addr = doubling_server(FP);
        let mut peers = HashMap::new();
        peers.insert(NodeId(1), addr);
        let t = TcpTransport::new(NodeId(0), peers, FP);
        t.verify_peers().expect("identical maps must verify");
    }

    #[test]
    fn mismatched_fingerprint_is_refused_not_retried() {
        // The peer built a different expert map. verify_peers must fail at startup
        // with both fingerprints, and exchange must refuse rather than send
        // activations to a node that would compute the wrong experts.
        let addr = doubling_server(FP);
        let mut peers = HashMap::new();
        peers.insert(NodeId(1), addr);
        let wrong = FP ^ 0xffff;
        let t = TcpTransport::new(NodeId(0), peers, wrong);

        match t.verify_peers().unwrap_err() {
            TransportError::FingerprintMismatch { node, local, remote } => {
                assert_eq!(node, 1);
                assert_eq!(local, wrong);
                assert_eq!(remote, FP);
            }
            e => panic!("expected FingerprintMismatch, got {e:?}"),
        }

        // And the data path refuses too — no silent wrong answer.
        assert!(matches!(
            t.exchange(NodeId(1), &sample_req()).unwrap_err(),
            TransportError::FingerprintMismatch { .. }
        ));
        // The error message must name the cause so an operator can act on it.
        let msg = t.exchange(NodeId(1), &sample_req()).unwrap_err().to_string();
        assert!(msg.contains("different expert sharding map"), "unhelpful: {msg}");
    }

    #[test]
    fn server_rejects_request_sent_without_hello() {
        // A client that skips the handshake must not get expert compute: the server
        // requires a hello as the very first frame.
        let addr = doubling_server(FP);
        let mut s = TcpStream::connect(addr).unwrap();
        write_frame(&mut s, &encode_request(&sample_req())).unwrap();
        // The server closes the connection instead of answering.
        assert!(read_frame(&mut s).is_err(), "server answered an un-handshaked request");
    }

    #[test]
    fn verify_peers_reports_unreachable_peer() {
        // Nothing is listening here; startup verification should surface it rather
        // than defer the failure to the first token.
        let mut peers = HashMap::new();
        peers.insert(NodeId(1), "127.0.0.1:1".parse::<SocketAddr>().unwrap());
        let t = TcpTransport::new(NodeId(0), peers, FP);
        assert!(matches!(t.verify_peers().unwrap_err(), TransportError::Io(_)));
    }

    #[test]
    fn exchange_unknown_peer_errors() {
        let t = TcpTransport::new(NodeId(0), HashMap::new(), FP);
        let e = t.exchange(NodeId(5), &sample_req()).unwrap_err();
        assert!(matches!(e, TransportError::Io(_)));
    }

    #[test]
    fn local_transport_verify_is_noop() {
        // Single node: nothing to agree with.
        assert!(crate::LocalTransport.verify_peers().is_ok());
    }
}
