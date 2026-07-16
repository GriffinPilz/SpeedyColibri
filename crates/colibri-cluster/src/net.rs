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
/// Reject frames larger than this (guards against a bad length prefix -> OOM).
const MAX_FRAME: usize = 1 << 30; // 1 GiB

// ---- wire encode/decode ----------------------------------------------------

fn put_u32(b: &mut Vec<u8>, v: u32) {
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
pub fn serve_experts<F>(listen: SocketAddr, handler: F) -> io::Result<SocketAddr>
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
                        let _ = serve_conn(stream, &*h);
                    });
                }
                Err(_) => continue,
            }
        }
    });
    Ok(addr)
}

fn serve_conn<F>(mut stream: TcpStream, handler: &F) -> io::Result<()>
where
    F: Fn(&ExpertRequest) -> ExpertResponse,
{
    let _ = stream.set_nodelay(true);
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
/// pooled connection per peer (reconnected on error).
pub struct TcpTransport {
    this: NodeId,
    peers: HashMap<NodeId, SocketAddr>,
    conns: Mutex<HashMap<NodeId, TcpStream>>,
}

impl TcpTransport {
    /// `this` is our node id; `peers` maps every *other* node to its
    /// `serve_experts` address.
    pub fn new(this: NodeId, peers: HashMap<NodeId, SocketAddr>) -> TcpTransport {
        TcpTransport { this, peers, conns: Mutex::new(HashMap::new()) }
    }
}

impl Transport for TcpTransport {
    fn is_local(&self, node: NodeId) -> bool {
        node == self.this
    }

    fn this_node(&self) -> NodeId {
        self.this
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
                let s = TcpStream::connect(addr).map_err(|e| TransportError::Io(e.to_string()))?;
                let _ = s.set_nodelay(true);
                conns.insert(node, s);
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
    fn tcp_exchange_end_to_end() {
        // Server doubles the activations and reports the expert count in outputs[0];
        // the client should get exactly that back over a real TCP socket.
        let addr = serve_experts("127.0.0.1:0".parse().unwrap(), |req| {
            let mut outputs: Vec<f32> = req.activations.iter().map(|x| x * 2.0).collect();
            outputs[0] = req.experts.len() as f32;
            ExpertResponse { outputs, n_tokens: req.n_tokens, hidden: req.hidden }
        })
        .unwrap();

        let mut peers = HashMap::new();
        peers.insert(NodeId(1), addr);
        let t = TcpTransport::new(NodeId(0), peers);
        assert!(t.is_local(NodeId(0)));
        assert!(!t.is_local(NodeId(1)));

        let req = sample_req();
        let resp = t.exchange(NodeId(1), &req).unwrap();
        assert_eq!(resp.n_tokens, 2);
        assert_eq!(resp.hidden, 4);
        assert_eq!(resp.outputs[0], 3.0); // 3 experts
        assert_eq!(resp.outputs[1], req.activations[1] * 2.0);

        // A second exchange reuses the pooled connection.
        let resp2 = t.exchange(NodeId(1), &req).unwrap();
        assert_eq!(resp2.outputs[0], 3.0);
    }

    #[test]
    fn exchange_unknown_peer_errors() {
        let t = TcpTransport::new(NodeId(0), HashMap::new());
        let e = t.exchange(NodeId(5), &sample_req()).unwrap_err();
        assert!(matches!(e, TransportError::Io(_)));
    }
}
