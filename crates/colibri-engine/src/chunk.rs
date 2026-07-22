//! Sub-expert chunked-fetch scaffolding (Phase 3 of the multispark plan).
//!
//! Splits an expert's weight matrices into contiguous **O-band** chunks — a slice of
//! output rows across all K columns — each ≈ a target byte size. O-bands are contiguous
//! in the row-major packed buffer (unlike K-bands, which are strided), so one chunk is
//! one contiguous fetch: ideal both for pulling a band from a peer over RDMA and for
//! feeding the tiled kernel a block's worth of weights at a time.
//!
//! **Dormant.** Nothing wires this into the load/compute hot path yet (gated by
//! [`chunk_enabled`], `COLI_EXPERT_CHUNK`). It is the shared foundation for (a) the
//! multispark interconnect fetch (box 2 ← box 1, a band at a time) and (b) pipelining a
//! band's arrival with compute. Kept pure and unit-tested so the real RDMA transport
//! and the compute pipeline can be built on — and measured against — it once the e4m3
//! snapshot lands (int4 baseline was deleted; no measurement is possible before then).
//!
//! Chunk sizing: the measured floors are ~14 GB/s RDMA and 6.7 GB/s NVMe, both at
//! ≥1 MiB blocks — below ~1–2 MiB, per-op overhead eats bandwidth. So bands target
//! [`DEFAULT_CHUNK_BYTES`] and never drop below a hard 64 KiB floor.

use crate::moe::Expert;
use colibri_core::QTensor;
use std::io;

/// Default O-band target (2 MiB) — matches the reader pool's disk sub-chunk and stays
/// above the RDMA/NVMe efficiency knee.
pub const DEFAULT_CHUNK_BYTES: usize = 2 << 20;

/// Which of an expert's three projections a chunk belongs to.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum ExpertMatrix {
    Gate,
    Up,
    Down,
}

/// One contiguous O-band of one expert matrix: output rows `[o_start, o_start+o_count)`,
/// all K columns. The `(layer, eid, matrix, o_start, o_count)` tuple is the logical id a
/// peer resolves against its own copy; `byte_off`/`byte_len` locate the band inside that
/// matrix's packed weight buffer for local slicing.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct ChunkDesc {
    pub layer: usize,
    pub eid: usize,
    pub matrix: ExpertMatrix,
    pub o_start: usize,
    pub o_count: usize,
    pub byte_off: usize,
    pub byte_len: usize,
}

/// Bytes per output row for a packed weight of `i` input columns at `fmt`
/// (0=f32, 1=int8, 3=int2, 4=e4m3). Mirrors the backend's `row_bytes`.
pub fn row_bytes(fmt: i32, i: usize) -> usize {
    match fmt {
        0 => i * 4,
        1 | 4 => i,
        3 => i.div_ceil(4),
        _ => 0,
    }
}

/// Plan the O-band chunks for one matrix `[o, i]` at `fmt`, each ≈ `target_bytes` (the
/// last band is the remainder). The bands tile the row range exactly and contiguously,
/// so reassembling them reproduces the matrix byte-for-byte.
pub fn plan_matrix_chunks(
    layer: usize,
    eid: usize,
    matrix: ExpertMatrix,
    o: usize,
    i: usize,
    fmt: i32,
    target_bytes: usize,
) -> Vec<ChunkDesc> {
    let rb = row_bytes(fmt, i);
    let mut out = Vec::new();
    if rb == 0 || o == 0 {
        return out;
    }
    let band_rows = (target_bytes / rb).max(1);
    let mut o0 = 0;
    while o0 < o {
        let cnt = band_rows.min(o - o0);
        out.push(ChunkDesc {
            layer,
            eid,
            matrix,
            o_start: o0,
            o_count: cnt,
            byte_off: o0 * rb,
            byte_len: cnt * rb,
        });
        o0 += cnt;
    }
    out
}

/// Plan chunks for a whole expert (gate, then up, then down) at `target_bytes`.
pub fn plan_expert_chunks(
    layer: usize,
    eid: usize,
    ex: &Expert,
    target_bytes: usize,
) -> Vec<ChunkDesc> {
    let mut v = plan_matrix_chunks(
        layer, eid, ExpertMatrix::Gate,
        ex.gate.o as usize, ex.gate.i as usize, ex.gate.fmt_code, target_bytes,
    );
    v.extend(plan_matrix_chunks(
        layer, eid, ExpertMatrix::Up,
        ex.up.o as usize, ex.up.i as usize, ex.up.fmt_code, target_bytes,
    ));
    v.extend(plan_matrix_chunks(
        layer, eid, ExpertMatrix::Down,
        ex.down.o as usize, ex.down.i as usize, ex.down.fmt_code, target_bytes,
    ));
    v
}

/// The packed weight buffer of one of an expert's matrices. Routed experts are always
/// q4-packed (int2/e4m3); int8/f32 experts aren't streamed, so they're rejected.
pub fn matrix_weight_bytes(ex: &Expert, m: ExpertMatrix) -> io::Result<&[u8]> {
    let t: &QTensor = match m {
        ExpertMatrix::Gate => &ex.gate,
        ExpertMatrix::Up => &ex.up,
        ExpertMatrix::Down => &ex.down,
    };
    match t.fmt_code {
        3 | 4 => Ok(t.q4.as_slice()),
        other => Err(io::Error::new(
            io::ErrorKind::Unsupported,
            format!("chunk fetch: fmt {other} experts unsupported (routed experts are q4-packed)"),
        )),
    }
}

/// A source of chunk bytes — abstracts over a locally-resident expert vs a remote peer
/// (the RDMA fetch, to come). `fetch_into` fills `dst`, whose length must equal
/// `d.byte_len`.
pub trait ChunkSource {
    fn fetch_into(&self, d: &ChunkDesc, dst: &mut [u8]) -> io::Result<()>;
}

/// Local chunk source: slices bands straight out of a resident [`Expert`]. Behaviorally
/// a memcpy — the reference the remote transport must reproduce byte-for-byte.
pub struct LocalChunkSource<'a> {
    pub expert: &'a Expert,
}

impl ChunkSource for LocalChunkSource<'_> {
    fn fetch_into(&self, d: &ChunkDesc, dst: &mut [u8]) -> io::Result<()> {
        let src = matrix_weight_bytes(self.expert, d.matrix)?;
        let end = d.byte_off + d.byte_len;
        if end > src.len() || dst.len() != d.byte_len {
            return Err(io::Error::new(io::ErrorKind::InvalidInput, "chunk out of range"));
        }
        dst.copy_from_slice(&src[d.byte_off..end]);
        Ok(())
    }
}

/// `COLI_EXPERT_CHUNK=1` will switch the loader/fetch onto the chunked path. Off by
/// default; nothing reads this yet (scaffolding).
pub fn chunk_enabled() -> bool {
    static ON: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *ON.get_or_init(|| std::env::var("COLI_EXPERT_CHUNK").ok().as_deref() == Some("1"))
}

/// Chunk-byte target from `COLI_EXPERT_CHUNK_BYTES` (default [`DEFAULT_CHUNK_BYTES`]),
/// clamped to ≥64 KiB so a misconfiguration can't drop below RDMA efficiency.
pub fn chunk_bytes() -> usize {
    static N: std::sync::OnceLock<usize> = std::sync::OnceLock::new();
    *N.get_or_init(|| {
        std::env::var("COLI_EXPERT_CHUNK_BYTES")
            .ok()
            .and_then(|v| v.parse().ok())
            .unwrap_or(DEFAULT_CHUNK_BYTES)
            .max(64 << 10)
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use colibri_core::{Bytes, QTensor};

    fn qt(fmt: i32, o: i32, i: i32) -> QTensor {
        let rb = row_bytes(fmt, i as usize);
        let bytes: Vec<u8> = (0..o as usize * rb).map(|k| (k % 251) as u8).collect();
        QTensor {
            fmt_code: fmt,
            o,
            i,
            q4: Bytes::Owned(bytes),
            s: vec![1.0; o as usize],
            ..Default::default()
        }
    }

    fn mk_expert() -> Expert {
        // e4m3-shaped (fmt=4, 1 B/weight): gate/up [8,16], down [16,8]
        Expert { gate: qt(4, 8, 16), up: qt(4, 8, 16), down: qt(4, 16, 8) }
    }

    #[test]
    fn chunks_tile_the_matrix_exactly() {
        // target 32 B, row = 16 B → 2 rows/band, o=8 → 4 bands, contiguous, cover all
        let cs = plan_matrix_chunks(0, 0, ExpertMatrix::Gate, 8, 16, 4, 32);
        assert_eq!(cs.len(), 4);
        let (mut off, mut rows) = (0usize, 0usize);
        for c in &cs {
            assert_eq!(c.byte_off, off);
            assert_eq!(c.byte_len, c.o_count * 16);
            off += c.byte_len;
            rows += c.o_count;
        }
        assert_eq!((rows, off), (8, 8 * 16));
    }

    #[test]
    fn last_band_is_the_remainder() {
        // o=8, 3 rows/band (target 48, row 16) → 3, 3, 2
        let cs = plan_matrix_chunks(0, 0, ExpertMatrix::Gate, 8, 16, 4, 48);
        let counts: Vec<usize> = cs.iter().map(|c| c.o_count).collect();
        assert_eq!(counts, vec![3, 3, 2]);
    }

    #[test]
    fn tiny_target_still_makes_one_row_bands_not_zero() {
        // target below one row must not divide to zero rows/band (infinite loop guard)
        let cs = plan_matrix_chunks(0, 0, ExpertMatrix::Down, 4, 8, 4, 1);
        assert_eq!(cs.len(), 4);
        assert!(cs.iter().all(|c| c.o_count == 1));
    }

    #[test]
    fn local_fetch_reassembles_each_matrix() {
        let ex = mk_expert();
        let descs = plan_expert_chunks(0, 5, &ex, 32);
        let src = LocalChunkSource { expert: &ex };
        for m in [ExpertMatrix::Gate, ExpertMatrix::Up, ExpertMatrix::Down] {
            let orig = matrix_weight_bytes(&ex, m).unwrap().to_vec();
            let mut re = vec![0u8; orig.len()];
            for d in descs.iter().filter(|d| d.matrix == m) {
                let mut buf = vec![0u8; d.byte_len];
                src.fetch_into(d, &mut buf).unwrap();
                re[d.byte_off..d.byte_off + d.byte_len].copy_from_slice(&buf);
            }
            assert_eq!(re, orig, "{m:?} reassembly mismatch");
        }
    }

    #[test]
    fn fetch_rejects_out_of_range() {
        let ex = mk_expert();
        let bad = ChunkDesc {
            layer: 0, eid: 0, matrix: ExpertMatrix::Gate,
            o_start: 0, o_count: 1, byte_off: 10_000, byte_len: 16,
        };
        let mut buf = vec![0u8; 16];
        assert!(LocalChunkSource { expert: &ex }.fetch_into(&bad, &mut buf).is_err());
    }
}
