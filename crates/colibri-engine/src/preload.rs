//! Parallel expert preloading — repack the routed experts into `num_files`
//! contiguous binary shards (one per CPU core) plus a manifest, then read all
//! shards **simultaneously** (one thread per shard) to saturate the NVMe on cold
//! start.
//!
//! The stock path streams each expert from scattered tensors inside the original
//! safetensors files — many small, random `pread`s, effectively serial. Here:
//!   - [`repack`] walks every routed expert once and writes its weights as a
//!     contiguous blob into one of `N` shard files (round-robin, so the shards
//!     are byte-balanced), recording `(layer, eid) -> (file, offset)` in a
//!     [`Manifest`];
//!   - [`PreloadStore::load`] opens the `N` shards and reads them in parallel —
//!     each thread does one large sequential scan of its shard — reconstructing
//!     the [`Expert`]s directly in RAM.
//!
//! A blob is `gate.q | gate.scales | up.q | up.scales | down.q | down.scales`;
//! the layer's `(fmt, dims)` in the manifest say how to slice it. int4 (fmt 2)
//! and int8 (fmt 1) experts are supported.

use crate::moe::{expert_gate_name, load_expert, Expert, ExpertProvider};
use colibri_core::{Config, QTensor};
use colibri_json::Json;
use colibri_safetensors::Shards;
use std::collections::HashMap;
use std::fs::File;
use std::io::{self, Write};
use std::path::{Path, PathBuf};
use std::sync::Arc;

/// Location of one expert's blob within the shards.
#[derive(Debug, Clone, Copy)]
pub struct ExpertLoc {
    pub layer: usize,
    pub eid: usize,
    pub file: usize,
    pub offset: u64,
}

/// Describes a repacked expert set: shard count, dims, per-layer blob layout, and
/// every expert's location.
#[derive(Debug, Clone)]
pub struct Manifest {
    pub num_files: usize,
    pub hidden: usize,
    pub moe_inter: usize,
    /// per sparse layer: `(layer, fmt, expert_blob_bytes)`
    pub layers: Vec<(usize, i32, u64)>,
    pub experts: Vec<ExpertLoc>,
}

impl Manifest {
    fn layer_meta(&self) -> HashMap<usize, (i32, u64)> {
        self.layers.iter().map(|&(l, f, b)| (l, (f, b))).collect()
    }

    /// Total bytes across all shards.
    pub fn total_bytes(&self) -> u64 {
        let meta = self.layer_meta();
        self.experts
            .iter()
            .map(|e| meta.get(&e.layer).map(|&(_, b)| b).unwrap_or(0))
            .sum()
    }

    pub fn shard_path(dir: &Path, file: usize) -> PathBuf {
        dir.join(format!("experts_{file:04}.bin"))
    }

    /// Write the manifest JSON (compact arrays).
    pub fn save(&self, path: impl AsRef<Path>) -> io::Result<()> {
        let mut s = format!(
            "{{\"num_files\":{},\"hidden\":{},\"moe_inter\":{},\"layers\":[",
            self.num_files, self.hidden, self.moe_inter
        );
        for (i, (l, f, b)) in self.layers.iter().enumerate() {
            if i > 0 {
                s.push(',');
            }
            s.push_str(&format!("[{l},{f},{b}]"));
        }
        s.push_str("],\"experts\":[");
        for (i, e) in self.experts.iter().enumerate() {
            if i > 0 {
                s.push(',');
            }
            s.push_str(&format!("[{},{},{},{}]", e.layer, e.eid, e.file, e.offset));
        }
        s.push_str("]}");
        std::fs::write(path, s)
    }

    /// Parse a manifest JSON.
    pub fn load(path: impl AsRef<Path>) -> io::Result<Manifest> {
        let text = std::fs::read_to_string(path)?;
        let j = Json::parse(&text)
            .ok_or_else(|| io::Error::new(io::ErrorKind::InvalidData, "empty manifest"))?;
        let num = |k: &str| j.get(k).and_then(Json::as_i64).unwrap_or(0);
        let layers = j
            .get("layers")
            .and_then(Json::as_array)
            .map(|a| {
                a.iter()
                    .filter_map(Json::as_array)
                    .map(|e| {
                        (
                            e[0].as_i64().unwrap_or(0) as usize,
                            e[1].as_i64().unwrap_or(0) as i32,
                            e[2].as_i64().unwrap_or(0) as u64,
                        )
                    })
                    .collect()
            })
            .unwrap_or_default();
        let experts = j
            .get("experts")
            .and_then(Json::as_array)
            .map(|a| {
                a.iter()
                    .filter_map(Json::as_array)
                    .map(|e| ExpertLoc {
                        layer: e[0].as_i64().unwrap_or(0) as usize,
                        eid: e[1].as_i64().unwrap_or(0) as usize,
                        file: e[2].as_i64().unwrap_or(0) as usize,
                        offset: e[3].as_i64().unwrap_or(0) as u64,
                    })
                    .collect()
            })
            .unwrap_or_default();
        Ok(Manifest {
            num_files: num("num_files") as usize,
            hidden: num("hidden") as usize,
            moe_inter: num("moe_inter") as usize,
            layers,
            experts,
        })
    }
}

/// Byte lengths of the six sub-tensors in an expert blob.
struct BlobLayout {
    gate_q: usize,
    gate_s: usize,
    up_q: usize,
    up_s: usize,
    down_q: usize,
    down_s: usize,
}

fn layout(hidden: usize, moe_inter: usize, fmt: i32) -> BlobLayout {
    // gate/up: [moe_inter, hidden]; down: [hidden, moe_inter]
    let (gq, dq) = match fmt {
        1 => (moe_inter * hidden, hidden * moe_inter), // int8: 1 byte/param
        2 => (
            moe_inter * hidden.div_ceil(2),
            hidden * moe_inter.div_ceil(2),
        ), // int4: 2/byte
        f => panic!("preload supports int8/int4 experts only, got fmt {f}"),
    };
    BlobLayout {
        gate_q: gq,
        gate_s: moe_inter * 4,
        up_q: gq,
        up_s: moe_inter * 4,
        down_q: dq,
        down_s: hidden * 4,
    }
}

fn blob_bytes(hidden: usize, moe_inter: usize, fmt: i32) -> usize {
    let l = layout(hidden, moe_inter, fmt);
    l.gate_q + l.gate_s + l.up_q + l.up_s + l.down_q + l.down_s
}

fn push_q(b: &mut Vec<u8>, t: &QTensor) {
    match t.fmt_code {
        1 => b.extend(t.q8.iter().map(|&x| x as u8)),
        2 => b.extend_from_slice(&t.q4),
        f => panic!("preload: unsupported expert fmt {f}"),
    }
}

fn push_s(b: &mut Vec<u8>, s: &[f32]) {
    for &v in s {
        b.extend_from_slice(&v.to_le_bytes());
    }
}

/// Serialize an expert to its blob.
fn expert_blob(ex: &Expert) -> Vec<u8> {
    let mut b = Vec::new();
    push_q(&mut b, &ex.gate);
    push_s(&mut b, &ex.gate.s);
    push_q(&mut b, &ex.up);
    push_s(&mut b, &ex.up.s);
    push_q(&mut b, &ex.down);
    push_s(&mut b, &ex.down.s);
    b
}

fn read_qt(blob: &[u8], off: &mut usize, qlen: usize, slen: usize, o: usize, i: usize, fmt: i32) -> QTensor {
    let q = &blob[*off..*off + qlen];
    *off += qlen;
    let sb = &blob[*off..*off + slen];
    *off += slen;
    let s: Vec<f32> = sb
        .chunks_exact(4)
        .map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]]))
        .collect();
    let mut t = QTensor {
        fmt_code: fmt,
        o: o as i32,
        i: i as i32,
        s,
        ..Default::default()
    };
    match fmt {
        1 => t.q8 = q.iter().map(|&b| b as i8).collect(),
        2 => t.q4 = q.to_vec().into(),
        f => panic!("preload: unsupported expert fmt {f}"),
    }
    t
}

/// Reconstruct an expert from its blob given the layer dims and format.
fn expert_from_blob(blob: &[u8], hidden: usize, moe_inter: usize, fmt: i32) -> Expert {
    let l = layout(hidden, moe_inter, fmt);
    let mut off = 0;
    let gate = read_qt(blob, &mut off, l.gate_q, l.gate_s, moe_inter, hidden, fmt);
    let up = read_qt(blob, &mut off, l.up_q, l.up_s, moe_inter, hidden, fmt);
    let down = read_qt(blob, &mut off, l.down_q, l.down_s, hidden, moe_inter, fmt);
    Expert { gate, up, down }
}

/// Default shard count: one per CPU core.
pub fn default_num_files() -> usize {
    std::thread::available_parallelism().map(|n| n.get()).unwrap_or(4)
}

/// Repack every routed expert (sparse layers) from `provider` into `num_files`
/// contiguous shard files + a manifest under `out_dir`. One-time preprocessing;
/// the payoff is [`PreloadStore::load`]'s parallel read.
pub fn repack<P: ExpertProvider>(
    provider: &P,
    cfg: &Config,
    out_dir: &Path,
    num_files: usize,
) -> io::Result<Manifest> {
    assert!(num_files >= 1);
    std::fs::create_dir_all(out_dir)?;
    let hidden = cfg.hidden as usize;
    let moe_inter = cfg.moe_inter as usize;

    let mut experts: Vec<(usize, usize)> = Vec::new();
    for layer in cfg.first_dense as usize..cfg.n_layers as usize {
        for eid in 0..cfg.n_experts as usize {
            experts.push((layer, eid));
        }
    }

    let mut files: Vec<File> = (0..num_files)
        .map(|f| File::create(Manifest::shard_path(out_dir, f)))
        .collect::<io::Result<_>>()?;
    let mut offsets = vec![0u64; num_files];
    let mut locs = Vec::with_capacity(experts.len());
    let mut layer_meta: HashMap<usize, (i32, u64)> = HashMap::new();

    for (idx, &(layer, eid)) in experts.iter().enumerate() {
        let ex = provider.expert(layer, eid)?;
        let fmt = ex.gate.fmt_code;
        let blob = expert_blob(&ex);
        debug_assert_eq!(blob.len(), blob_bytes(hidden, moe_inter, fmt));
        layer_meta.entry(layer).or_insert((fmt, blob.len() as u64));
        let f = idx % num_files; // round-robin → byte-balanced shards
        files[f].write_all(&blob)?;
        locs.push(ExpertLoc {
            layer,
            eid,
            file: f,
            offset: offsets[f],
        });
        offsets[f] += blob.len() as u64;
    }
    for f in &mut files {
        f.flush()?;
    }

    let mut layers: Vec<(usize, i32, u64)> =
        layer_meta.into_iter().map(|(l, (f, b))| (l, f, b)).collect();
    layers.sort();
    let manifest = Manifest {
        num_files,
        hidden,
        moe_inter,
        layers,
        experts: locs,
    };
    manifest.save(out_dir.join("manifest.json"))?;
    Ok(manifest)
}

/// Preload experts **directly from the original safetensors** in parallel — no
/// repack, no second copy on disk. The `Shards` index already gives every
/// tensor's `(file, offset)`, so this is the loader you usually want.
///
/// Experts are sorted by their gate tensor's on-disk offset and split into
/// `num_threads` contiguous chunks, so each thread scans a contiguous region of
/// the model (near-sequential reads) while N threads together keep the NVMe
/// queue deep. Each thread loads until its share of `budget_bytes` is used.
pub fn preload_parallel(
    shards: &Shards,
    cfg: &Config,
    ebits: u32,
    num_threads: usize,
    budget_bytes: u64,
) -> io::Result<PreloadStore> {
    let hidden = cfg.hidden as usize;
    let moe_inter = cfg.moe_inter as usize;
    let nthreads = num_threads.max(1);

    // (layer, eid) sorted by on-disk offset of the gate tensor → sequential reads
    let mut experts: Vec<(usize, usize)> = Vec::new();
    for layer in cfg.first_dense as usize..cfg.n_layers as usize {
        for eid in 0..cfg.n_experts as usize {
            experts.push((layer, eid));
        }
    }
    experts.sort_by_key(|&(l, e)| {
        shards
            .find(&expert_gate_name(l, e))
            .map(|t| (t.file_idx, t.off))
            .unwrap_or((usize::MAX, u64::MAX))
    });

    let per_thread = experts.len().div_ceil(nthreads).max(1);
    let per_thread_budget = budget_bytes / nthreads as u64;

    let results: Vec<io::Result<HashMap<(usize, usize), Arc<Expert>>>> =
        std::thread::scope(|scope| {
            let handles: Vec<_> = experts
                .chunks(per_thread)
                .map(|chunk| {
                    scope.spawn(move || -> io::Result<HashMap<(usize, usize), Arc<Expert>>> {
                        let mut map = HashMap::new();
                        let mut used = 0u64;
                        for &(layer, eid) in chunk {
                            if used > per_thread_budget && !map.is_empty() {
                                break;
                            }
                            // Bulk preload is already ~20-way parallel across experts,
                            // so each expert's read stays a single stream (chunking it
                            // too would only oversubscribe the drive).
                            let mut ex = load_expert(shards, hidden, moe_inter, ebits, layer, eid, 1)?;
                            ex.mark_gpu_eligible(); // preloaded == resident/stable
                            used += ex.bytes();
                            map.insert((layer, eid), Arc::new(ex));
                        }
                        Ok(map)
                    })
                })
                .collect();
            handles.into_iter().map(|h| h.join().unwrap()).collect()
        });

    let mut all = HashMap::new();
    for r in results {
        all.extend(r?);
    }
    Ok(PreloadStore { experts: all })
}

/// Experts held resident in RAM. Serves inference with zero disk I/O per token.
/// Built either by [`preload_parallel`] (direct from the model) or
/// [`PreloadStore::load`] (from repacked shards).
pub struct PreloadStore {
    experts: HashMap<(usize, usize), Arc<Expert>>,
}

impl PreloadStore {
    /// Read the shards in parallel (one thread per shard) into RAM. Each thread
    /// loads its shard's experts in offset order (sequential scan) until
    /// `per_file_budget_bytes` is reached — so passing `total/num_files` loads
    /// "as many as fit", balanced across shards.
    pub fn load(
        manifest: &Manifest,
        dir: &Path,
        per_file_budget_bytes: u64,
    ) -> io::Result<PreloadStore> {
        let mut by_file: Vec<Vec<ExpertLoc>> = vec![Vec::new(); manifest.num_files];
        for e in &manifest.experts {
            by_file[e.file].push(*e);
        }
        for v in &mut by_file {
            v.sort_by_key(|e| e.offset);
        }
        let layer_meta = manifest.layer_meta();
        let hidden = manifest.hidden;
        let moe_inter = manifest.moe_inter;

        let results: Vec<io::Result<HashMap<(usize, usize), Arc<Expert>>>> =
            std::thread::scope(|scope| {
                let handles: Vec<_> = by_file
                    .iter()
                    .enumerate()
                    .map(|(fi, locs)| {
                        let layer_meta = &layer_meta;
                        scope.spawn(move || -> io::Result<HashMap<(usize, usize), Arc<Expert>>> {
                            let file = File::open(Manifest::shard_path(dir, fi))?;
                            let mut map = HashMap::new();
                            let mut buf: Vec<u8> = Vec::new();
                            let mut used = 0u64;
                            for loc in locs {
                                let (fmt, ebytes) = layer_meta[&loc.layer];
                                if used + ebytes > per_file_budget_bytes && !map.is_empty() {
                                    break; // budget reached for this shard
                                }
                                buf.resize(ebytes as usize, 0);
                                read_exact_at(&file, loc.offset, &mut buf)?;
                                let mut ex = expert_from_blob(&buf, hidden, moe_inter, fmt);
                                ex.mark_gpu_eligible(); // preloaded == resident/stable
                                map.insert((loc.layer, loc.eid), Arc::new(ex));
                                used += ebytes;
                            }
                            Ok(map)
                        })
                    })
                    .collect();
                handles.into_iter().map(|h| h.join().unwrap()).collect()
            });

        let mut experts = HashMap::new();
        for r in results {
            experts.extend(r?);
        }
        Ok(PreloadStore { experts })
    }

    pub fn len(&self) -> usize {
        self.experts.len()
    }

    pub fn is_empty(&self) -> bool {
        self.experts.is_empty()
    }
}

impl ExpertProvider for PreloadStore {
    fn expert(&self, layer: usize, eid: usize) -> io::Result<Arc<Expert>> {
        self.experts.get(&(layer, eid)).cloned().ok_or_else(|| {
            io::Error::new(
                io::ErrorKind::NotFound,
                format!("expert ({layer},{eid}) not preloaded"),
            )
        })
    }
}

#[cfg(unix)]
fn read_exact_at(f: &File, off: u64, buf: &mut [u8]) -> io::Result<()> {
    use std::os::unix::fs::FileExt;
    f.read_exact_at(buf, off)
}

#[cfg(not(unix))]
fn read_exact_at(f: &File, off: u64, buf: &mut [u8]) -> io::Result<()> {
    use std::io::{Read, Seek, SeekFrom};
    let mut f = f.try_clone()?;
    f.seek(SeekFrom::Start(off))?;
    f.read_exact(buf)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::quantize::qtensor_from_f32;

    fn tiny_expert(seed: usize) -> Expert {
        let mk = |o: usize, i: usize, s: usize| {
            let w: Vec<f32> = (0..o * i).map(|k| (((k + s) % 9) as f32 - 4.0) * 0.1).collect();
            qtensor_from_f32(&w, o, i, 4) // int4
        };
        Expert {
            gate: mk(4, 8, seed),
            up: mk(4, 8, seed + 1),
            down: mk(8, 4, seed + 2),
        }
    }

    #[test]
    fn blob_roundtrip_is_byte_identical() {
        let ex = tiny_expert(3);
        let blob = expert_blob(&ex);
        assert_eq!(blob.len(), blob_bytes(8, 4, 2));
        let back = expert_from_blob(&blob, 8, 4, 2);
        assert_eq!(back.gate.q4, ex.gate.q4);
        assert_eq!(back.gate.s, ex.gate.s);
        assert_eq!(back.up.q4, ex.up.q4);
        assert_eq!(back.down.q4, ex.down.q4);
        assert_eq!(back.down.s, ex.down.s);
        assert_eq!((back.gate.o, back.gate.i), (4, 8));
        assert_eq!((back.down.o, back.down.i), (8, 4));
    }

    #[test]
    fn manifest_save_load_roundtrip() {
        let m = Manifest {
            num_files: 3,
            hidden: 8,
            moe_inter: 4,
            layers: vec![(1, 2, 128)],
            experts: vec![
                ExpertLoc { layer: 1, eid: 0, file: 0, offset: 0 },
                ExpertLoc { layer: 1, eid: 1, file: 1, offset: 128 },
            ],
        };
        let dir = std::env::var("TMPDIR").unwrap_or_else(|_| "/tmp".into());
        let path = std::path::Path::new(&dir).join(format!("coli-manifest-{}.json", std::process::id()));
        m.save(&path).unwrap();
        let m2 = Manifest::load(&path).unwrap();
        assert_eq!(m2.num_files, 3);
        assert_eq!(m2.layers, vec![(1, 2, 128)]);
        assert_eq!(m2.experts.len(), 2);
        assert_eq!((m2.experts[1].file, m2.experts[1].offset), (1, 128));
        std::fs::remove_file(&path).ok();
    }
}
