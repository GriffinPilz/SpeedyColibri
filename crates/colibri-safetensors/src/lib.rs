//! On-demand tensor indexing and reads across multiple safetensors shards.
//!
//! Port of `c/st.h`. Equivalent to `Shards` in the reference `engine.py`, but:
//!   - reads with positioned reads (`pread`) instead of mmap, so pages do not
//!     stay resident in the process (the RSS fix — peak RAM stays dense+cache
//!     rather than the whole model);
//!   - always converts to `f32` on the float path (BF16/F16/F32), and reads the
//!     quantized container tensors (`U8`) raw.
//!
//! Fidelity note: the C version also calls `posix_fadvise(DONTNEED)` after
//! streaming-expert reads and keeps an `O_DIRECT` twin fd to bypass the page
//! cache. Those are *performance* behaviors (they bound RSS and lift cold
//! throughput on VHDX-backed ext4); this port omits them for now and reads
//! buffered. Correctness is unaffected. See PORTING.md — tracked as a follow-up
//! to reintroduce via `libc::posix_fadvise` behind a `cfg(unix)` gate.

use colibri_core::dtype::{bf16_to_f32, f16_to_f32, f8e4m3_to_f32, f8e5m2_to_f32, DType};
use colibri_core::SharedBuf;
use colibri_json::Json;
use std::collections::HashMap;
use std::fs::File;
use std::io::{self, Read, Seek, SeekFrom, Write};
use std::path::{Path, PathBuf};
use std::sync::{Arc, Mutex};

/// One tensor located within a shard file.
#[derive(Debug, Clone)]
pub struct StTensor {
    pub name: String,
    /// index into `Shards::files`
    pub file_idx: usize,
    /// absolute byte offset of the data within the file
    pub off: u64,
    pub nbytes: u64,
    pub dtype: DType,
    pub numel: i64,
    /// full tensor shape from the safetensors header (needed by the converter to
    /// recover `[O, I]` and the `[⌈O/128⌉, ⌈I/128⌉]` FP8 block-scale grid)
    pub shape: Vec<i64>,
}

/// A set of indexed safetensors shards, supporting on-demand reads by name.
pub struct Shards {
    tensors: Vec<StTensor>,
    files: Vec<(PathBuf, File)>,
    index: HashMap<String, usize>,
}

impl Shards {
    /// Index every `*.safetensors` file in `snap_dir`, in sorted filename order
    /// (so `model-00001-of-...` precedes `model-00002-...`). Port of `st_init`.
    pub fn open(snap_dir: impl AsRef<Path>) -> io::Result<Shards> {
        let dir = snap_dir.as_ref();
        let mut paths: Vec<PathBuf> = Vec::new();
        for entry in std::fs::read_dir(dir)? {
            let entry = entry?;
            let p = entry.path();
            if p.extension().map(|e| e == "safetensors").unwrap_or(false) {
                paths.push(p);
            }
        }
        paths.sort();

        let mut s = Shards {
            tensors: Vec::new(),
            files: Vec::new(),
            index: HashMap::new(),
        };

        for path in paths {
            let mut file = File::open(&path)?;
            let file_idx = s.files.len();

            // 8-byte little-endian header length, then the JSON header.
            let mut len_buf = [0u8; 8];
            file.seek(SeekFrom::Start(0))?;
            file.read_exact(&mut len_buf)?;
            let hlen = u64::from_le_bytes(len_buf);
            let mut hdr = vec![0u8; hlen as usize];
            file.read_exact(&mut hdr)?;
            let data_start = 8 + hlen;

            let hdr_str = String::from_utf8_lossy(&hdr);
            let root = Json::parse(&hdr_str).ok_or_else(|| {
                io::Error::new(
                    io::ErrorKind::InvalidData,
                    format!("{}: unparseable safetensors header", path.display()),
                )
            })?;
            let obj = root.as_object().ok_or_else(|| {
                io::Error::new(io::ErrorKind::InvalidData, "safetensors header not an object")
            })?;

            for (name, meta) in obj.iter() {
                if name == "__metadata__" {
                    continue;
                }
                let dtype_str = meta
                    .get("dtype")
                    .and_then(Json::as_str)
                    .ok_or_else(|| bad(&path, name, "dtype"))?;
                let dtype = DType::parse(dtype_str).ok_or_else(|| {
                    io::Error::new(
                        io::ErrorKind::InvalidData,
                        format!("unsupported dtype: {dtype_str}"),
                    )
                })?;
                let offsets = meta
                    .get("data_offsets")
                    .and_then(Json::as_array)
                    .ok_or_else(|| bad(&path, name, "data_offsets"))?;
                let shape = meta
                    .get("shape")
                    .and_then(Json::as_array)
                    .ok_or_else(|| bad(&path, name, "shape"))?;
                let a0 = offsets.first().and_then(Json::as_i64).unwrap_or(0);
                let b0 = offsets.get(1).and_then(Json::as_i64).unwrap_or(0);
                let dims: Vec<i64> = shape.iter().map(|d| d.as_i64().unwrap_or(0)).collect();
                let numel: i64 = dims.iter().product();

                let idx = s.tensors.len();
                s.tensors.push(StTensor {
                    name: name.to_string(),
                    file_idx,
                    off: data_start + a0 as u64,
                    nbytes: (b0 - a0) as u64,
                    dtype,
                    numel,
                    shape: dims,
                });
                s.index.insert(name.to_string(), idx);
            }

            s.files.push((path, file));
        }

        Ok(s)
    }

    /// Number of indexed tensors.
    pub fn len(&self) -> usize {
        self.tensors.len()
    }

    pub fn is_empty(&self) -> bool {
        self.tensors.is_empty()
    }

    /// Look up a tensor by name.
    pub fn find(&self, name: &str) -> Option<&StTensor> {
        self.index.get(name).map(|&i| &self.tensors[i])
    }

    /// All indexed tensors, in file/offset discovery order. The `file_idx` field
    /// groups them by shard — used by the FP8→int4 converter to stream one input
    /// shard at a time.
    pub fn tensors(&self) -> &[StTensor] {
        &self.tensors
    }

    /// Number of shard files indexed.
    pub fn num_files(&self) -> usize {
        self.files.len()
    }

    /// Whether a tensor exists — port of `st_has`.
    pub fn has(&self, name: &str) -> bool {
        self.index.contains_key(name)
    }

    /// Element count of a tensor, or `-1` if absent (`st_numel`).
    pub fn numel(&self, name: &str) -> i64 {
        self.find(name).map(|t| t.numel).unwrap_or(-1)
    }

    /// Byte count of a tensor, or `-1` if absent (`st_nbytes`).
    pub fn nbytes(&self, name: &str) -> i64 {
        self.find(name).map(|t| t.nbytes as i64).unwrap_or(-1)
    }

    fn tensor(&self, name: &str) -> io::Result<&StTensor> {
        self.find(name).ok_or_else(|| {
            io::Error::new(io::ErrorKind::NotFound, format!("missing tensor: {name}"))
        })
    }

    /// Read a tensor into a caller-provided `f32` slice (`numel` floats),
    /// converting from BF16/F16/F32. Port of `st_read_f32`.
    pub fn read_f32(&self, name: &str, out: &mut [f32]) -> io::Result<i64> {
        let t = self.tensor(name)?;
        let raw = self.pread(t.file_idx, t.off, t.nbytes as usize)?;
        convert_to_f32(t.dtype, &raw, out);
        Ok(t.numel)
    }

    /// Read the raw bytes of a tensor with no dtype conversion — for the already
    /// int4/int8-quantized container weights (`U8`). Port of `st_read_raw`.
    pub fn read_raw(&self, name: &str, out: &mut [u8]) -> io::Result<()> {
        let t = self.tensor(name)?;
        let n = t.nbytes as usize;
        self.pread_into(t.file_idx, t.off, &mut out[..n])
    }

    /// Write a subset of this snapshot's tensors (by `names`) into a fresh
    /// safetensors snapshot under `out_dir`, split across `out-NNNNN.safetensors`
    /// files each up to ~`max_file_bytes`. Bytes are copied **verbatim** — no dtype
    /// conversion — so a quantized / e4m3 container round-trips exactly through the
    /// loader. A name absent from this snapshot is an error. Returns the file count.
    /// Backs `coli shard-export`: writing one node's resident weights + owned experts.
    pub fn write_subset(
        &self,
        names: &[String],
        out_dir: &Path,
        max_file_bytes: u64,
    ) -> io::Result<usize> {
        std::fs::create_dir_all(out_dir)?;
        // Resolve up front so a missing name fails before any file is written.
        let mut items: Vec<&StTensor> = Vec::with_capacity(names.len());
        for n in names {
            items.push(self.find(n).ok_or_else(|| {
                io::Error::new(io::ErrorKind::NotFound, format!("shard-export: tensor not found: {n}"))
            })?);
        }
        let mut buf: Vec<u8> = Vec::new();
        let mut file_no = 0usize;
        let mut i = 0usize;
        while i < items.len() {
            // Greedily pack tensors into this output file up to the size cap (always
            // at least one, so a single oversized tensor still gets its own file).
            let start = i;
            let mut acc = 0u64;
            while i < items.len() && (i == start || acc + items[i].nbytes <= max_file_bytes) {
                acc += items[i].nbytes;
                i += 1;
            }
            let group = &items[start..i];
            // Build the JSON header; data_offsets are relative to the data segment.
            let mut header = String::from("{");
            let mut rel = 0u64;
            for (gi, t) in group.iter().enumerate() {
                if gi > 0 {
                    header.push(',');
                }
                let shape: Vec<String> = t.shape.iter().map(|d| d.to_string()).collect();
                header.push_str(&format!(
                    "\"{}\":{{\"dtype\":\"{}\",\"shape\":[{}],\"data_offsets\":[{},{}]}}",
                    t.name,
                    t.dtype.safetensors_str(),
                    shape.join(","),
                    rel,
                    rel + t.nbytes
                ));
                rel += t.nbytes;
            }
            header.push('}');
            let path = out_dir.join(format!("out-{file_no:05}.safetensors"));
            let mut f = io::BufWriter::new(File::create(&path)?);
            f.write_all(&(header.len() as u64).to_le_bytes())?;
            f.write_all(header.as_bytes())?;
            for t in group {
                let n = t.nbytes as usize;
                if buf.len() < n {
                    buf.resize(n, 0);
                }
                self.read_raw(&t.name, &mut buf[..n])?;
                f.write_all(&buf[..n])?;
            }
            f.flush()?;
            file_no += 1;
        }
        Ok(file_no)
    }

    /// Read several raw (U8) tensors, coalescing any that are **contiguous in the
    /// same file** into a single positioned read backed by a shared
    /// `Arc<SharedBuf>`. Returns, in the input order, `(buf, offset, len)` per
    /// name — a view into the shared allocation. One read (and one allocation) for
    /// a contiguous group; non-contiguous names fall back to their own reads. This
    /// is what lets an expert's gate/up/down (18 MB, contiguous) load in one shot
    /// instead of three. The buffer comes from the [`SharedBuf`] recycle pool and
    /// is wrapped with `Arc::new` (header-only move): no fresh mmap, no zero-fill
    /// page faults, and no payload copy in steady state — the naive
    /// `Arc::<[u8]>::from(Box<[u8]>)` alternative re-allocates and memcpys the
    /// 18 MB payload, which (with the fault churn) made warm expert loads 8×
    /// slower than the underlying read.
    pub fn read_raw_shared(
        &self,
        names: &[&str],
        nthreads: usize,
    ) -> io::Result<Vec<(Arc<SharedBuf>, usize, usize)>> {
        let n = names.len();
        let mut meta = Vec::with_capacity(n); // (file_idx, off, nbytes)
        for &nm in names {
            let t = self.tensor(nm)?;
            meta.push((t.file_idx, t.off, t.nbytes));
        }
        let mut order: Vec<usize> = (0..n).collect();
        order.sort_by_key(|&i| (meta[i].0, meta[i].1));

        let mut result: Vec<Option<(Arc<SharedBuf>, usize, usize)>> = (0..n).map(|_| None).collect();
        let mut g = 0;
        while g < n {
            let (file, off0, nb0) = meta[order[g]];
            let mut end = g + 1;
            let mut span_end = off0 + nb0;
            while end < n {
                let (f, o, nb) = meta[order[end]];
                if f == file && o == span_end {
                    span_end = o + nb;
                    end += 1;
                } else {
                    break;
                }
            }
            let span = (span_end - off0) as usize;
            // Pool-recycled buffer; on error it returns to the pool unread.
            let mut buf = SharedBuf::with_len(span);
            self.pread_chunked(file, off0, buf.as_mut_slice(), nthreads)?;
            let arc = Arc::new(buf);
            for gi in g..end {
                let idx = order[gi];
                let (_, o, nb) = meta[idx];
                result[idx] = Some((arc.clone(), (o - off0) as usize, nb as usize));
            }
            g = end;
        }
        // Every index is assigned exactly once (each name lands in some group).
        Ok(result.into_iter().map(|x| x.unwrap()).collect())
    }

    /// Batched analog of [`read_raw_shared`]: read several independent groups of
    /// contiguous tensors, pooling **all** groups' sub-chunk reads through one set
    /// of worker threads that drain a shared queue. Calling `read_raw_shared` in a
    /// loop spawns/joins a fresh thread scope per group and idles the drive at each
    /// barrier; here the NVMe streams continuously across every group. Measured
    /// ~6.85 vs ~5.5 GB/s (spawn-per-group) on the GB10's PCIe-4-x4 NVMe — the
    /// per-group join barrier was the entire gap.
    ///
    /// Returns, per group in input order, the per-name `(buf, off, len)` views —
    /// identical in shape to `read_raw_shared`.
    pub fn read_raw_shared_batched(
        &self,
        groups: &[&[&str]],
        nthreads: usize,
    ) -> io::Result<Vec<Vec<(Arc<SharedBuf>, usize, usize)>>> {
        // 1. Per group, coalesce contiguous names into spans (same rule as
        //    read_raw_shared), allocate a SharedBuf per span, and record how each
        //    name maps back to (span index, offset-in-span, len).
        struct Span {
            file: usize,
            off0: u64,
            buf: SharedBuf,
        }
        let mut spans: Vec<Span> = Vec::new();
        let mut mapping: Vec<Vec<(usize, usize, usize)>> = Vec::with_capacity(groups.len());
        for grp in groups {
            let n = grp.len();
            let mut meta = Vec::with_capacity(n);
            for &nm in grp.iter() {
                let t = self.tensor(nm)?;
                meta.push((t.file_idx, t.off, t.nbytes));
            }
            let mut order: Vec<usize> = (0..n).collect();
            order.sort_by_key(|&i| (meta[i].0, meta[i].1));
            let mut names_map: Vec<(usize, usize, usize)> = vec![(0, 0, 0); n];
            let mut g = 0;
            while g < n {
                let (file, off0, nb0) = meta[order[g]];
                let mut end = g + 1;
                let mut span_end = off0 + nb0;
                while end < n {
                    let (f, o, nb) = meta[order[end]];
                    if f == file && o == span_end {
                        span_end = o + nb;
                        end += 1;
                    } else {
                        break;
                    }
                }
                let span_len = (span_end - off0) as usize;
                let span_idx = spans.len();
                spans.push(Span { file, off0, buf: SharedBuf::with_len(span_len) });
                for gi in g..end {
                    let idx = order[gi];
                    let (_, o, nb) = meta[idx];
                    names_map[idx] = (span_idx, (o - off0) as usize, nb as usize);
                }
                g = end;
            }
            mapping.push(names_map);
        }

        // 2. Tile every span into fixed sub-chunks and drain them all through one
        //    pool of `nthreads` workers pulling via an atomic cursor — the drive
        //    stays saturated with no per-span barrier.
        const SUB: usize = 2 << 20; // 2 MiB — grid-optimal in-flight granularity
        struct Job {
            file: usize,
            off: u64,
            ptr: usize,
            len: usize,
        }
        let mut jobs: Vec<Job> = Vec::new();
        for s in spans.iter_mut() {
            let (file, off0) = (s.file, s.off0);
            let sl = s.buf.as_mut_slice();
            let base = sl.as_mut_ptr() as usize;
            let total = sl.len();
            let mut o = 0usize;
            while o < total {
                let clen = SUB.min(total - o);
                jobs.push(Job { file, off: off0 + o as u64, ptr: base + o, len: clen });
                o += clen;
            }
        }
        if !jobs.is_empty() {
            use std::sync::atomic::{AtomicUsize, Ordering};
            let nt = nthreads.max(1).min(jobs.len());
            let cursor = AtomicUsize::new(0);
            let err: Mutex<Option<io::Error>> = Mutex::new(None);
            let (jobs_ref, cursor_ref, err_ref) = (&jobs, &cursor, &err);
            std::thread::scope(|scope| {
                for _ in 0..nt {
                    scope.spawn(move || loop {
                        let i = cursor_ref.fetch_add(1, Ordering::Relaxed);
                        if i >= jobs_ref.len() {
                            break;
                        }
                        let j = &jobs_ref[i];
                        // SAFETY: each job addresses a disjoint sub-range of a
                        // distinct span buffer that outlives the scope; the ranges
                        // tile each buffer without overlap, so no two workers alias.
                        let dst =
                            unsafe { std::slice::from_raw_parts_mut(j.ptr as *mut u8, j.len) };
                        if let Err(e) = self.pread_into(j.file, j.off, dst) {
                            *err_ref.lock().unwrap() = Some(e);
                        }
                    });
                }
            });
            if let Some(e) = err.into_inner().unwrap() {
                return Err(e);
            }
        }

        // 3. Arc-wrap each span, then rebuild the per-group name views.
        let arcs: Vec<Arc<SharedBuf>> = spans.into_iter().map(|s| Arc::new(s.buf)).collect();
        Ok(mapping
            .into_iter()
            .map(|names_map| {
                names_map
                    .into_iter()
                    .map(|(si, off, len)| (arcs[si].clone(), off, len))
                    .collect()
            })
            .collect())
    }

    /// Read a slice of a tensor: `n_elems` starting at element `elem_off`. Used
    /// for GLM's fused experts (one tensor is a `[E, ...]` block; read only the
    /// requested expert's sub-range). Port of `st_read_slice_f32`.
    pub fn read_slice_f32(
        &self,
        name: &str,
        elem_off: i64,
        n_elems: i64,
        out: &mut [f32],
    ) -> io::Result<()> {
        let t = self.tensor(name)?;
        let esz = t.dtype.elem_size() as u64;
        let boff = t.off + elem_off as u64 * esz;
        let nb = n_elems as u64 * esz;
        let raw = self.pread(t.file_idx, boff, nb as usize)?;
        convert_to_f32(t.dtype, &raw, &mut out[..n_elems as usize]);
        Ok(())
    }

    /// Async readahead hint. In the C engine this is `posix_fadvise(WILLNEED)`;
    /// here it is a no-op placeholder (see the fidelity note at the top).
    pub fn prefetch(&self, _name: &str) {}

    // ---- positioned reads --------------------------------------------------

    fn pread(&self, file_idx: usize, off: u64, len: usize) -> io::Result<Vec<u8>> {
        let mut buf = vec![0u8; len];
        self.pread_into(file_idx, off, &mut buf)?;
        Ok(buf)
    }

    #[cfg(unix)]
    fn pread_into(&self, file_idx: usize, off: u64, buf: &mut [u8]) -> io::Result<()> {
        use std::os::unix::fs::FileExt;
        self.files[file_idx].1.read_exact_at(buf, off)
    }

    #[cfg(not(unix))]
    fn pread_into(&self, file_idx: usize, off: u64, buf: &mut [u8]) -> io::Result<()> {
        let mut f = self.files[file_idx].1.try_clone()?;
        f.seek(SeekFrom::Start(off))?;
        f.read_exact(buf)
    }

    /// Fill `buf` from `file` at `off`, splitting the read across up to `nthreads`
    /// positioned reads of disjoint sub-ranges. A single synchronous stream tops out
    /// far below the NVMe's bandwidth (it needs queue depth ~10); chunking one
    /// expert's 18 MB read across threads lets even a single cache miss saturate the
    /// drive, instead of feeding it 1–2 outstanding requests. Positioned reads
    /// (`pread`/`read_exact_at`) don't touch a shared file cursor, so this is safe.
    fn pread_chunked(
        &self,
        file: usize,
        off: u64,
        buf: &mut [u8],
        nthreads: usize,
    ) -> io::Result<()> {
        const MIN_CHUNK: usize = 1 << 20; // 1 MiB floor — smaller reads lose throughput
        let len = buf.len();
        let nt = nthreads.min(len / MIN_CHUNK).max(1);
        if nt <= 1 {
            return self.pread_into(file, off, buf);
        }
        let per = len.div_ceil(nt);
        let base = buf.as_mut_ptr() as usize;
        let err: Mutex<Option<io::Error>> = Mutex::new(None);
        std::thread::scope(|scope| {
            let mut start = 0;
            while start < len {
                let clen = per.min(len - start);
                let err = &err;
                scope.spawn(move || {
                    // SAFETY: each thread writes a disjoint [start, start+clen) sub-range
                    // of a buffer valid for `len` bytes (checked: start+clen <= len); the
                    // ranges never overlap and the buffer outlives the scope.
                    let dst = unsafe {
                        std::slice::from_raw_parts_mut((base + start) as *mut u8, clen)
                    };
                    if let Err(e) = self.pread_into(file, off + start as u64, dst) {
                        *err.lock().unwrap() = Some(e);
                    }
                });
                start += clen;
            }
        });
        match err.into_inner().unwrap() {
            Some(e) => Err(e),
            None => Ok(()),
        }
    }
}

fn bad(path: &Path, name: &str, field: &str) -> io::Error {
    io::Error::new(
        io::ErrorKind::InvalidData,
        format!("{}: tensor {name} missing {field}", path.display()),
    )
}

/// Convert `raw` bytes of the given dtype into `out[..numel]` as f32.
fn convert_to_f32(dtype: DType, raw: &[u8], out: &mut [f32]) {
    match dtype {
        DType::F32 => {
            for (o, chunk) in out.iter_mut().zip(raw.chunks_exact(4)) {
                *o = f32::from_le_bytes([chunk[0], chunk[1], chunk[2], chunk[3]]);
            }
        }
        DType::Bf16 => {
            for (o, chunk) in out.iter_mut().zip(raw.chunks_exact(2)) {
                *o = bf16_to_f32(u16::from_le_bytes([chunk[0], chunk[1]]));
            }
        }
        DType::F16 => {
            for (o, chunk) in out.iter_mut().zip(raw.chunks_exact(2)) {
                *o = f16_to_f32(u16::from_le_bytes([chunk[0], chunk[1]]));
            }
        }
        DType::U8 => {
            // No float interpretation for raw quantized data; caller should use
            // read_raw. Fall back to byte-as-float to avoid surprises.
            for (o, &b) in out.iter_mut().zip(raw.iter()) {
                *o = b as f32;
            }
        }
        DType::F8E4M3 => {
            for (o, &b) in out.iter_mut().zip(raw.iter()) {
                *o = f8e4m3_to_f32(b);
            }
        }
        DType::F8E5M2 => {
            for (o, &b) in out.iter_mut().zip(raw.iter()) {
                *o = f8e5m2_to_f32(b);
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::io::Write;

    /// Build a tiny one-shard safetensors file in a temp dir and index it.
    fn write_shard(dir: &Path) -> PathBuf {
        // Two tensors: an f32 [2,2] and a bf16 [2].
        // f32 payload: 1,2,3,4  (16 bytes)
        // bf16 payload: 1.0 (0x3F80), -1.0 (0xBF80)  (4 bytes)
        let header = r#"{"a":{"dtype":"F32","shape":[2,2],"data_offsets":[0,16]},"b":{"dtype":"BF16","shape":[2],"data_offsets":[16,20]}}"#;
        let hbytes = header.as_bytes();
        let path = dir.join("model.safetensors");
        let mut f = File::create(&path).unwrap();
        f.write_all(&(hbytes.len() as u64).to_le_bytes()).unwrap();
        f.write_all(hbytes).unwrap();
        for v in [1.0f32, 2.0, 3.0, 4.0] {
            f.write_all(&v.to_le_bytes()).unwrap();
        }
        f.write_all(&0x3F80u16.to_le_bytes()).unwrap();
        f.write_all(&0xBF80u16.to_le_bytes()).unwrap();
        path
    }

    fn temp_dir() -> PathBuf {
        let base =
            std::env::var("TMPDIR").unwrap_or_else(|_| "/tmp".to_string());
        let mut p = PathBuf::from(base);
        // unique-ish without external deps: pid + a counter file is overkill;
        // use pid + nanos-free monotonic via a static.
        use std::sync::atomic::{AtomicU64, Ordering};
        static N: AtomicU64 = AtomicU64::new(0);
        p.push(format!(
            "colibri-st-test-{}-{}",
            std::process::id(),
            N.fetch_add(1, Ordering::Relaxed)
        ));
        std::fs::create_dir_all(&p).unwrap();
        p
    }

    #[test]
    fn index_and_read() {
        let dir = temp_dir();
        write_shard(&dir);
        let s = Shards::open(&dir).unwrap();
        assert_eq!(s.len(), 2);
        assert!(s.has("a"));
        assert_eq!(s.numel("a"), 4);
        assert_eq!(s.nbytes("a"), 16);

        let mut out = vec![0f32; 4];
        s.read_f32("a", &mut out).unwrap();
        assert_eq!(out, vec![1.0, 2.0, 3.0, 4.0]);

        let mut bout = vec![0f32; 2];
        s.read_f32("b", &mut bout).unwrap();
        assert_eq!(bout, vec![1.0, -1.0]);

        std::fs::remove_dir_all(&dir).ok();
    }

    #[test]
    fn read_slice() {
        let dir = temp_dir();
        write_shard(&dir);
        let s = Shards::open(&dir).unwrap();
        // Read the middle two elements of "a" (3.0, 4.0 is elems 2..4; take 1..3).
        let mut out = vec![0f32; 2];
        s.read_slice_f32("a", 1, 2, &mut out).unwrap();
        assert_eq!(out, vec![2.0, 3.0]);
        std::fs::remove_dir_all(&dir).ok();
    }

    #[test]
    fn missing_tensor_errors() {
        let dir = temp_dir();
        write_shard(&dir);
        let s = Shards::open(&dir).unwrap();
        let mut out = vec![0f32; 1];
        let err = s.read_f32("nope", &mut out).unwrap_err();
        assert_eq!(err.kind(), io::ErrorKind::NotFound);
        std::fs::remove_dir_all(&dir).ok();
    }

    /// Write a shard of `U8` tensors at explicit `(name, offset, len)` positions in
    /// the data section, with `data` as the raw payload blob.
    fn write_u8_shard(dir: &Path, entries: &[(&str, usize, usize)], data: &[u8]) -> PathBuf {
        let parts: Vec<String> = entries
            .iter()
            .map(|(name, off, len)| {
                format!(
                    r#""{}":{{"dtype":"U8","shape":[{}],"data_offsets":[{},{}]}}"#,
                    name,
                    len,
                    off,
                    off + len
                )
            })
            .collect();
        let header = format!("{{{}}}", parts.join(","));
        let hbytes = header.as_bytes();
        let path = dir.join("model.safetensors");
        let mut f = File::create(&path).unwrap();
        f.write_all(&(hbytes.len() as u64).to_le_bytes()).unwrap();
        f.write_all(hbytes).unwrap();
        f.write_all(data).unwrap();
        path
    }

    fn view(v: &(Arc<SharedBuf>, usize, usize)) -> Vec<u8> {
        v.0[v.1..v.1 + v.2].to_vec()
    }

    #[test]
    fn read_raw_shared_contiguous_shares_one_buffer() {
        // gate|up|down contiguous on disk (like a real expert): one coalesced read
        // into a single shared buffer, each tensor a correctly-bounded view.
        let dir = temp_dir();
        let data: Vec<u8> = (0..12).collect();
        write_u8_shard(&dir, &[("g", 0, 4), ("u", 4, 4), ("d", 8, 4)], &data);
        let s = Shards::open(&dir).unwrap();
        let r = s.read_raw_shared(&["g", "u", "d"], 4).unwrap();
        // all three views are into the same Arc allocation
        assert!(Arc::ptr_eq(&r[0].0, &r[1].0));
        assert!(Arc::ptr_eq(&r[1].0, &r[2].0));
        // each view holds exactly its bytes, in range
        assert_eq!(view(&r[0]), vec![0, 1, 2, 3]);
        assert_eq!(view(&r[1]), vec![4, 5, 6, 7]);
        assert_eq!(view(&r[2]), vec![8, 9, 10, 11]);
        std::fs::remove_dir_all(&dir).ok();
    }

    #[test]
    fn read_raw_shared_non_contiguous_separate_reads() {
        // Gaps between tensors → separate reads (correctness preserved, no
        // coalescing). Each view must still hold exactly its own bytes and skip
        // the gap bytes entirely.
        let dir = temp_dir();
        let data: Vec<u8> = (0..20).collect();
        write_u8_shard(&dir, &[("g", 0, 4), ("u", 8, 4), ("d", 16, 4)], &data);
        let s = Shards::open(&dir).unwrap();
        let r = s.read_raw_shared(&["g", "u", "d"], 4).unwrap();
        assert!(!Arc::ptr_eq(&r[0].0, &r[1].0));
        assert!(!Arc::ptr_eq(&r[1].0, &r[2].0));
        assert_eq!(view(&r[0]), vec![0, 1, 2, 3]);
        assert_eq!(view(&r[1]), vec![8, 9, 10, 11]); // gap [4,8) skipped
        assert_eq!(view(&r[2]), vec![16, 17, 18, 19]); // gap [12,16) skipped
        std::fs::remove_dir_all(&dir).ok();
    }

    #[test]
    fn read_raw_shared_preserves_input_order() {
        // On disk (offset order) a|b|c contiguous, but queried as c,a,b. Each result
        // slot must map to its *input* name's bytes via its own computed offset —
        // this is the physical-vs-input-order case that guards against wrong ranges.
        let dir = temp_dir();
        let data: Vec<u8> = (0..12).collect();
        write_u8_shard(&dir, &[("a", 0, 4), ("b", 4, 4), ("c", 8, 4)], &data);
        let s = Shards::open(&dir).unwrap();
        let r = s.read_raw_shared(&["c", "a", "b"], 4).unwrap();
        assert_eq!(view(&r[0]), vec![8, 9, 10, 11]); // c
        assert_eq!(view(&r[1]), vec![0, 1, 2, 3]); // a
        assert_eq!(view(&r[2]), vec![4, 5, 6, 7]); // b
        // contiguous on disk → still one shared buffer despite the query order
        assert!(Arc::ptr_eq(&r[0].0, &r[1].0));
        std::fs::remove_dir_all(&dir).ok();
    }

    #[test]
    fn pread_chunked_multi_chunk_matches_content() {
        // A tensor several MiB larger than MIN_CHUNK (1 MiB) with a non-aligned tail,
        // so read_raw_shared actually splits it into multiple disjoint reads. Every
        // byte — including across chunk boundaries and the short final chunk — must
        // match the on-disk content.
        let dir = temp_dir();
        let n = 9 * (1 << 20) + 777; // 9 MiB + tail → 8 chunks at nthreads=8
        let data: Vec<u8> = (0..n).map(|i| (i % 251) as u8).collect();
        write_u8_shard(&dir, &[("w", 0, n)], &data);
        let s = Shards::open(&dir).unwrap();
        let r = s.read_raw_shared(&["w"], 8).unwrap();
        let got = view(&r[0]);
        assert_eq!(got.len(), n);
        assert_eq!(got, data);
        std::fs::remove_dir_all(&dir).ok();
    }

    #[test]
    fn read_raw_shared_batched_matches_looped() {
        // Three expert-like groups (each g|u|d contiguous), spans large enough to
        // tile into multiple 2 MiB sub-chunks, so the pooled cursor interleaves
        // reads across spans. Batched output must be byte-identical to calling
        // read_raw_shared per group.
        let dir = temp_dir();
        let w = 1usize << 20; // 1 MiB per weight → 3 MiB span → tiles past SUB (2 MiB)
        let ne = 3;
        let mut entries: Vec<(String, usize, usize)> = Vec::new();
        let mut off = 0;
        for e in 0..ne {
            for part in ["g", "u", "d"] {
                entries.push((format!("e{e}.{part}"), off, w));
                off += w;
            }
        }
        let data: Vec<u8> = (0..off).map(|i| (i % 251) as u8).collect();
        let eref: Vec<(&str, usize, usize)> =
            entries.iter().map(|(n, o, l)| (n.as_str(), *o, *l)).collect();
        write_u8_shard(&dir, &eref, &data);
        let s = Shards::open(&dir).unwrap();

        let names: Vec<[String; 3]> = (0..ne)
            .map(|e| [format!("e{e}.g"), format!("e{e}.u"), format!("e{e}.d")])
            .collect();
        let groups: Vec<[&str; 3]> =
            names.iter().map(|g| [g[0].as_str(), g[1].as_str(), g[2].as_str()]).collect();
        let group_refs: Vec<&[&str]> = groups.iter().map(|g| &g[..]).collect();

        let batched = s.read_raw_shared_batched(&group_refs, 8).unwrap();
        assert_eq!(batched.len(), ne);
        for (gi, grp) in group_refs.iter().enumerate() {
            let looped = s.read_raw_shared(grp, 8).unwrap();
            assert_eq!(batched[gi].len(), looped.len());
            for k in 0..looped.len() {
                assert_eq!(view(&batched[gi][k]), view(&looped[k]), "group {gi} name {k}");
            }
        }
        // Contiguous g|u|d within a group still share one buffer …
        assert!(Arc::ptr_eq(&batched[0][0].0, &batched[0][2].0));
        // … while different experts get distinct buffers.
        assert!(!Arc::ptr_eq(&batched[0][0].0, &batched[1][0].0));
        std::fs::remove_dir_all(&dir).ok();
    }
}
