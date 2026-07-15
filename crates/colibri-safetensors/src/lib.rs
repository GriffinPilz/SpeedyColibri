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

use colibri_core::dtype::{bf16_to_f32, f16_to_f32, DType};
use colibri_json::Json;
use std::collections::HashMap;
use std::fs::File;
use std::io::{self, Read, Seek, SeekFrom};
use std::path::{Path, PathBuf};
use std::sync::Arc;

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
                let numel: i64 = shape.iter().map(|d| d.as_i64().unwrap_or(0)).product();

                let idx = s.tensors.len();
                s.tensors.push(StTensor {
                    name: name.to_string(),
                    file_idx,
                    off: data_start + a0 as u64,
                    nbytes: (b0 - a0) as u64,
                    dtype,
                    numel,
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

    /// Read several raw (U8) tensors, coalescing any that are **contiguous in the
    /// same file** into a single positioned read backed by a shared `Arc<[u8]>`.
    /// Returns, in the input order, `(buf, offset, len)` per name — a view into the
    /// shared allocation. One read (and one allocation) for a contiguous group;
    /// non-contiguous names fall back to their own reads. This is what lets an
    /// expert's gate/up/down (18 MB, contiguous) load in one shot instead of three,
    /// avoiding the per-tensor read + allocation overhead that dominated streaming.
    pub fn read_raw_shared(&self, names: &[&str]) -> io::Result<Vec<(Arc<[u8]>, usize, usize)>> {
        let n = names.len();
        let mut meta = Vec::with_capacity(n); // (file_idx, off, nbytes)
        for &nm in names {
            let t = self.tensor(nm)?;
            meta.push((t.file_idx, t.off, t.nbytes));
        }
        let mut order: Vec<usize> = (0..n).collect();
        order.sort_by_key(|&i| (meta[i].0, meta[i].1));

        let mut result: Vec<Option<(Arc<[u8]>, usize, usize)>> = (0..n).map(|_| None).collect();
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
            // Read into an uninitialized buffer (pread_into fills it fully on Ok).
            let mut buf = Vec::<u8>::with_capacity(span);
            // SAFETY: pread_into writes all `span` bytes on success; on error we
            // drop `buf` without reading it.
            unsafe { buf.set_len(span) };
            self.pread_into(file, off0, &mut buf)?;
            let arc: Arc<[u8]> = Arc::from(buf.into_boxed_slice());
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
        let esz = if t.dtype == DType::F32 { 4 } else { 2 } as u64;
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

    fn view(v: &(Arc<[u8]>, usize, usize)) -> Vec<u8> {
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
        let r = s.read_raw_shared(&["g", "u", "d"]).unwrap();
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
        let r = s.read_raw_shared(&["g", "u", "d"]).unwrap();
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
        let r = s.read_raw_shared(&["c", "a", "b"]).unwrap();
        assert_eq!(view(&r[0]), vec![8, 9, 10, 11]); // c
        assert_eq!(view(&r[1]), vec![0, 1, 2, 3]); // a
        assert_eq!(view(&r[2]), vec![4, 5, 6, 7]); // b
        // contiguous on disk → still one shared buffer despite the query order
        assert!(Arc::ptr_eq(&r[0].0, &r[1].0));
        std::fs::remove_dir_all(&dir).ok();
    }
}
