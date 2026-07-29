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
//! cache. Both now exist here behind `COLI_FADVISE` (see [`fadvise_enabled`] —
//! measured a large regression, leave off) and `COLI_O_DIRECT` (see
//! [`o_direct_enabled`]). The `O_DIRECT` twin is not a drop-in fd swap: **no
//! tensor offset in our containers is 512-aligned** (measured: 0 of 4326
//! sampled), so reads are aligned at *span* granularity rather than bounced
//! through a scratch buffer, which would otherwise reintroduce a full memcpy of
//! every expert. Correctness is unaffected by either flag.

use colibri_core::dtype::{bf16_to_f32, f16_to_f32, f8e4m3_to_f32, f8e5m2_to_f32, DType};
use colibri_core::SharedBuf;
use colibri_json::Json;
use std::collections::HashMap;
use std::fs::File;
use std::io::{self, Read, Seek, SeekFrom, Write};
use std::path::{Path, PathBuf};
use std::sync::{Arc, Mutex};

/// Whether to drop streamed bytes from the page cache after reading them
/// (`COLI_FADVISE=1`). **Off by default, and measured to be a large regression —
/// do not turn it on.** Kept only so the result stays reproducible.
///
/// The hypothesis was that the ~58 GB of page cache holding expert bytes was dead
/// weight competing with the LFRU expert cache, so freeing it would let the cache
/// cap rise past `MemTotal/3`. **That was wrong.** Measured (1 node, prompt 512,
/// ngen 12, tokens byte-identical in both arms):
///
/// |                | ms/token | cache misses | buff/cache |
/// |----------------|----------|--------------|------------|
/// | off (control)  |   3388.7 |       26,021 |      43 GB |
/// | `COLI_FADVISE=1` | 5517.5 |       46,045 |      13 GB |
///
/// It freed the RAM exactly as intended and cost 63% of decode throughput (and 80%
/// of prefill). Misses rose 77%, which is the tell: **the page cache was serving a
/// large share of reads**. It is not competing with the expert cache, it is a
/// second and much larger cache tier — effective residency is ~98 GB
/// (40 GB LFRU + ~58 GB page cache), not 40 GB.
///
/// Both follow-ups have since been measured, and both confirmed it:
/// [`o_direct_enabled`] bypasses the same tier and is 18% slower at equal cache size
/// despite winning the raw-bandwidth microbenchmark; and raising the expert-cache budget hurts
/// for the same reason, since it moves RAM from a tier that caches at 4 KB page
/// granularity into one that caches whole 38.3 MB experts.
/// Runtime fadvise toggle. The adaptive max-residency path ([`ExpertCache::spawn_adaptive_budget`])
/// turns this on so the explicit cache is the single tier and `MemAvailable` reflects
/// true free RAM (no page-cache double-hold of the model). Distinct from the
/// `COLI_FADVISE=1` env opt-in; either enables it.
static FADVISE_RUNTIME: std::sync::atomic::AtomicBool = std::sync::atomic::AtomicBool::new(false);

/// Enable/disable `fadvise(DONTNEED)`-after-read at runtime (set during cache setup).
pub fn set_fadvise(on: bool) {
    FADVISE_RUNTIME.store(on, std::sync::atomic::Ordering::Relaxed);
}

fn fadvise_enabled() -> bool {
    static ENV_ON: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    let env = *ENV_ON.get_or_init(|| std::env::var("COLI_FADVISE").ok().as_deref() == Some("1"));
    env || FADVISE_RUNTIME.load(std::sync::atomic::Ordering::Relaxed)
}

/// Sub-chunk size, in bytes, that [`Shards::read_raw_shared`] tiles each span into.
/// `COLI_READ_SUB_KB` overrides it (KiB); default 2 MiB.
///
/// This sets read *concurrency*, not request size — see the call site. Rounded down to
/// a 512-multiple with a 512 B floor, because O_DIRECT requires every job's offset,
/// address and length to be 512-aligned and the tiling inherits that from this value.
/// A non-numeric or sub-512 setting falls back to the default rather than failing: this
/// is a perf knob, and a typo should not stop the process from loading a model.
fn read_sub_bytes() -> usize {
    static SUB: std::sync::OnceLock<usize> = std::sync::OnceLock::new();
    *SUB.get_or_init(|| parse_sub_kb(std::env::var("COLI_READ_SUB_KB").ok().as_deref()))
}

/// Pure half of [`read_sub_bytes`], split out so the alignment and fallback rules are
/// testable without touching process-global env.
fn parse_sub_kb(v: Option<&str>) -> usize {
    const DEFAULT: usize = 2 << 20;
    v.and_then(|s| s.trim().parse::<usize>().ok())
        .map(|kb| kb.saturating_mul(1024) & !511)
        .filter(|&b| b >= 512)
        .unwrap_or(DEFAULT)
}

/// Serve page-cache-resident expert spans from a shared mapping instead of copying them
/// into a heap buffer (`COLI_MMAP_EXPERTS=0` disables).
///
/// The warm path is the whole point: once a span is in the page cache, `pread` still
/// memcpy's it into a fresh buffer at ~12 GB/s, against ~146 GB/s of memory bandwidth on
/// this box — measured as MiniMax-M2.7 spending 2805 ms moving 33.6 GB while reading
/// **zero** bytes from the drive. A mapped view hands the same bytes to the GPU with no
/// copy at all, and a CUDA microbenchmark puts the GPU's read rate at 163 GB/s from a
/// file-backed mapping vs 171 GB/s from a heap buffer — a 4.5% penalty to skip the copy
/// entirely.
fn mmap_enabled() -> bool {
    static ON: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *ON.get_or_init(|| std::env::var("COLI_MMAP_EXPERTS").ok().as_deref() != Some("0"))
}

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

/// Alignment O_DIRECT requires of the file offset, length, and destination address.
/// The NVMe's logical block size (measured 512 on the DGX Spark).
const DIO_ALIGN: u64 = 512;

/// Bypass the page cache entirely for shard reads.
///
/// **Chosen automatically** from the model's expert-set coverage during cache wiring
/// (`O_DIRECT_MAX_COVERAGE_PCT` in `coli`), not by the operator. `COLI_O_DIRECT=0|1`
/// overrides in either direction, for pinning an arm during a measurement.
///
/// Distinct from [`fadvise_enabled`], and the difference is the whole point: `fadvise`
/// still routes every read *through* the page cache and discards it afterwards, so it
/// pays the landing-zone cost and gets no caching in return — measured a 63% loss.
/// `O_DIRECT` never touches that tier at all.
///
/// Whether skipping it helps depends on whether the page cache can hold a useful share
/// of THIS model's experts. Measured across four models (table in
/// `O_DIRECT_MAX_COVERAGE_PCT`), the crossover is monotonic in coverage: at 7% and 27%
/// O_DIRECT wins by 1.089× and 1.145×; at 47% and 86% buffered wins. The byte counters
/// show the mechanism directly — at 7% both arms read the same device bytes (the page
/// cache was serving ~nothing), while at 86% the buffered arm read *zero* device bytes.
///
/// The historical measurement that said "leave off" is still real, but it was one model
/// (GLM) on its older 735 GB e4m3 container — roughly double today's 379 GB, so a
/// different coverage point than the name suggests. It is the 47%/86% end of the same
/// curve, not a contradiction. That table also showed the memory ceiling is real rather
/// than an artifact of buffered I/O: at 70 GB the O_DIRECT arm still thrashed (+325,409
/// major faults) and at 90 GB it could not finish.
///
/// The `O_DIRECT` twin is not a drop-in fd swap: **no tensor offset in our containers is
/// 512-aligned** (measured: 0 of 4326 sampled), so reads are aligned at *span*
/// granularity rather than bounced through a scratch buffer, which would otherwise
/// reintroduce a full memcpy of every expert. Correctness is unaffected either way.
/// Runtime O_DIRECT selection, set from the model's expert-set coverage during cache
/// wiring (see `wire_adaptive_cache`). `COLI_O_DIRECT` still wins when set, in either
/// direction, so a measurement can pin an arm.
static O_DIRECT_RUNTIME: std::sync::atomic::AtomicBool = std::sync::atomic::AtomicBool::new(false);

/// Choose O_DIRECT for shard reads. Called once, before any expert is read, with the
/// decision derived from how much of the model's expert set RAM can actually hold.
pub fn set_o_direct(on: bool) {
    O_DIRECT_RUNTIME.store(on, std::sync::atomic::Ordering::Relaxed);
}

fn o_direct_enabled() -> bool {
    static ENV: std::sync::OnceLock<Option<bool>> = std::sync::OnceLock::new();
    let env = *ENV.get_or_init(|| match std::env::var("COLI_O_DIRECT").ok().as_deref() {
        Some("1") => Some(true),
        Some("0") => Some(false),
        _ => None,
    });
    env.unwrap_or_else(|| O_DIRECT_RUNTIME.load(std::sync::atomic::Ordering::Relaxed))
}

/// Open a second, `O_DIRECT` descriptor for `path`. `None` if the platform or
/// filesystem refuses it, in which case reads fall back to the buffered fd.
#[cfg(target_os = "linux")]
fn open_direct(path: &Path) -> Option<File> {
    use std::os::unix::ffi::OsStrExt;
    use std::os::unix::io::FromRawFd;
    let c = std::ffi::CString::new(path.as_os_str().as_bytes()).ok()?;
    // SAFETY: `c` is a valid NUL-terminated path and the flags are read-only.
    let fd = unsafe { libc::open(c.as_ptr(), libc::O_RDONLY | libc::O_DIRECT) };
    if fd < 0 {
        return None;
    }
    // SAFETY: `fd` is a fresh, owned descriptor that nothing else holds.
    Some(unsafe { File::from_raw_fd(fd) })
}

#[cfg(not(target_os = "linux"))]
fn open_direct(_path: &Path) -> Option<File> {
    None
}

/// A whole shard file mapped read-only, for the lifetime of the process.
///
/// **Mapped once, never per read.** A per-read `mmap`/`munmap` pair serialises every
/// reader thread on the process-wide `mmap_lock` and tears down PTEs on each unmap, so
/// the 40-thread read pool collapses onto one core in the kernel while the GPU starves.
/// (The same effect is why [`colibri_core::SharedBuf`] recycles its heap allocations:
/// a fresh `mmap` + zero-fill faults cost ~14 ms per expert, 8× the read itself.)
/// Mapping the whole file costs only address space — 1.4 TB of it is free on 64-bit,
/// and physical pages materialise only when touched.
struct Mapping {
    ptr: *mut libc::c_void,
    len: usize,
}

// SAFETY: the mapping is read-only (`PROT_READ`) and never mutated after creation, so
// sharing the pointer across threads is sound. It is unmapped only in `Drop`, by which
// point no `SharedBuf` view can still reference it (each holds an `Arc<Mapping>`).
unsafe impl Send for Mapping {}
unsafe impl Sync for Mapping {}

impl Mapping {
    #[cfg(target_os = "linux")]
    fn open(file: &File, len: u64) -> Option<Mapping> {
        use std::os::unix::io::AsRawFd;
        if len == 0 {
            return None;
        }
        // SAFETY: a fresh read-only mapping of a valid fd; failure is reported as MAP_FAILED.
        let ptr = unsafe {
            libc::mmap(
                std::ptr::null_mut(),
                len as usize,
                libc::PROT_READ,
                libc::MAP_SHARED,
                file.as_raw_fd(),
                0,
            )
        };
        if ptr == libc::MAP_FAILED {
            return None;
        }
        // The kernel's default readahead is tuned for sequential streaming; our access is
        // scattered expert spans, and we never want it faulting *ahead* of a span we only
        // reached because `mincore` said it was already resident.
        // SAFETY: `ptr`/`len` name the mapping just created.
        unsafe { libc::madvise(ptr, len as usize, libc::MADV_RANDOM) };
        Some(Mapping { ptr, len: len as usize })
    }

    #[cfg(not(target_os = "linux"))]
    fn open(_file: &File, _len: u64) -> Option<Mapping> {
        None
    }

    /// `true` when every page of `[off, off+len)` is already in the page cache.
    ///
    /// This gate is the whole difference between the mapped path being free and being a
    /// catastrophe: touching a non-resident page faults it in 4 KiB at a time, where the
    /// `pread` path fetches the same span in one request. An earlier attempt without the
    /// gate took 407,570 major faults and ran ~300× slower cold; `MADV_WILLNEED` did not
    /// rescue it, because advisory readahead loses to a fault storm already underway.
    #[cfg(target_os = "linux")]
    fn resident(&self, off: usize, len: usize) -> bool {
        if len == 0 || off.saturating_add(len) > self.len {
            return false;
        }
        let page = 4096usize;
        let start = off & !(page - 1);
        let end = off + len;
        let pages = (end - start).div_ceil(page);
        let mut vec = vec![0u8; pages];
        // SAFETY: `start` is page-aligned and inside the mapping; `vec` has one byte per page.
        let rc = unsafe {
            libc::mincore(
                (self.ptr as *mut u8).add(start) as *mut libc::c_void,
                end - start,
                vec.as_mut_ptr() as *mut _,
            )
        };
        rc == 0 && vec.iter().all(|b| b & 1 == 1)
    }

    #[cfg(not(target_os = "linux"))]
    fn resident(&self, _off: usize, _len: usize) -> bool {
        false
    }
}

impl Drop for Mapping {
    fn drop(&mut self) {
        // SAFETY: unmapping exactly what `open` mapped.
        unsafe { libc::munmap(self.ptr, self.len) };
    }
}

/// A set of indexed safetensors shards, supporting on-demand reads by name.
pub struct Shards {
    tensors: Vec<StTensor>,
    files: Vec<(PathBuf, File)>,
    /// Parallel to `files`: `O_DIRECT` twin descriptors, opened **lazily** on first
    /// dispatched read. Lazy because the choice is made from the model's expert-set
    /// coverage during cache wiring, which happens *after* `Shards::open` — sizing the
    /// expert set needs a real expert probed through these very shards. An inner `None`
    /// means the open was tried and refused (non-Linux, or a filesystem that rejects
    /// `O_DIRECT`) and reads fall back to the buffered fd.
    dio: Vec<std::sync::OnceLock<Option<File>>>,
    /// Parallel to `files`: the whole-file read-only mapping, opened lazily on first
    /// eligible read. `None` means mapping was tried and refused. Only consulted when
    /// [`mmap_enabled`]; see [`Mapping`] for why this is per-file and not per-read.
    maps: Vec<std::sync::OnceLock<Option<std::sync::Arc<Mapping>>>>,
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
            dio: Vec::new(),
            maps: Vec::new(),
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

            s.dio.push(std::sync::OnceLock::new());
            s.maps.push(std::sync::OnceLock::new());
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
    /// groups them by shard — used by the FP8/NVFP4 converter to stream one input
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
    /// quantized container weights (int8/int2 codes, nvfp4 nibbles, e4m3; `U8`).
    /// Port of `st_read_raw`.
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
            // Zero-copy when the span is already resident; otherwise a pool-recycled
            // buffer, which on error returns to the pool unread.
            let arc = match self.mapped_view(file, off0, span) {
                // SAFETY: the range was checked to lie inside the mapping and to be
                // fully resident; the `Arc<Mapping>` keeps it alive under the view.
                Some(map) => Arc::new(unsafe {
                    let ptr = (map.ptr as *const u8).add(off0 as usize);
                    SharedBuf::from_view(map, ptr, span)
                }),
                None => {
                    let mut buf = SharedBuf::with_len(span);
                    self.pread_chunked(file, off0, buf.as_mut_slice(), nthreads)?;
                    Arc::new(buf)
                }
            };
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
            /// File offset the read starts at — 512-aligned under `O_DIRECT`, else
            /// exactly the span's first tensor offset.
            read_off: u64,
            /// Bytes to read from `read_off`; a 512-multiple under `O_DIRECT`.
            read_len: usize,
            /// Offset within `buf` at which `read_off`'s byte lands: the allocation's
            /// alignment skew. Views additionally add the file-side padding.
            skew: usize,
            buf: SharedBuf,
        }
        let dio = o_direct_enabled();
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

                // Zero-copy when this span is already page-cache resident: hand out a
                // view into the shard's mapping rather than allocating a buffer and
                // memcpy'ing the same bytes into it. `read_len: 0` marks the span as
                // needing no I/O, and the job builder below skips views.
                if let Some(map) = self.mapped_view(file, off0, span_len) {
                    // SAFETY: `mapped_view` checked the range lies inside the mapping and
                    // that every page of it is resident, and the `Arc<Mapping>` moved into
                    // the buffer keeps it alive for as long as any view survives.
                    let buf = unsafe {
                        let ptr = (map.ptr as *const u8).add(off0 as usize);
                        SharedBuf::from_view(map, ptr, span_len)
                    };
                    spans.push(Span { file, read_off: off0, read_len: 0, skew: 0, buf });
                    for gi in g..end {
                        let idx = order[gi];
                        let (_, o, nb) = meta[idx];
                        names_map[idx] = (span_idx, (o - off0) as usize, nb as usize);
                    }
                    g = end;
                    continue;
                }

                // Under O_DIRECT the file offset, the length, and the destination
                // address must all be 512-aligned, and no tensor offset in our
                // containers is. Align the *span* rather than bouncing each read
                // through an aligned scratch buffer: start at the 512 boundary below
                // `off0`, round the length up, and over-allocate by one alignment unit
                // so a 512-aligned address is guaranteed to exist inside the buffer.
                // Views then sit at `skew + pad + (o - off0)`. Costs <1 KiB per span
                // and, crucially, no copy — a bounce buffer would reintroduce exactly
                // the per-expert memcpy removed in 29e27b2.
                let (pad, extra) = if dio {
                    ((off0 & (DIO_ALIGN - 1)) as usize, DIO_ALIGN as usize)
                } else {
                    (0, 0)
                };
                let read_off = off0 - pad as u64;
                let read_len = if dio {
                    (pad + span_len).next_multiple_of(DIO_ALIGN as usize)
                } else {
                    span_len
                };
                let mut buf = SharedBuf::with_len(read_len + extra);
                // Skew is derived from the *actual* allocation: `SharedBuf` recycles
                // pooled buffers, so alignment varies between calls and cannot be
                // assumed.
                let skew = if dio {
                    let p = buf.as_mut_slice().as_ptr() as usize;
                    p.wrapping_neg() & (DIO_ALIGN as usize - 1)
                } else {
                    0
                };
                let prefix = skew + pad;
                spans.push(Span { file, read_off, read_len, skew, buf });
                for gi in g..end {
                    let idx = order[gi];
                    let (_, o, nb) = meta[idx];
                    names_map[idx] = (span_idx, prefix + (o - off0) as usize, nb as usize);
                }
                g = end;
            }
            mapping.push(names_map);
        }

        // 2. Tile every span into fixed sub-chunks and drain them all through one
        //    pool of `nthreads` workers pulling via an atomic cursor — the drive
        //    stays saturated with no per-span barrier.
        //
        // The tile size is an I/O *concurrency* knob, not a request-size one: workers
        // are handed whole jobs, so `nt = min(nthreads, jobs.len())` and the tile size
        // decides how many of them a given span can occupy. A single m2.7 expert is
        // ~7.6 MB contiguous, which at 2 MiB tiles decomposes into just 4 jobs — so an
        // on-demand miss uses 4 of the 40 available read threads and the NVMe queue
        // sits near 6 when it wants >= 32. Smaller tiles do not shrink the actual
        // device requests: the block layer already splits these at ~112 KB.
        // See `read_sub_bytes` for the override.
        let sub = read_sub_bytes();
        struct Job {
            file: usize,
            off: u64,
            ptr: usize,
            len: usize,
        }
        let mut jobs: Vec<Job> = Vec::new();
        for s in spans.iter_mut() {
            // Mapped spans are already satisfied — and `as_mut_slice` would panic on a
            // read-only view.
            if s.buf.is_view() {
                continue;
            }
            let (file, read_off, total, skew) = (s.file, s.read_off, s.read_len, s.skew);
            // Tiling starts at the aligned address, not the allocation base. `sub` is a
            // 512-multiple (enforced by `read_sub_bytes`) and `total` was rounded up, so
            // every job — including the last — keeps its offset, address and length
            // 512-aligned under O_DIRECT.
            let base = s.buf.as_mut_slice().as_mut_ptr() as usize + skew;
            let mut o = 0usize;
            while o < total {
                let clen = sub.min(total - o);
                jobs.push(Job { file, off: read_off + o as u64, ptr: base + o, len: clen });
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
                        if let Err(e) = self.pread_dispatch(j.file, j.off, dst) {
                            *err_ref.lock().unwrap() = Some(e);
                        }
                    });
                }
            });
            if let Some(e) = err.into_inner().unwrap() {
                return Err(e);
            }
        }

        // 3. Release the page-cache copy of everything we just streamed. Done after
        //    the join so each span is advised once as a whole range rather than per
        //    2 MiB job, and only once the bytes are safely in our own buffer.
        if fadvise_enabled() && !dio {
            for s in spans.iter() {
                self.fadvise_dontneed(s.file, s.read_off, s.read_len);
            }
        }

        // 4. Arc-wrap each span, then rebuild the per-group name views.
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

    /// A zero-copy view of `[off, off+len)` in `file_idx`, or `None` when the mapped
    /// path is unavailable or the span is not already page-cache resident.
    ///
    /// Returning `None` is not a failure — it means "use the copy path", which is the
    /// right answer whenever the bytes would have to be faulted in one page at a time.
    fn mapped_view(&self, file_idx: usize, off: u64, len: usize) -> Option<Arc<Mapping>> {
        if !mmap_enabled() {
            return None;
        }
        let cell = self.maps.get(file_idx)?;
        let map = cell
            .get_or_init(|| {
                let (_, f) = self.files.get(file_idx)?;
                let size = f.metadata().ok()?.len();
                Mapping::open(f, size).map(Arc::new)
            })
            .clone()?;
        map.resident(off as usize, len).then_some(map)
    }

    /// Read into `buf`, preferring the `O_DIRECT` descriptor when one was opened.
    ///
    /// The caller is responsible for 512-alignment of `off`, `buf.as_ptr()` and
    /// `buf.len()`; [`Shards::read_raw_shared_batched`] arranges it by aligning whole
    /// spans. Falls back to the buffered fd wherever no direct descriptor exists —
    /// non-Linux, or a filesystem that refused `O_DIRECT` — so enabling the flag can
    /// never break a read, only fail to accelerate it.
    fn pread_dispatch(&self, file_idx: usize, off: u64, buf: &mut [u8]) -> io::Result<()> {
        #[cfg(target_os = "linux")]
        if o_direct_enabled() {
            // Opened on first use, then cached for the process's life. `get_or_init` is
            // idempotent under the read pool's concurrency: several threads may race here,
            // one open wins and the losers drop theirs.
            let slot = self.dio.get(file_idx).and_then(|cell| {
                cell.get_or_init(|| open_direct(&self.files[file_idx].0)).as_ref()
            });
            if let Some(f) = slot {
                use std::os::unix::fs::FileExt;
                return f.read_exact_at(buf, off);
            }
        }
        self.pread_into(file_idx, off, buf)
    }

    /// Drop `[off, off+len)` of `file_idx` from the page cache.
    ///
    /// Expert bytes are streamed once and never re-read *through the page cache* —
    /// the LFRU expert cache is the reuse tier, and measured decode reuse is ~0%
    /// anyway. Leaving them resident costs twice: the pages evict the expert cache
    /// they compete with for the same RAM, and the kernel burns time reclaiming a
    /// cache that never produces a hit. Measured on GB10: buffered reads at 8-way
    /// concurrency reach only 4.06 GB/s against 10.18 GB/s for `O_DIRECT` at the
    /// same depth, and this is the cheap half of closing that gap.
    ///
    /// Advisory and best-effort: the pages are clean (read-only), so the kernel can
    /// always honour it, but a failure is not actionable and is deliberately ignored.
    ///
    /// Linux-only: `posix_fadvise` is not in macOS's libc, and the deployment target
    /// is the (Linux) DGX Spark. Elsewhere this is a no-op and reads stay buffered.
    #[cfg(target_os = "linux")]
    fn fadvise_dontneed(&self, file_idx: usize, off: u64, len: usize) {
        use std::os::unix::io::AsRawFd;
        // SAFETY: `fd` is owned by `self.files` and outlives the call; posix_fadvise
        // only consults the page cache and never touches user memory.
        unsafe {
            libc::posix_fadvise(
                self.files[file_idx].1.as_raw_fd(),
                off as libc::off_t,
                len as libc::off_t,
                libc::POSIX_FADV_DONTNEED,
            );
        }
    }

    #[cfg(not(target_os = "linux"))]
    fn fadvise_dontneed(&self, _file_idx: usize, _off: u64, _len: usize) {}

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

    /// The tile size feeds O_DIRECT's alignment requirement, so every accepted value
    /// must be a 512-multiple and anything unusable must fall back rather than produce
    /// a misaligned job (which fails the read at runtime, not at parse time).
    #[test]
    fn read_sub_kb_rounds_to_512_and_falls_back() {
        const DEFAULT: usize = 2 << 20;
        assert_eq!(parse_sub_kb(None), DEFAULT);
        assert_eq!(parse_sub_kb(Some("512")), 512 * 1024);
        assert_eq!(parse_sub_kb(Some(" 256 ")), 256 * 1024, "surrounding space tolerated");
        // Not a 512-multiple in bytes -> rounded DOWN, never up past the request.
        assert_eq!(parse_sub_kb(Some("1")), 1024);
        // Unusable settings fall back to the default instead of erroring or yielding 0:
        // a 0-length job would spin the tiling loop forever.
        for bad in ["0", "", "-1", "abc", "1.5"] {
            assert_eq!(parse_sub_kb(Some(bad)), DEFAULT, "{bad:?} should fall back");
        }
        // Every accepted value is 512-aligned, which is what the tiling relies on.
        for kb in [1usize, 7, 64, 256, 1024, 4096] {
            assert_eq!(parse_sub_kb(Some(&kb.to_string())) % 512, 0);
        }
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
    fn mapped_spans_are_taken_and_byte_identical_to_the_copy_path() {
        // A just-written file is page-cache resident, so the residency gate opens and
        // both readers should serve views. The point of the test is that they are
        // ACTUALLY views (otherwise it silently only ever tests the copy path) and that
        // the bytes are identical to what the file holds.
        let dir = temp_dir();
        let w = 1usize << 20;
        let mut entries: Vec<(String, usize, usize)> = Vec::new();
        let mut off = 0;
        for part in ["g", "u", "d"] {
            entries.push((format!("e0.{part}"), off, w));
            off += w;
        }
        let data: Vec<u8> = (0..off).map(|i| ((i * 7 + 3) % 251) as u8).collect();
        let eref: Vec<(&str, usize, usize)> =
            entries.iter().map(|(n, o, l)| (n.as_str(), *o, *l)).collect();
        write_u8_shard(&dir, &eref, &data);
        let s = Shards::open(&dir).unwrap();

        let names = ["e0.g", "e0.u", "e0.d"];
        let got = s.read_raw_shared(&names, 4).unwrap();
        assert_eq!(got.len(), 3);

        // The mapped path must be the one under test wherever it exists. Mapping is
        // Linux-only, so elsewhere this necessarily falls back to the copy path — record
        // which one ran rather than letting the test silently cover only the fallback.
        let was_view = got[0].0.is_view();
        if mmap_enabled() && cfg!(target_os = "linux") {
            assert!(was_view, "a resident span should be served from the mapping");
        }

        // Contents match the file, and the three tensors coalesced into ONE span, so
        // all three views share a single buffer at increasing offsets.
        for (i, (buf, o, l)) in got.iter().enumerate() {
            assert_eq!(*l, w);
            assert_eq!(&buf[*o..*o + *l], &data[i * w..(i + 1) * w], "tensor {i} mismatch");
        }
        assert!(
            Arc::ptr_eq(&got[0].0, &got[1].0) && Arc::ptr_eq(&got[1].0, &got[2].0),
            "contiguous tensors should share one span buffer"
        );

        // A view must never be recycled into the heap pool: dropping it and then
        // allocating must not hand back the mapping's memory. Only meaningful when a view
        // actually ran — for a HEAP buffer the pool is *supposed* to return the same
        // allocation, which is what this asserts the absence of.
        let ptr = got[0].0.as_ptr();
        drop(got);
        let fresh = SharedBuf::with_len(w);
        if was_view {
            assert_ne!(fresh.as_ptr(), ptr, "a mapped view leaked into the buffer pool");
        } else {
            assert_eq!(fresh.as_ptr(), ptr, "a heap span should be recycled by the pool");
        }
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
