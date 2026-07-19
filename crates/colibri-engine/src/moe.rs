//! Mixture-of-Experts block — port of `moe()` from `c/glm.c` (the CPU core).
//!
//! GLM-5.2 routing: a sigmoid router with a per-expert `e_score_correction_bias`
//! (DeepSeek-V3 noaux_tc), top-K by the bias-augmented score, but the routing
//! *weights* are the raw sigmoids (bias affects selection only). Optionally
//! renormalized (`norm_topk_prob`) and scaled (`routed_scaling_factor`). Each
//! selected expert and the always-on shared expert are SwiGLU FFNs:
//! `down(silu(gate·x) ⊙ up·x)`.
//!
//! The routed experts are **streamed** — not held in the model — so this block
//! fetches each one it needs through an [`ExpertProvider`]. That indirection is
//! also the expert-parallel split point: [`ShardsExpertProvider`] checks
//! `colibri-cluster` ownership and (today, single-node) loads locally; a future
//! provider will fetch non-local experts over the RDMA transport.
//!
//! Not yet ported: the expert LRU/pinned-hot-store cache, the CACHE_ROUTE / top-p
//! routing variants (all opt-in, default off), and the batched GPU groups. This
//! is the exact default CPU path.

use crate::linear::{matmul_f32, matmul_qt};
use crate::math::silu;
use crate::model::Layer;
use colibri_cluster::{ExpertRequest, ExpertSharding, NodeId, Transport};
use colibri_core::{Bytes, Config, QTensor};
use colibri_safetensors::Shards;
use std::io;
use std::sync::{Arc, OnceLock};

/// Process-wide expert-parallel context. `serve`/`worker` set this once at startup
/// when `COLI_NUM_NODES > 1`; while present, [`moe`] transparently dispatches to
/// [`moe_sharded`] so the forward pass needs no signature change. Left unset on a
/// single node (and in tests), so `moe` runs the plain local path.
pub struct ClusterCtx {
    pub sharding: ExpertSharding,
    pub transport: Box<dyn Transport>,
}

static CLUSTER: OnceLock<ClusterCtx> = OnceLock::new();

/// Optional expert-routing log, enabled with `COLI_EXPERT_LOG=<file>` (or
/// `stderr`). Each routed position writes one line `step layer pos e0 e1 … ek`
/// (top-K expert ids, best-first). `step` is the forward/decode-token counter, so
/// the sequence of experts **across layers** within a token (predict layer L+1 from
/// L) and **across tokens** at the same layer (temporal locality) can both be mined
/// offline — the raw material for a predictive expert prefetcher.
fn expert_log() -> Option<&'static std::sync::Mutex<Box<dyn io::Write + Send>>> {
    static LOG: OnceLock<Option<std::sync::Mutex<Box<dyn io::Write + Send>>>> = OnceLock::new();
    LOG.get_or_init(|| {
        use io::Write;
        let path = std::env::var("COLI_EXPERT_LOG").ok()?;
        let mut w: Box<dyn io::Write + Send> = if matches!(path.as_str(), "stderr" | "-" | "1") {
            Box::new(io::stderr())
        } else {
            match std::fs::File::create(&path) {
                Ok(f) => Box::new(std::io::BufWriter::new(f)),
                Err(e) => {
                    eprintln!("[expert-log] cannot open {path}: {e}");
                    return None;
                }
            }
        };
        let _ = writeln!(w, "# step layer pos experts...  (top-K routed, best-first)");
        Some(std::sync::Mutex::new(w))
    })
    .as_ref()
}

/// Write the per-position routing lines `step layer pos e0 … ek` to `w`.
fn write_routing_lines<W: io::Write + ?Sized>(
    w: &mut W,
    step: u64,
    layer: usize,
    s_len: usize,
    k: usize,
    idxs: &[usize],
) -> io::Result<()> {
    for s in 0..s_len {
        write!(w, "{step} {layer} {s}")?;
        for kk in 0..k {
            write!(w, " {}", idxs[s * k + kk])?;
        }
        writeln!(w)?;
    }
    Ok(())
}

/// Emit one routing line per position when the expert log is enabled (no-op
/// otherwise). `idxs` is the `[s_len * k]` top-K expert ids from routing.
fn log_routing(layer: usize, s_len: usize, k: usize, idxs: &[usize]) {
    let lg = match expert_log() {
        Some(l) => l,
        None => return,
    };
    let step = crate::forward::current_step();
    if let Ok(mut w) = lg.lock() {
        let _ = write_routing_lines(&mut **w, step, layer, s_len, k, idxs);
        let _ = w.flush(); // opt-in log; keep it durable (the writer is never dropped)
    }
}

/// Install the cluster context (idempotent; a second call is ignored).
pub fn set_cluster(ctx: ClusterCtx) {
    let _ = CLUSTER.set(ctx);
}

/// The installed cluster context, if multi-node.
pub fn cluster_ctx() -> Option<&'static ClusterCtx> {
    CLUSTER.get()
}

/// One routed expert's SwiGLU weights.
#[derive(Debug, Clone, Default)]
pub struct Expert {
    /// gate_proj `[moe_inter, hidden]`
    pub gate: QTensor,
    /// up_proj `[moe_inter, hidden]`
    pub up: QTensor,
    /// down_proj `[hidden, moe_inter]`
    pub down: QTensor,
}

impl Expert {
    /// Resident byte size of this expert (sum of the three tensors).
    pub fn bytes(&self) -> u64 {
        (self.gate.bytes() + self.up.bytes() + self.down.bytes()) as u64
    }

    /// Mark all three tensors as GPU-cacheable (for preloaded/resident experts).
    pub fn mark_gpu_eligible(&mut self) {
        self.gate.gpu_eligible = true;
        self.up.gpu_eligible = true;
        self.down.gpu_eligible = true;
    }
}

/// Supplies routed experts to the MoE block on demand. The split point between
/// single-node local loads and multi-node remote fetches.
///
/// Returns `Arc<Expert>` so a resident cache ([`crate::cache::ExpertCache`]) can
/// hand out shared references without copying ~19 MB of weights per token.
pub trait ExpertProvider {
    fn expert(&self, layer: usize, eid: usize) -> io::Result<Arc<Expert>>;

    /// Preload `eids` for `layer` into RAM ahead of use. A resident cache reads
    /// the missing ones **in parallel** (disk→RAM is the decode bottleneck once
    /// compute is on the GPU); the default is a no-op for cacheless providers,
    /// which load lazily in [`ExpertProvider::expert`].
    fn prefetch(&self, _layer: usize, _eids: &[usize]) -> io::Result<()> {
        Ok(())
    }

    /// Load several experts for `layer` at once, in `eids` order. Providers backed
    /// by local disk can pool the reads through one continuously-streaming worker
    /// set (see [`load_experts_batch`]) instead of a per-expert spawn/join; the
    /// default just loads each through [`ExpertProvider::expert`].
    fn experts_batch(&self, layer: usize, eids: &[usize]) -> io::Result<Vec<Arc<Expert>>> {
        eids.iter().map(|&e| self.expert(layer, e)).collect()
    }
}

/// Loads experts from local safetensors shards, honoring `colibri-cluster`
/// ownership. Single-node by default (every expert local).
pub struct ShardsExpertProvider<'a> {
    shards: &'a Shards,
    hidden: usize,
    moe_inter: usize,
    ebits: u32,
    sharding: ExpertSharding,
    this_node: NodeId,
    /// Cores each expert's ~18 MB read is chunked across (a single stream tops out
    /// far below the NVMe). `COLI_LOAD_THREADS` overrides; defaults to core count.
    read_threads: usize,
}

/// Read-thread count for on-demand expert streaming: `COLI_LOAD_THREADS` else cores.
fn default_read_threads() -> usize {
    std::env::var("COLI_LOAD_THREADS")
        .ok()
        .and_then(|s| s.parse().ok())
        .unwrap_or_else(crate::preload::default_num_files)
}

impl<'a> ShardsExpertProvider<'a> {
    /// Single-node provider: all `n_experts` are local.
    pub fn new(shards: &'a Shards, cfg: &Config, ebits: u32) -> ShardsExpertProvider<'a> {
        ShardsExpertProvider {
            shards,
            hidden: cfg.hidden as usize,
            moe_inter: cfg.moe_inter as usize,
            ebits,
            sharding: ExpertSharding::single(cfg.n_experts as u32),
            this_node: NodeId(0),
            read_threads: default_read_threads(),
        }
    }

    /// Provider for one node of an expert-parallel cluster.
    pub fn with_sharding(
        shards: &'a Shards,
        cfg: &Config,
        ebits: u32,
        sharding: ExpertSharding,
        this_node: NodeId,
    ) -> ShardsExpertProvider<'a> {
        ShardsExpertProvider {
            shards,
            hidden: cfg.hidden as usize,
            moe_inter: cfg.moe_inter as usize,
            ebits,
            sharding,
            this_node,
            read_threads: default_read_threads(),
        }
    }
}

/// GLM tensor name of a routed expert's `gate_proj` (also the sort key for
/// offset-ordered parallel loading).
pub fn expert_gate_name(layer: usize, eid: usize) -> String {
    format!("model.layers.{layer}.mlp.experts.{eid}.gate_proj.weight")
}

/// Whether streamed experts should run on the GPU (marked `gpu_eligible`). Read
/// once. **On exactly when the zero-copy path is available** (unified memory, e.g.
/// GB10) — there the GPU reads the expert's RAM buffer in place: no device copy, no
/// pointer-keyed device cache, and ~2× the copy path.
///
/// Streamed experts are *never* eligible off the zero-copy path, and
/// `COLI_GPU_EXPERTS=1` cannot force it. This is a safety property, not a tuning
/// knob: their payloads live in `SharedBuf` buffers that are **recycled through a
/// global pool**, so an address is reused by a different expert as soon as the
/// cache evicts. The copy path's device cache is keyed by exactly that address
/// (`upload_ffn`), so it would hit a stale entry and compute the wrong expert's
/// weights — silently. `=1` therefore only opts in when zero-copy is available;
/// `=0` opts out. Off the zero-copy path streamed experts run on the CPU, which is
/// slower but correct. (Unified memory is the only supported target, so this is not
/// a live configuration — the guard exists so it can't become one by accident.)
fn gpu_experts_enabled() -> bool {
    use std::sync::OnceLock;
    static ON: OnceLock<bool> = OnceLock::new();
    *ON.get_or_init(|| {
        let setting = std::env::var("COLI_GPU_EXPERTS").ok();
        let zerocopy = zerocopy_available();
        if setting.as_deref() == Some("1") && !zerocopy {
            eprintln!(
                "coli: COLI_GPU_EXPERTS=1 ignored — zero-copy is unavailable, and streamed \
                 experts cannot use the device copy path (their pooled buffers are recycled, \
                 so its address-keyed cache would return another expert's weights). \
                 Running them on the CPU instead."
            );
        }
        experts_gpu_decision(setting.as_deref(), zerocopy)
    })
}

/// Whether the zero-copy path is usable; always `false` without the `cuda` feature.
fn zerocopy_available() -> bool {
    #[cfg(feature = "cuda")]
    {
        crate::gpu::zerocopy()
    }
    #[cfg(not(feature = "cuda"))]
    {
        false
    }
}

/// Pure decision behind [`gpu_experts_enabled`]: `=0` opts out; anything else opts
/// in *only* when zero-copy is available. Split out so the safety property is
/// unit-testable without a GPU or the environment.
fn experts_gpu_decision(setting: Option<&str>, zerocopy: bool) -> bool {
    match setting {
        Some("0") => false,
        _ => zerocopy,
    }
}

/// Load one routed expert (gate/up/down) directly from the shards. Shared by
/// `ShardsExpertProvider` and the direct parallel preloader.
pub fn load_expert(
    shards: &Shards,
    hidden: usize,
    moe_inter: usize,
    ebits: u32,
    layer: usize,
    eid: usize,
    read_threads: usize,
) -> io::Result<Expert> {
    let wn = |suf: &str| format!("model.layers.{layer}.mlp.experts.{eid}.{suf}.weight");
    let (gate_w, up_w, down_w) = (wn("gate_proj"), wn("up_proj"), wn("down_proj"));
    let mut ex = if shards.has(&format!("{gate_w}.qs")) {
        // Pre-quantized container: the 3 weights are contiguous on disk (~18 MB),
        // so read them in ONE coalesced read into a shared buffer the tensors view
        // — instead of 3 separate reads + allocations (the streaming bottleneck).
        // The read is chunked across `read_threads` cores so a single miss saturates
        // the disk. Scales are tiny and elsewhere; keep them as small per-tensor reads.
        let ws = shards.read_raw_shared(&[&gate_w, &up_w, &down_w], read_threads)?;
        expert_from_views(shards, hidden, moe_inter, layer, eid, &ws)?
    } else {
        // Full-tensor (runtime-quantized) path — the tiny oracle model.
        Expert {
            gate: crate::loader::qt_load(shards, &gate_w, moe_inter, hidden, ebits)?,
            up: crate::loader::qt_load(shards, &up_w, moe_inter, hidden, ebits)?,
            down: crate::loader::qt_load(shards, &down_w, hidden, moe_inter, ebits)?,
        }
    };
    // Route streamed experts through the GPU fused-FFN path. This only ever happens
    // on unified memory (the GB10), via the zero-copy wrap: the kernel reads the RAM
    // copy in place, so there is no VRAM double-store, no eviction and no OOM — and
    // it is ~2× the copy path. Off the zero-copy path they stay on the CPU by
    // construction; see [`gpu_experts_enabled`] for why that is a safety property
    // rather than a tuning choice.
    if gpu_experts_enabled() {
        ex.mark_gpu_eligible();
    }
    Ok(ex)
}

/// `COLI_EXPERT_FP8=1` converts routed experts to e4m3 fp8 at load so the tiled
/// tensor-core kernel (`coli_cuda_expert_mlp_fp8`) runs instead of the naive
/// per-row `quant_matmul`. Off by default (doubles in-RAM expert size).
fn expert_fp8_enabled() -> bool {
    static ON: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *ON.get_or_init(|| std::env::var("COLI_EXPERT_FP8").ok().as_deref() == Some("1"))
}

/// Convert a packed int4 weight matrix (offset-binary nibbles, value = nibble − 8,
/// `o` rows × ceil(i/2) bytes) to e4m3 fp8 (1 byte/weight, `o`×`i`). The LUT was
/// generated + roundtrip-verified with the hardware fp8 encoder (`__nv_cvt_float_to_fp8`):
/// every int4 value −8..7 is exactly representable in e4m3.
fn int4_to_e4m3(src: &[u8], o: usize, i: usize) -> Vec<u8> {
    const LUT: [u8; 16] = [
        0xD0, 0xCE, 0xCC, 0xCA, 0xC8, 0xC4, 0xC0, 0xB8, 0x00, 0x38, 0x40, 0x44, 0x48, 0x4A, 0x4C,
        0x4E,
    ];
    let rb = i.div_ceil(2);
    let mut out = vec![0u8; o * i];
    for r in 0..o {
        let srow = &src[r * rb..r * rb + rb];
        let orow = &mut out[r * i..(r + 1) * i];
        for c in 0..i {
            let b = srow[c >> 1];
            orow[c] = LUT[(if c & 1 == 1 { b >> 4 } else { b & 0x0f }) as usize];
        }
    }
    out
}

/// Build an `Expert` from three raw weight views (`gate,up,down`, each as returned
/// by [`Shards::read_raw_shared`]/`read_raw_shared_batched`), reading the tiny
/// per-weight scales separately. Shared by the single-expert and batched loaders.
fn expert_from_views(
    shards: &Shards,
    hidden: usize,
    moe_inter: usize,
    layer: usize,
    eid: usize,
    views: &[(Arc<colibri_core::SharedBuf>, usize, usize)],
) -> io::Result<Expert> {
    let mk = |o: usize,
              i: usize,
              w: &(Arc<colibri_core::SharedBuf>, usize, usize),
              sname: String|
     -> io::Result<QTensor> {
        let (buf, off, len) = w;
        let fmt = if *len == o * i {
            1
        } else if *len == o * i.div_ceil(2) {
            2
        } else {
            3
        };
        let mut s = vec![0f32; o];
        shards.read_f32(&sname, &mut s)?;
        let mut t = QTensor { fmt_code: fmt, o: o as i32, i: i as i32, s, ..Default::default() };
        if fmt == 1 {
            // int8 goes in q8 (signed) — a copy; experts are int4 so this is rare.
            t.q8 = buf[*off..*off + *len].iter().map(|&b| b as i8).collect();
        } else {
            t.q4 = Bytes::Shared { buf: buf.clone(), off: *off, len: *len };
        }
        if expert_fp8_enabled() {
            if fmt == 2 {
                // int4 snapshot → convert to e4m3 at load (scaffolding for a non-fp8
                // container). Same per-row scales; e4m3 represents int4 −8..7 exactly,
                // so it is lossless vs the int4 weights. Doubles in-RAM size.
                t.q4 = Bytes::Owned(int4_to_e4m3(&buf[*off..*off + *len], o, i));
                t.fmt_code = 4;
            } else if fmt == 1 {
                // e4m3 snapshot (COLI_XFP8 container): the bytes are already e4m3 —
                // 1 B/weight, indistinguishable by length from int8. Use them directly,
                // no conversion. Routed experts are never genuinely int8.
                t.q8 = Vec::new();
                t.q4 = Bytes::Shared { buf: buf.clone(), off: *off, len: *len };
                t.fmt_code = 4;
            }
        }
        Ok(t)
    };
    let wn = |suf: &str| format!("model.layers.{layer}.mlp.experts.{eid}.{suf}.weight.qs");
    Ok(Expert {
        gate: mk(moe_inter, hidden, &views[0], wn("gate_proj"))?,
        up: mk(moe_inter, hidden, &views[1], wn("up_proj"))?,
        down: mk(hidden, moe_inter, &views[2], wn("down_proj"))?,
    })
}

/// Pool a whole layer's expert reads through one continuously-streaming worker
/// set instead of the per-expert spawn/join in [`load_expert`]. **On by default**;
/// set `COLI_READER_POOL=0` to fall back to the per-expert path. Measured on the
/// GB10 (PCIe-4-x4 NVMe): +19.6% decode tok/s in the miss-heavy regime with
/// byte-identical output, and 2.0× warm load bandwidth (9.27 → 18.58 GB/s). The
/// per-expert spawn/join barrier — paid ~18 times per expert — was the bottleneck.
fn reader_pool_enabled() -> bool {
    static ON: OnceLock<bool> = OnceLock::new();
    *ON.get_or_init(|| std::env::var("COLI_READER_POOL").ok().as_deref() != Some("0"))
}

/// Load several routed experts through the pooled batched reader — one worker set
/// drains every expert's sub-chunk reads, so the NVMe streams continuously rather
/// than stalling at a per-expert barrier. Falls back to per-expert loads for the
/// full-tensor (oracle) path. Returns experts in `eids` order.
pub fn load_experts_batch(
    shards: &Shards,
    hidden: usize,
    moe_inter: usize,
    ebits: u32,
    layer: usize,
    eids: &[usize],
    read_threads: usize,
) -> io::Result<Vec<Expert>> {
    if eids.is_empty() {
        return Ok(Vec::new());
    }
    // The pooled path applies only to the pre-quantized container (contiguous
    // gate|up|down + sidecar scales). Detect via the first expert's scales tensor.
    let probe = format!(
        "model.layers.{layer}.mlp.experts.{}.gate_proj.weight.qs",
        eids[0]
    );
    if !shards.has(&probe) {
        return eids
            .iter()
            .map(|&e| load_expert(shards, hidden, moe_inter, ebits, layer, e, read_threads))
            .collect();
    }
    // One [gate,up,down] name group per expert; keep the owned strings alive so
    // the borrowed &str slices handed to the reader stay valid.
    let names: Vec<[String; 3]> = eids
        .iter()
        .map(|&eid| {
            let wn = |suf: &str| format!("model.layers.{layer}.mlp.experts.{eid}.{suf}.weight");
            [wn("gate_proj"), wn("up_proj"), wn("down_proj")]
        })
        .collect();
    let groups: Vec<[&str; 3]> =
        names.iter().map(|g| [g[0].as_str(), g[1].as_str(), g[2].as_str()]).collect();
    let group_refs: Vec<&[&str]> = groups.iter().map(|g| &g[..]).collect();
    let views = shards.read_raw_shared_batched(&group_refs, read_threads)?;

    let mut out = Vec::with_capacity(eids.len());
    for (gi, &eid) in eids.iter().enumerate() {
        let mut ex = expert_from_views(shards, hidden, moe_inter, layer, eid, &views[gi])?;
        if gpu_experts_enabled() {
            ex.mark_gpu_eligible();
        }
        out.push(ex);
    }
    Ok(out)
}

impl ExpertProvider for ShardsExpertProvider<'_> {
    fn expert(&self, layer: usize, eid: usize) -> io::Result<Arc<Expert>> {
        // Expert-parallel ownership, enforced at the *load* layer. `moe_sharded`
        // already dispatches non-local experts to their owner over the transport and
        // never asks us for one, so reaching this is a bug (bad routing, or a node
        // built a different map). Erring is the point: without it we would silently
        // load a peer's expert from disk — right answer, wasted I/O, hidden bug.
        // Single-node providers use `ExpertSharding::single`, so everything is local.
        if !self.sharding.is_local(self.this_node, eid as u32) {
            return Err(io::Error::new(
                io::ErrorKind::Unsupported,
                format!(
                    "expert {eid} (layer {layer}) is owned by another node, not {}; \
                     it should have been dispatched over the transport",
                    self.this_node.0
                ),
            ));
        }
        Ok(Arc::new(load_expert(
            self.shards,
            self.hidden,
            self.moe_inter,
            self.ebits,
            layer,
            eid,
            self.read_threads,
        )?))
    }

    fn experts_batch(&self, layer: usize, eids: &[usize]) -> io::Result<Vec<Arc<Expert>>> {
        // Same ownership guard as `expert`: a non-local expert should have been
        // dispatched over the transport and never reach this local provider.
        for &eid in eids {
            if !self.sharding.is_local(self.this_node, eid as u32) {
                return Err(io::Error::new(
                    io::ErrorKind::Unsupported,
                    format!(
                        "expert {eid} (layer {layer}) is owned by another node, not {}; \
                         it should have been dispatched over the transport",
                        self.this_node.0
                    ),
                ));
            }
        }
        if reader_pool_enabled() {
            let exps = load_experts_batch(
                self.shards,
                self.hidden,
                self.moe_inter,
                self.ebits,
                layer,
                eids,
                self.read_threads,
            )?;
            Ok(exps.into_iter().map(Arc::new).collect())
        } else {
            eids.iter().map(|&e| self.expert(layer, e)).collect()
        }
    }
}

/// Route one position: apply sigmoid, add the selection bias, take top-K, and
/// return `(expert_ids, weights)`. Port of the default routing path in `moe()`.
///
/// Selection uses `sigmoid(logit) + bias`; the returned weights are the raw
/// `sigmoid(logit)` of the chosen experts, then optionally renormalized and
/// scaled by `routed_scaling_factor`.
pub fn route(cfg: &Config, logits: &[f32], bias: &[f32]) -> (Vec<usize>, Vec<f32>) {
    let e_n = logits.len();
    let k = (cfg.topk as usize).min(e_n);
    let logit: Vec<f32> = logits.iter().map(|&z| crate::math::sigmoid(z)).collect();
    let choice: Vec<f32> = (0..e_n).map(|e| logit[e] + bias[e]).collect();

    let mut idx = vec![0usize; k];
    let mut w = vec![0f32; k];
    let mut chosen = vec![false; e_n];
    for kk in 0..k {
        let mut best = 0usize;
        let mut bv = f32::NEG_INFINITY;
        for e in 0..e_n {
            if !chosen[e] && choice[e] > bv {
                bv = choice[e];
                best = e;
            }
        }
        chosen[best] = true;
        idx[kk] = best;
        w[kk] = logit[best];
    }
    if cfg.norm_topk {
        let sm: f32 = w.iter().sum::<f32>() + 1e-20;
        for x in w.iter_mut() {
            *x /= sm;
        }
    }
    for x in w.iter_mut() {
        *x *= cfg.routed_scale;
    }
    (idx, w)
}

/// Apply a SwiGLU FFN over `x[nr, D]` into `out[nr, D]`:
/// `out = down(silu(gate·x) ⊙ up·x)`. Port of the expert compute in `moe()`.
fn ffn(gate: &QTensor, up: &QTensor, down: &QTensor, x: &[f32], nr: usize, out: &mut [f32]) {
    // Fused GPU expert pipeline (one host round-trip) for resident weights.
    #[cfg(feature = "cuda")]
    {
        if crate::gpu::try_expert_ffn(gate, up, down, x, nr, out) {
            return;
        }
    }
    ffn_cpu(gate, up, down, x, nr, out);
}

/// CPU SwiGLU FFN (the reference / fallback path).
fn ffn_cpu(gate: &QTensor, up: &QTensor, down: &QTensor, x: &[f32], nr: usize, out: &mut [f32]) {
    let inter = gate.o as usize; // moe_inter (or shared intermediate)
    let mut gg = vec![0f32; nr * inter];
    let mut uu = vec![0f32; nr * inter];
    matmul_qt(&mut gg, x, gate, nr);
    matmul_qt(&mut uu, x, up, nr);
    for (g, &u) in gg.iter_mut().zip(uu.iter()) {
        *g = silu(*g) * u;
    }
    matmul_qt(out, &gg, down, nr);
}

/// Dense MLP for non-MoE layers (the first `first_k_dense_replace` layers):
/// the same SwiGLU as an expert, over `gate_proj`/`up_proj`/`down_proj`. Port of
/// `dense_mlp` in `c/glm.c`.
pub fn dense_mlp(l: &Layer, x: &[f32], s_len: usize, out: &mut [f32]) {
    ffn(&l.gate_proj, &l.up_proj, &l.down_proj, x, s_len, out);
}

/// Union of the routed experts across the batch, plus a dense `[S, n_uniq]` weight
/// matrix: `w_mat[s * n_uniq + ui]` is the routing weight of token `s` for
/// `uniq[ui]` (0 if it doesn't route there). This is the exact per-(token,expert)
/// weight the expert loop applies, laid out for [`compute_experts_partial`].
fn union_and_weights(
    idxs: &[usize],
    ws: &[f32],
    s_len: usize,
    k: usize,
    e_n: usize,
) -> (Vec<usize>, Vec<f32>) {
    let mut seen = vec![usize::MAX; e_n]; // expert id -> its column in uniq
    let mut uniq = Vec::new();
    for &e in idxs {
        if seen[e] == usize::MAX {
            seen[e] = uniq.len();
            uniq.push(e);
        }
    }
    let n_uniq = uniq.len();
    let mut w_mat = vec![0f32; s_len * n_uniq];
    for s in 0..s_len {
        for kk in 0..k {
            let e = idxs[s * k + kk];
            w_mat[s * n_uniq + seen[e]] = ws[s * k + kk];
        }
    }
    (uniq, w_mat)
}

/// The one expert-compute primitive: for each token `t`, accumulate
/// `Σ_e weights[t * n_experts + e] * expert_e(activations[t])` and return the flat
/// `[n_tokens * hidden]` partial MoE sum. `moe()` runs it over all experts locally;
/// `moe_sharded()` runs it over the node's own experts; and the transport server
/// runs it as the handler for a peer's [`ExpertRequest`]. Zero-weight (token,
/// expert) pairs are skipped, so a token only touches the experts it routes to.
pub fn compute_experts_partial<P: ExpertProvider>(
    provider: &P,
    layer: usize,
    experts: &[u32],
    weights: &[f32],
    activations: &[f32],
    n_tokens: usize,
    hidden: usize,
) -> io::Result<Vec<f32>> {
    let d = hidden;
    let ne = experts.len();
    let mut out = vec![0f32; n_tokens * d];
    if ne == 0 {
        return Ok(out);
    }
    let eids: Vec<usize> = experts.iter().map(|&e| e as usize).collect();

    // Fetch this layer's experts disk→RAM in parallel before computing (serial
    // per-expert loading is otherwise ~74% of MoE time).
    if crate::forward::profile_on() {
        let t = std::time::Instant::now();
        provider.prefetch(layer, &eids)?;
        crate::forward::LOAD_US
            .fetch_add(t.elapsed().as_micros() as u64, std::sync::atomic::Ordering::Relaxed);
    } else {
        provider.prefetch(layer, &eids)?;
    }

    // Per-expert row lists: the tokens routing to each expert, with their weights.
    let mut per_expert: Vec<(usize, Vec<usize>, Vec<f32>)> = Vec::new();
    for (ei, &e) in eids.iter().enumerate() {
        let mut rows = Vec::new();
        let mut rw = Vec::new();
        for t in 0..n_tokens {
            let w = weights[t * ne + ei];
            if w != 0.0 {
                rows.push(t);
                rw.push(w);
            }
        }
        if !rows.is_empty() {
            per_expert.push((e, rows, rw));
        }
    }

    // Batched grouped path (`COLI_EXPERT_GROUP`): one H2D/D2H per ≤64-expert chunk
    // instead of a synchronous upload/kernel/download per expert — the per-expert
    // round-trip is what dominates moe-compute. Falls through per-expert if it can't run.
    #[cfg(feature = "cuda")]
    if crate::gpu::expert_group_enabled() {
        let mut active = Vec::with_capacity(per_expert.len());
        for (e, rows, rw) in &per_expert {
            active.push((provider.expert(layer, *e)?, rows.clone(), rw.clone()));
        }
        if crate::gpu::try_expert_group(&active, activations, d, &mut out) {
            return Ok(out);
        }
    }

    let prof = crate::forward::profile_on();
    for (e, rows, rw) in &per_expert {
        let nr = rows.len();
        let ex = provider.expert(layer, *e)?; // cache hit (prefetched); not timed here
        let mut xg = vec![0f32; nr * d];
        let t0 = std::time::Instant::now();
        for (r, &t) in rows.iter().enumerate() {
            xg[r * d..(r + 1) * d].copy_from_slice(&activations[t * d..(t + 1) * d]);
        }
        if prof {
            crate::forward::GATHER_US
                .fetch_add(t0.elapsed().as_micros() as u64, std::sync::atomic::Ordering::Relaxed);
        }
        let mut hh = vec![0f32; nr * d];
        let t1 = std::time::Instant::now();
        ffn(&ex.gate, &ex.up, &ex.down, &xg, nr, &mut hh);
        if prof {
            crate::forward::GPUFFN_US
                .fetch_add(t1.elapsed().as_micros() as u64, std::sync::atomic::Ordering::Relaxed);
        }
        let t2 = std::time::Instant::now();
        for (r, &t) in rows.iter().enumerate() {
            let wgt = rw[r];
            for dd in 0..d {
                out[t * d + dd] += wgt * hh[r * d + dd];
            }
        }
        if prof {
            crate::forward::SCATTER_US
                .fetch_add(t2.elapsed().as_micros() as u64, std::sync::atomic::Ordering::Relaxed);
        }
    }
    Ok(out)
}

/// Sub-column a `[S, n_uniq]` weight matrix down to the experts in `cols` (their
/// positions in `uniq`), giving a `[S, cols.len()]` matrix aligned to `cols`.
fn subcols(w_mat: &[f32], s_len: usize, n_uniq: usize, cols: &[usize]) -> Vec<f32> {
    let mut out = vec![0f32; s_len * cols.len()];
    for s in 0..s_len {
        for (j, &c) in cols.iter().enumerate() {
            out[s * cols.len() + j] = w_mat[s * n_uniq + c];
        }
    }
    out
}

/// Expert-parallel MoE: identical to [`moe`], but the routed experts are split by
/// ownership — this node computes the experts it owns in-process and fetches the
/// partial sums for experts owned by peers over `transport` (sending the token
/// activations + routing weights, receiving `Σ w·expert(x)`). On a single node
/// (`sharding.num_nodes() == 1`) every expert is local and no `exchange` happens,
/// so it matches `moe` exactly. `provider` must be able to load *this node's*
/// experts; the peer's `serve_experts` handler computes theirs.
/// Router projection `logits[s,e] = x[s,d] @ router[e,d]^T`. Runs on the GPU (full
/// f32, no quality change) when CUDA is available — a single-threaded CPU `matmul_f32`
/// here was ~40% of moe-compute at long context — falling back to CPU otherwise.
#[inline]
fn router_matmul(logits: &mut [f32], x: &[f32], router: &[f32], s_len: usize, d: usize, e_n: usize) {
    #[cfg(feature = "cuda")]
    {
        if crate::gpu::try_matmul_f32(logits, x, router, s_len, d, e_n) {
            return;
        }
    }
    matmul_f32(logits, x, router, s_len, d, e_n);
}

#[allow(clippy::too_many_arguments)]
pub fn moe_sharded<P: ExpertProvider, T: Transport + ?Sized>(
    cfg: &Config,
    l: &Layer,
    layer: usize,
    x: &[f32],
    s_len: usize,
    out: &mut [f32],
    with_shared: bool,
    provider: &P,
    sharding: &ExpertSharding,
    transport: &T,
) -> io::Result<()> {
    let d = cfg.hidden as usize;
    let e_n = cfg.n_experts as usize;
    let k = (cfg.topk as usize).min(e_n);

    let mut logits = vec![0f32; s_len * e_n];
    router_matmul(&mut logits, x, &l.router, s_len, d, e_n);
    let mut idxs = vec![0usize; s_len * k];
    let mut ws = vec![0f32; s_len * k];
    for s in 0..s_len {
        let (idx, w) = route(cfg, &logits[s * e_n..(s + 1) * e_n], &l.router_bias);
        idxs[s * k..s * k + k].copy_from_slice(&idx);
        ws[s * k..s * k + k].copy_from_slice(&w);
    }
    log_routing(layer, s_len, k, &idxs);
    for v in out.iter_mut() {
        *v = 0.0;
    }

    let (uniq, w_mat) = union_and_weights(&idxs, &ws, s_len, k, e_n);
    let n_uniq = uniq.len();
    let me = transport.this_node();

    // Partition the unique experts by owning node (columns into w_mat).
    let mut by_node: std::collections::BTreeMap<u32, Vec<usize>> = std::collections::BTreeMap::new();
    for (ui, &e) in uniq.iter().enumerate() {
        by_node.entry(sharding.owner(e as u32).0).or_default().push(ui);
    }

    for (node, cols) in by_node {
        let experts: Vec<u32> = cols.iter().map(|&ui| uniq[ui] as u32).collect();
        let weights = subcols(&w_mat, s_len, n_uniq, &cols);
        if NodeId(node) == me {
            // Local: compute in-process against our provider.
            let partial = compute_experts_partial(provider, layer, &experts, &weights, x, s_len, d)?;
            for (o, p) in out.iter_mut().zip(partial.iter()) {
                *o += *p;
            }
        } else {
            // Remote: ship activations + weights to the owner, add its partial sum.
            let req = ExpertRequest {
                experts,
                weights,
                activations: x.to_vec(),
                n_tokens: s_len,
                hidden: d,
                layer: layer as u32,
            };
            let resp = transport
                .exchange(NodeId(node), &req)
                .map_err(|e| io::Error::new(io::ErrorKind::Other, e.to_string()))?;
            if resp.outputs.len() != s_len * d {
                return Err(io::Error::new(
                    io::ErrorKind::InvalidData,
                    format!("node {node}: expected {} outputs, got {}", s_len * d, resp.outputs.len()),
                ));
            }
            for (o, p) in out.iter_mut().zip(resp.outputs.iter()) {
                *o += *p;
            }
        }
    }

    if with_shared {
        let mut sh = vec![0f32; s_len * d];
        ffn(&l.sh_gate, &l.sh_up, &l.sh_down, x, s_len, &mut sh);
        for (o, &s) in out.iter_mut().zip(sh.iter()) {
            *o += s;
        }
    }
    Ok(())
}

/// MoE forward over `x[S, hidden]` into `out[S, hidden]`. Routes each position,
/// applies every selected expert (fetched via `provider`), and adds the shared
/// expert when `with_shared`. Port of `moe()`'s default CPU path.
pub fn moe<P: ExpertProvider>(
    cfg: &Config,
    l: &Layer,
    layer: usize,
    x: &[f32],
    s_len: usize,
    out: &mut [f32],
    with_shared: bool,
    provider: &P,
) -> io::Result<()> {
    // Expert-parallel dispatch: when a multi-node cluster context is installed,
    // route experts by ownership (local in-process, remote over the transport).
    // Single node (or unset) falls through to the local path below.
    if let Some(ctx) = cluster_ctx() {
        if ctx.sharding.num_nodes() > 1 {
            return moe_sharded(
                cfg, l, layer, x, s_len, out, with_shared, provider, &ctx.sharding, &*ctx.transport,
            );
        }
    }

    let d = cfg.hidden as usize;
    let e_n = cfg.n_experts as usize;
    let k = (cfg.topk as usize).min(e_n);

    // ---- router (f32) + top-K per position --------------------------------
    let mut logits = vec![0f32; s_len * e_n];
    let _rt = std::time::Instant::now();
    router_matmul(&mut logits, x, &l.router, s_len, d, e_n);
    if crate::forward::profile_on() {
        crate::forward::ROUTER_US
            .fetch_add(_rt.elapsed().as_micros() as u64, std::sync::atomic::Ordering::Relaxed);
    }

    let mut idxs = vec![0usize; s_len * k];
    let mut ws = vec![0f32; s_len * k];
    for s in 0..s_len {
        let (idx, w) = route(cfg, &logits[s * e_n..(s + 1) * e_n], &l.router_bias);
        idxs[s * k..s * k + k].copy_from_slice(&idx);
        ws[s * k..s * k + k].copy_from_slice(&w);
    }
    log_routing(layer, s_len, k, &idxs);

    for v in out.iter_mut() {
        *v = 0.0;
    }

    // ---- routed experts (all local on a single node) ----------------------
    let (uniq, w_mat) = union_and_weights(&idxs, &ws, s_len, k, e_n);
    let uniq_u32: Vec<u32> = uniq.iter().map(|&e| e as u32).collect();
    let partial = compute_experts_partial(provider, layer, &uniq_u32, &w_mat, x, s_len, d)?;
    for (o, p) in out.iter_mut().zip(partial.iter()) {
        *o += *p;
    }

    // ---- shared expert (weight 1.0, all positions) ------------------------
    if with_shared {
        let _st = std::time::Instant::now();
        let mut sh = vec![0f32; s_len * d];
        ffn(&l.sh_gate, &l.sh_up, &l.sh_down, x, s_len, &mut sh);
        for (o, &s) in out.iter_mut().zip(sh.iter()) {
            *o += s;
        }
        if crate::forward::profile_on() {
            crate::forward::SHARED_US
                .fetch_add(_st.elapsed().as_micros() as u64, std::sync::atomic::Ordering::Relaxed);
        }
    }

    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::quantize::qtensor_from_f32;
    use std::collections::HashMap;

    // The int4→e4m3 conversion + CPU e4m3 decode must reproduce the int4 matmul: every
    // int4 value −8..7 is exact in e4m3, so the two paths compute (nibble−8)·scale
    // identically (only f32 summation order differs). Guards the fp8 plumbing's math.
    #[test]
    fn int4_to_e4m3_reproduces_int4_matmul() {
        use crate::linear::matmul_qt;
        let (o, i, ns) = (4usize, 8usize, 3usize);
        let rb = i.div_ceil(2);
        let q4: Vec<u8> = (0..o * rb).map(|k| (k * 37 + 5) as u8).collect();
        let s: Vec<f32> = (0..o).map(|r| 0.5 + r as f32 * 0.25).collect();
        let mk = |fmt: i32, bytes: Vec<u8>| QTensor {
            fmt_code: fmt,
            q4: Bytes::Owned(bytes),
            s: s.clone(),
            o: o as i32,
            i: i as i32,
            ..Default::default()
        };
        let int4 = mk(2, q4.clone());
        let fp8 = mk(4, int4_to_e4m3(&q4, o, i));
        let x: Vec<f32> = (0..ns * i).map(|k| k as f32 * 0.1 - 0.35).collect();
        let (mut y4, mut y8) = (vec![0f32; ns * o], vec![0f32; ns * o]);
        matmul_qt(&mut y4, &x, &int4, ns);
        matmul_qt(&mut y8, &x, &fp8, ns);
        for (a, b) in y4.iter().zip(&y8) {
            assert!((a - b).abs() < 1e-4, "int4 {a} vs e4m3 {b}");
        }
    }

    // In-memory provider for MoE math tests (no safetensors needed).
    struct MapProvider {
        experts: HashMap<(usize, usize), Arc<Expert>>,
    }
    impl ExpertProvider for MapProvider {
        fn expert(&self, layer: usize, eid: usize) -> io::Result<Arc<Expert>> {
            self.experts
                .get(&(layer, eid))
                .cloned()
                .ok_or_else(|| io::Error::new(io::ErrorKind::NotFound, "no expert"))
        }
    }

    fn cfg() -> Config {
        let json = colibri_json::Json::parse(
            r#"{"hidden_size":4,"num_hidden_layers":1,"num_attention_heads":1,
                "n_routed_experts":4,"num_experts_per_tok":2,"moe_intermediate_size":3,
                "intermediate_size":4,"first_k_dense_replace":0,"q_lora_rank":2,
                "kv_lora_rank":2,"qk_nope_head_dim":2,"qk_rope_head_dim":2,"v_head_dim":2,
                "n_shared_experts":1,"vocab_size":8,"n_group":1,"topk_group":1,
                "norm_topk_prob":false,"rms_norm_eps":1e-5,"routed_scaling_factor":1.0,
                "rope_parameters":{"rope_theta":10000.0},"eos_token_id":[7],
                "index_topk":0,"index_n_heads":0,"index_head_dim":0}"#,
        )
        .unwrap();
        Config::from_json(&json).unwrap()
    }

    fn expert(seed: usize, inter: usize, d: usize) -> Expert {
        let mk = |o: usize, i: usize, s: usize| {
            let w: Vec<f32> = (0..o * i)
                .map(|k| (((k * 3 + s * 7 + 1) % 9) as f32 - 4.0) * 0.1)
                .collect();
            qtensor_from_f32(&w, o, i, 16)
        };
        Expert {
            gate: mk(inter, d, seed),
            up: mk(inter, d, seed + 1),
            down: mk(d, inter, seed + 2),
        }
    }

    // Fused GPU expert FFN vs CPU at GLM expert sizes (hidden 6144, moe_inter 2048).
    // `cargo test -p colibri-engine --features cuda --release -- --ignored --nocapture bench_expert_ffn`
    #[cfg(feature = "cuda")]
    #[test]
    #[ignore]
    fn bench_expert_ffn_gpu_vs_cpu() {
        if !crate::gpu::available() {
            eprintln!("skip: no CUDA device");
            return;
        }
        let (d, inter) = (6144usize, 2048usize);
        let mk = |o: usize, i: usize| {
            let w: Vec<f32> = (0..o * i).map(|k| ((k % 13) as f32 - 6.0) * 0.01).collect();
            let mut t = qtensor_from_f32(&w, o, i, 4);
            t.gpu_eligible = true;
            t
        };
        let mut gate = mk(inter, d);
        let mut up = mk(inter, d);
        let mut down = mk(d, inter);
        let nr = 1usize;
        let x = vec![0.01f32; nr * d];
        let mut out = vec![0f32; nr * d];
        let iters = 1000u64;
        ffn(&gate, &up, &down, &x, nr, &mut out); // warm upload
        let t = std::time::Instant::now();
        for _ in 0..iters {
            ffn(&gate, &up, &down, &x, nr, &mut out);
        }
        let gpu = t.elapsed().as_secs_f64();
        gate.gpu_eligible = false;
        up.gpu_eligible = false;
        down.gpu_eligible = false;
        let t = std::time::Instant::now();
        for _ in 0..iters {
            ffn(&gate, &up, &down, &x, nr, &mut out);
        }
        let cpu = t.elapsed().as_secs_f64();
        eprintln!(
            "expert FFN (d={d} inter={inter} nr={nr}) x{iters}: GPU-fused {:.3}s ({:.0} us/expert) | CPU-NEON {:.3}s ({:.0} us) | {:.2}x",
            gpu,
            gpu / iters as f64 * 1e6,
            cpu,
            cpu / iters as f64 * 1e6,
            cpu / gpu
        );
    }

    #[test]
    fn route_selects_top_k_by_bias_augmented_score() {
        let c = cfg(); // topk=2, 4 experts
        // logits chosen so sigmoid is monotonic; bias flips the order.
        let logits = [0.0f32, 1.0, 2.0, 3.0]; // sigmoids ~ .5,.73,.88,.95
        let bias = [10.0f32, 0.0, 0.0, 0.0]; // huge bias on expert 0
        let (idx, w) = route(&c, &logits, &bias);
        // expert 0 wins on bias; expert 3 is next by sigmoid.
        assert_eq!(idx, vec![0, 3]);
        // weights are the RAW sigmoids (bias not included)
        assert!((w[0] - crate::math::sigmoid(0.0)).abs() < 1e-6);
        assert!((w[1] - crate::math::sigmoid(3.0)).abs() < 1e-6);
    }

    #[test]
    fn norm_topk_normalizes_weights() {
        let mut c = cfg();
        c.norm_topk = true;
        c.routed_scale = 2.0;
        let logits = [3.0f32, 2.0, 1.0, 0.0];
        let bias = [0.0f32; 4];
        let (_idx, w) = route(&c, &logits, &bias);
        // after norm the weights sum to routed_scale (2.0)
        let sum: f32 = w.iter().sum();
        assert!((sum - 2.0).abs() < 1e-5, "sum {sum}");
    }

    #[test]
    fn single_expert_moe_equals_weighted_ffn() {
        // topk=1, no shared: out == w * ffn(chosen expert). Independent check of
        // router weight * FFN * accumulation.
        let mut c = cfg();
        c.topk = 1;
        let d = c.hidden as usize;
        let inter = c.moe_inter as usize;

        let mut l = Layer::default();
        // router that always picks expert 2 (largest logit) — bias 0.
        let mut router = vec![0f32; c.n_experts as usize * d];
        // expert 2's row large so its logit dominates
        for i in 0..d {
            router[2 * d + i] = 1.0;
        }
        l.router = router;
        l.router_bias = vec![0.0; c.n_experts as usize];

        let ex2 = expert(20, inter, d);
        let mut experts = HashMap::new();
        experts.insert((0usize, 2usize), Arc::new(ex2.clone()));
        let provider = MapProvider { experts };

        let x = vec![0.3f32, 0.5, -0.2, 0.7];
        let mut out = vec![0f32; d];
        moe(&c, &l, 0, &x, 1, &mut out, false, &provider).unwrap();

        // expected: w * ffn(ex2, x), w = sigmoid(router·x) * routed_scale(1)
        let logit = x.iter().sum::<f32>(); // router row 2 is all ones
        let w = crate::math::sigmoid(logit);
        let mut ffn_out = vec![0f32; d];
        ffn(&ex2.gate, &ex2.up, &ex2.down, &x, 1, &mut ffn_out);
        for dd in 0..d {
            assert!(
                (out[dd] - w * ffn_out[dd]).abs() < 1e-5,
                "out {} vs {}",
                out[dd],
                w * ffn_out[dd]
            );
        }
    }

    #[test]
    fn shared_expert_adds_its_ffn() {
        // out(with_shared) - out(without) == shared FFN(x).
        let c = cfg();
        let d = c.hidden as usize;
        let inter = c.moe_inter as usize;
        let s_i = (c.moe_inter * c.n_shared) as usize;

        let mut l = Layer::default();
        l.router = vec![0.1f32; c.n_experts as usize * d];
        l.router_bias = vec![0.0; c.n_experts as usize];
        let sh = expert(50, s_i, d);
        l.sh_gate = sh.gate.clone();
        l.sh_up = sh.up.clone();
        l.sh_down = sh.down.clone();

        let mut experts = HashMap::new();
        for e in 0..c.n_experts as usize {
            experts.insert((0, e), Arc::new(expert(e * 5, inter, d)));
        }
        let provider = MapProvider { experts };

        let x = vec![0.2f32, -0.1, 0.4, 0.3];
        let mut with = vec![0f32; d];
        let mut without = vec![0f32; d];
        moe(&c, &l, 0, &x, 1, &mut with, true, &provider).unwrap();
        moe(&c, &l, 0, &x, 1, &mut without, false, &provider).unwrap();

        let mut sh_out = vec![0f32; d];
        ffn(&sh.gate, &sh.up, &sh.down, &x, 1, &mut sh_out);
        for dd in 0..d {
            assert!((with[dd] - without[dd] - sh_out[dd]).abs() < 1e-5);
        }
    }

    #[test]
    fn routing_log_line_format() {
        // Two positions, k=2: one line per position, `step layer pos e0 e1`.
        let mut buf = Vec::new();
        write_routing_lines(&mut buf, 5, 3, 2, 2, &[10, 20, 30, 40]).unwrap();
        assert_eq!(String::from_utf8(buf).unwrap(), "5 3 0 10 20\n5 3 1 30 40\n");
    }

    #[test]
    fn provider_refuses_experts_owned_by_another_node() {
        // Ownership is enforced at the *load* layer, not only at dispatch. Asking for
        // a peer's expert must fail loudly — otherwise a routing bug quietly streams
        // it off this node's disk: right answer, wasted I/O, invisible bug.
        use std::io::Write;
        let dir = {
            static N: std::sync::atomic::AtomicU64 = std::sync::atomic::AtomicU64::new(0);
            let base = std::env::var("TMPDIR").unwrap_or_else(|_| "/tmp".into());
            let p = std::path::PathBuf::from(base).join(format!(
                "colibri-own-{}-{}",
                std::process::id(),
                N.fetch_add(1, std::sync::atomic::Ordering::Relaxed)
            ));
            std::fs::create_dir_all(&p).unwrap();
            p
        };
        // Minimal valid safetensors so `Shards::open` succeeds; the ownership gate
        // returns before any tensor is touched.
        let hdr = br#"{"dummy":{"dtype":"F32","shape":[1],"data_offsets":[0,4]}}"#;
        let mut f = std::fs::File::create(dir.join("m.safetensors")).unwrap();
        f.write_all(&(hdr.len() as u64).to_le_bytes()).unwrap();
        f.write_all(hdr).unwrap();
        f.write_all(&0f32.to_le_bytes()).unwrap();
        drop(f);

        let shards = Shards::open(&dir).unwrap();
        let c = cfg(); // 4 routed experts
        // 2 nodes over 4 experts: node 0 owns {0,1}, node 1 owns {2,3}.
        let sharding = ExpertSharding::new(2, c.n_experts as u32);
        let p = ShardsExpertProvider::with_sharding(&shards, &c, 4, sharding, NodeId(0));

        for peer_expert in [2usize, 3] {
            let e = p.expert(0, peer_expert).unwrap_err();
            assert_eq!(
                e.kind(),
                io::ErrorKind::Unsupported,
                "expert {peer_expert} belongs to node 1 and must be refused"
            );
            assert!(e.to_string().contains("owned by another node"), "unhelpful: {e}");
        }

        // A locally-owned expert gets *past* the gate and fails for an unrelated
        // reason (this fixture has no expert data) — proving the gate discriminates
        // by ownership rather than rejecting everything.
        let local = p.expert(0, 0).unwrap_err();
        assert_ne!(local.kind(), io::ErrorKind::Unsupported, "local expert must pass the gate");

        // A single-node provider owns everything: no expert is ever refused.
        let solo = ShardsExpertProvider::new(&shards, &c, 4);
        for e in 0..c.n_experts as usize {
            assert_ne!(solo.expert(0, e).unwrap_err().kind(), io::ErrorKind::Unsupported);
        }
        std::fs::remove_dir_all(&dir).ok();
    }

    #[test]
    fn streamed_experts_are_gpu_eligible_only_with_zerocopy() {
        // The safety property: streamed experts live in recycled pool buffers, and the
        // copy path's device cache is keyed by their address, so caching them there
        // would compute a *different* expert's weights. COLI_GPU_EXPERTS=1 must not be
        // able to force that — it may only opt in when zero-copy is available.
        assert!(experts_gpu_decision(Some("1"), true), "=1 opts in when zero-copy is available");
        assert!(
            !experts_gpu_decision(Some("1"), false),
            "=1 must NOT force streamed experts onto the address-keyed copy path"
        );

        // =0 always opts out.
        assert!(!experts_gpu_decision(Some("0"), true));
        assert!(!experts_gpu_decision(Some("0"), false));

        // Unset follows zero-copy availability.
        assert!(experts_gpu_decision(None, true));
        assert!(!experts_gpu_decision(None, false));

        // Unrecognised values behave like unset, never like a force.
        assert!(!experts_gpu_decision(Some("yes"), false));
        assert!(!experts_gpu_decision(Some(""), false));
    }

    #[test]
    fn moe_sharded_two_nodes_equals_single_node() {
        // The expert-parallel path must reproduce the single-node result exactly:
        // node 0 owns experts {0,1}, node 1 owns {2,3}; node 1's experts are served
        // over a real TCP loopback whose handler runs `compute_experts_partial`. With
        // topk=2 the token routes to one expert per node, exercising both the local
        // and the remote (transport) branch.
        use colibri_cluster::{serve_experts, ExpertResponse, TcpTransport};

        let c = cfg(); // 4 experts, topk 2, hidden 4
        let d = c.hidden as usize;
        let inter = c.moe_inter as usize;

        // Router rows are per-expert constants, so logit_e ∝ const_e: order 2>1>3>0,
        // top-2 = {2 (node 1), 1 (node 0)}.
        let consts = [-1.0f32, 0.5, 1.0, 0.0];
        let mut router = vec![0f32; c.n_experts as usize * d];
        for (e, &cst) in consts.iter().enumerate() {
            for i in 0..d {
                router[e * d + i] = cst;
            }
        }
        let mut l = Layer::default();
        l.router = router;
        l.router_bias = vec![0.0; c.n_experts as usize];
        let sh = expert(50, (c.moe_inter * c.n_shared) as usize, d);
        l.sh_gate = sh.gate.clone();
        l.sh_up = sh.up.clone();
        l.sh_down = sh.down.clone();

        // All four experts live in one provider (both "nodes" share it here).
        let experts: HashMap<(usize, usize), Arc<Expert>> =
            (0..4).map(|e| ((0usize, e), Arc::new(expert(e * 10, inter, d)))).collect();
        let provider = Arc::new(MapProvider { experts });

        let x = vec![0.3f32, 0.5, -0.2, 0.7];

        // Reference: single-node moe (all local), with the shared expert.
        let mut out_single = vec![0f32; d];
        moe(&c, &l, 0, &x, 1, &mut out_single, true, &*provider).unwrap();

        // Both sides share one map, so the connect-time handshake agrees.
        let sharding = ExpertSharding::new(2, c.n_experts as u32);

        // Node 1's expert server (loopback TCP), handler = compute_experts_partial.
        let hp = provider.clone();
        let addr = serve_experts("127.0.0.1:0".parse().unwrap(), sharding.fingerprint(), move |req| {
            let outputs = compute_experts_partial(
                &*hp,
                req.layer as usize,
                &req.experts,
                &req.weights,
                &req.activations,
                req.n_tokens,
                req.hidden,
            )
            .unwrap();
            ExpertResponse { outputs, n_tokens: req.n_tokens, hidden: req.hidden }
        })
        .unwrap();

        let mut peers = HashMap::new();
        peers.insert(NodeId(1), addr);
        let transport = TcpTransport::new(NodeId(0), peers, sharding.fingerprint());

        let mut out_sharded = vec![0f32; d];
        moe_sharded(&c, &l, 0, &x, 1, &mut out_sharded, true, &*provider, &sharding, &transport)
            .unwrap();

        for dd in 0..d {
            assert!(
                (out_single[dd] - out_sharded[dd]).abs() < 1e-5,
                "mismatch at {dd}: single {} vs sharded {}",
                out_single[dd],
                out_sharded[dd]
            );
        }
    }

    #[test]
    fn moe_sharded_hot_aware_map_equals_single_node() {
        // A hot-aware (traffic-balanced) map is only a *different* expert->node
        // assignment; the math must be unchanged. Weights [100,100,1,1] make LPT place
        // e0,e2 on node 0 and e1,e3 on node 1 — the opposite of the contiguous split
        // for the routed pair {2,1}, so the local and remote branches swap sides.
        // The output must still match single-node exactly.
        use colibri_cluster::{serve_experts, ExpertResponse, TcpTransport};

        let c = cfg();
        let d = c.hidden as usize;
        let inter = c.moe_inter as usize;

        let consts = [-1.0f32, 0.5, 1.0, 0.0]; // top-2 routes to {2, 1}
        let mut router = vec![0f32; c.n_experts as usize * d];
        for (e, &cst) in consts.iter().enumerate() {
            for i in 0..d {
                router[e * d + i] = cst;
            }
        }
        let mut l = Layer::default();
        l.router = router;
        l.router_bias = vec![0.0; c.n_experts as usize];
        let sh = expert(50, (c.moe_inter * c.n_shared) as usize, d);
        l.sh_gate = sh.gate.clone();
        l.sh_up = sh.up.clone();
        l.sh_down = sh.down.clone();

        let experts: HashMap<(usize, usize), Arc<Expert>> =
            (0..4).map(|e| ((0usize, e), Arc::new(expert(e * 10, inter, d)))).collect();
        let provider = Arc::new(MapProvider { experts });
        let x = vec![0.3f32, 0.5, -0.2, 0.7];

        let mut out_single = vec![0f32; d];
        moe(&c, &l, 0, &x, 1, &mut out_single, true, &*provider).unwrap();

        let weights = [100u64, 100, 1, 1];
        let sharding = ExpertSharding::balanced(2, c.n_experts as u32, &weights);
        assert!(sharding.is_hot_aware());

        let hp = provider.clone();
        let addr = serve_experts("127.0.0.1:0".parse().unwrap(), sharding.fingerprint(), move |req| {
            let outputs = compute_experts_partial(
                &*hp,
                req.layer as usize,
                &req.experts,
                &req.weights,
                &req.activations,
                req.n_tokens,
                req.hidden,
            )
            .unwrap();
            ExpertResponse { outputs, n_tokens: req.n_tokens, hidden: req.hidden }
        })
        .unwrap();

        let mut peers = HashMap::new();
        peers.insert(NodeId(1), addr);
        let transport = TcpTransport::new(NodeId(0), peers, sharding.fingerprint());
        // The hot pair is split across nodes, unlike the contiguous map.
        assert_ne!(sharding.owner(0), sharding.owner(1), "hot experts must be spread");
        let contig = ExpertSharding::new(2, c.n_experts as u32);
        assert_ne!(
            sharding.fingerprint(),
            contig.fingerprint(),
            "test needs a map that differs from contiguous"
        );

        let mut out_sharded = vec![0f32; d];
        moe_sharded(&c, &l, 0, &x, 1, &mut out_sharded, true, &*provider, &sharding, &transport)
            .unwrap();

        for dd in 0..d {
            assert!(
                (out_single[dd] - out_sharded[dd]).abs() < 1e-5,
                "hot-aware mismatch at {dd}: single {} vs sharded {}",
                out_single[dd],
                out_sharded[dd]
            );
        }
    }

    /// End-to-end: write a real int4 `.weight` + f32 `.qs` shard for one expert,
    /// load it through the coalesced + chunked path (`read_threads=8`), and assert
    /// the resulting `Bytes::Shared` views (a) hold exactly the on-disk bytes and
    /// (b) dequant identically to an owned byte-for-byte reference via `matmul_qt`.
    /// Dims chosen so the 2.25 MiB weight span splits into 2 chunks whose boundary
    /// lands *inside* the gate tensor, and the disk order (down|gate|up) differs from
    /// the request order (gate,up,down).
    #[test]
    fn load_expert_roundtrip_chunked_shared_dequant() {
        use std::fs::File;
        use std::io::Write;
        use std::path::{Path, PathBuf};
        use std::sync::atomic::{AtomicU64, Ordering};

        fn temp_dir() -> PathBuf {
            static N: AtomicU64 = AtomicU64::new(0);
            let base = std::env::var("TMPDIR").unwrap_or_else(|_| "/tmp".into());
            let mut p = PathBuf::from(base);
            p.push(format!(
                "colibri-loadexpert-{}-{}",
                std::process::id(),
                N.fetch_add(1, Ordering::Relaxed)
            ));
            std::fs::create_dir_all(&p).unwrap();
            p
        }

        // Write U8/F32 tensors laid out contiguously in the given order.
        fn write_tensors(dir: &Path, entries: &[(&str, &str, Vec<u8>)]) -> PathBuf {
            let mut hjson = String::from("{");
            let mut off = 0usize;
            for (i, (name, dtype, b)) in entries.iter().enumerate() {
                if i > 0 {
                    hjson.push(',');
                }
                let numel = if *dtype == "F32" { b.len() / 4 } else { b.len() };
                hjson.push_str(&format!(
                    "\"{}\":{{\"dtype\":\"{}\",\"shape\":[{}],\"data_offsets\":[{},{}]}}",
                    name,
                    dtype,
                    numel,
                    off,
                    off + b.len()
                ));
                off += b.len();
            }
            hjson.push('}');
            let hbytes = hjson.as_bytes();
            let path = dir.join("model.safetensors");
            let mut f = File::create(&path).unwrap();
            f.write_all(&(hbytes.len() as u64).to_le_bytes()).unwrap();
            f.write_all(hbytes).unwrap();
            for (_, _, b) in entries {
                f.write_all(b).unwrap();
            }
            path
        }
        let f32_bytes = |v: &[f32]| -> Vec<u8> { v.iter().flat_map(|x| x.to_le_bytes()).collect() };

        let hidden = 768usize;
        let moe_inter = 2048usize;
        let rb_gu = hidden.div_ceil(2); // int4 row bytes for gate/up [moe_inter, hidden]
        let rb_d = moe_inter.div_ceil(2); // for down [hidden, moe_inter]

        // Distinct byte + scale patterns per tensor so a wrong offset/length shows up.
        let gate_q4: Vec<u8> = (0..moe_inter * rb_gu).map(|k| (k * 7 + 1) as u8).collect();
        let up_q4: Vec<u8> = (0..moe_inter * rb_gu).map(|k| (k * 5 + 2) as u8).collect();
        let down_q4: Vec<u8> = (0..hidden * rb_d).map(|k| (k * 3 + 9) as u8).collect();
        let gate_s: Vec<f32> = (0..moe_inter).map(|o| 0.011 + 0.001 * (o % 13) as f32).collect();
        let up_s: Vec<f32> = (0..moe_inter).map(|o| 0.007 + 0.002 * (o % 11) as f32).collect();
        let down_s: Vec<f32> = (0..hidden).map(|o| 0.013 + 0.001 * (o % 7) as f32).collect();

        let dir = temp_dir();
        let p = |suf: &str| format!("model.layers.0.mlp.experts.0.{suf}");
        let (gw, uw, dw) = (p("gate_proj.weight"), p("up_proj.weight"), p("down_proj.weight"));
        let (gs, us, ds) = (
            p("gate_proj.weight.qs"),
            p("up_proj.weight.qs"),
            p("down_proj.weight.qs"),
        );
        // Physical order down|gate|up (weights contiguous → one coalesced read),
        // then the scales — mirrors the real model, where request order != disk order.
        write_tensors(
            &dir,
            &[
                (&dw, "U8", down_q4.clone()),
                (&gw, "U8", gate_q4.clone()),
                (&uw, "U8", up_q4.clone()),
                (&gs, "F32", f32_bytes(&gate_s)),
                (&us, "F32", f32_bytes(&up_s)),
                (&ds, "F32", f32_bytes(&down_s)),
            ],
        );

        let shards = Shards::open(&dir).unwrap();
        let ex = load_expert(&shards, hidden, moe_inter, 4, 0, 0, 8).unwrap();

        // (a) each Bytes::Shared view holds exactly its on-disk bytes + scales + dims.
        assert!(ex.gate.q4.as_slice() == gate_q4.as_slice(), "gate q4 mismatch");
        assert!(ex.up.q4.as_slice() == up_q4.as_slice(), "up q4 mismatch");
        assert!(ex.down.q4.as_slice() == down_q4.as_slice(), "down q4 mismatch");
        assert_eq!(ex.gate.s, gate_s);
        assert_eq!(ex.up.s, up_s);
        assert_eq!(ex.down.s, down_s);
        assert_eq!((ex.gate.fmt_code, ex.gate.o, ex.gate.i), (2, moe_inter as i32, hidden as i32));
        assert_eq!((ex.down.fmt_code, ex.down.o, ex.down.i), (2, hidden as i32, moe_inter as i32));

        // (b) the shared views dequant identically to an owned reference through the
        // real matmul kernel (proves the QTensor is usable, not just byte-equal).
        let check = |loaded: &QTensor, q4: &[u8], s: &[f32], o: usize, i: usize| {
            let reference = QTensor {
                fmt_code: 2,
                q4: Bytes::Owned(q4.to_vec()),
                s: s.to_vec(),
                o: o as i32,
                i: i as i32,
                ..Default::default()
            };
            let x: Vec<f32> = (0..i).map(|k| 0.5 - 0.001 * (k % 17) as f32).collect();
            let mut y_loaded = vec![0f32; o];
            let mut y_ref = vec![0f32; o];
            matmul_qt(&mut y_loaded, &x, loaded, 1);
            matmul_qt(&mut y_ref, &x, &reference, 1);
            // Not assert_eq!. Under `--features cuda` these two deliberately take
            // different kernels: `load_expert` marks the expert gpu_eligible, so
            // `loaded` runs on the GPU, while `reference` gets gpu_eligible=false from
            // Default and stays on the CPU. They accumulate in different orders and
            // land ~1e-7 apart in relative terms — f32 epsilon, not a math error.
            // Demanding bit-identity made `cargo test --features cuda` fail on the
            // only platform that ships CUDA, which hid every other CUDA regression
            // behind a permanently red suite.
            //
            // 1e-5 still catches what this test is for: a mis-decoded int4 nibble or a
            // dropped offset-binary bias moves a value by ~8*scale, i.e. orders of
            // magnitude, not epsilons.
            for (k, (&a, &b)) in y_loaded.iter().zip(&y_ref).enumerate() {
                let tol = 1e-5 * a.abs().max(b.abs()).max(1.0);
                assert!(
                    (a - b).abs() <= tol,
                    "row {k}: loaded {a} vs reference {b} (diff {}, tol {tol})",
                    (a - b).abs()
                );
            }
        };
        check(&ex.gate, &gate_q4, &gate_s, moe_inter, hidden);
        check(&ex.up, &up_q4, &up_s, moe_inter, hidden);
        check(&ex.down, &down_q4, &down_s, hidden, moe_inter);

        std::fs::remove_dir_all(&dir).ok();
    }
}
