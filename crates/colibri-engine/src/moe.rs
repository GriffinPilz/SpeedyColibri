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
use colibri_core::{Arch, Bytes, Config, QTensor};
use colibri_safetensors::Shards;
use std::io;
use std::sync::{Arc, OnceLock};

/// The model's SwiGLU variant, recorded once at load by [`set_activation`].
/// GLM uses plain SiLU-gated SwiGLU (the default); MiniMax-M3 uses the clamped
/// OpenAI-SwiGLU. Held as a process global so the FFN choke point ([`ffn`]) can
/// read it without threading `cfg` through the expert-parallel boundary.
#[derive(Clone, Copy)]
struct ActCfg {
    oai: bool,
    alpha: f32,
    limit: f32,
    /// Gateless ReLU² (Nemotron-H `mlp_hidden_act == "relu2"`): the FFN is
    /// `down(relu(up·x)²)` with **no gate projection**. Overrides `oai` when set.
    relu2: bool,
}

static ACTIVATION: OnceLock<ActCfg> = OnceLock::new();

/// Record the model's SwiGLU variant for the FFN path. Call once after building
/// the [`Config`] (first value wins — one model per process).
pub fn set_activation(cfg: &Config) {
    let _ = ACTIVATION.set(ActCfg {
        oai: cfg.swiglu_oai,
        alpha: cfg.swiglu_alpha,
        limit: cfg.swiglu_limit,
        relu2: cfg.relu2,
    });
    // Mirror the choice into the CUDA backend so the fused FFN kernels apply the
    // same SwiGLU variant (host-side globals; safe to set before device init).
    #[cfg(feature = "cuda")]
    crate::gpu::set_activation(cfg.swiglu_oai, cfg.swiglu_alpha, cfg.swiglu_limit);
}

/// The active SwiGLU variant (defaults to SiLU when unset — the GLM path and
/// unit tests that never call [`set_activation`]).
fn activation() -> ActCfg {
    *ACTIVATION.get().unwrap_or(&ActCfg { oai: false, alpha: 0.0, limit: 0.0, relu2: false })
}

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

/// One routed expert's FFN weights.
///
/// 3-tensor SwiGLU experts (GLM/MiniMax) populate all three projections; Nemotron-H's
/// **gateless** ReLU² experts ship only `up`/`down` and leave `gate` at its empty
/// default (the ReLU² FFN ignores it). Kept as a plain `QTensor` rather than an
/// `Option` so both shapes share one struct and `bytes()`/`mark_gpu_eligible()` treat
/// an empty gate as zero-cost.
#[derive(Debug, Clone, Default)]
pub struct Expert {
    /// gate_proj `[moe_inter, hidden]` — **empty** for gateless (Nemotron-H) experts.
    pub gate: QTensor,
    /// up_proj `[moe_inter, hidden]` (or `[moe_inter, moe_latent]` for Nemotron-H).
    pub up: QTensor,
    /// down_proj `[hidden, moe_inter]` (or `[moe_latent, moe_inter]` for Nemotron-H).
    pub down: QTensor,
}

impl Expert {
    /// Resident byte size of this expert (sum of its tensors; an empty gate is 0).
    pub fn bytes(&self) -> u64 {
        (self.gate.bytes() + self.up.bytes() + self.down.bytes()) as u64
    }

    /// Mark this expert's tensors as GPU-cacheable (for preloaded/resident experts).
    /// A gateless expert's empty gate is marked too — harmless, it is never used.
    pub fn mark_gpu_eligible(&mut self) {
        self.gate.gpu_eligible = true;
        self.up.gpu_eligible = true;
        self.down.gpu_eligible = true;
    }
}

/// How a routed expert's weights are **named and shaped** in the container — the two
/// axes the streaming loader needs beyond the raw dims. GLM/MiniMax experts are
/// 3-tensor SwiGLU (`gate_proj`/`up_proj`/`down_proj`) under
/// `model.layers.N.mlp.experts.E.`; Nemotron-H experts are 2-tensor **gateless**
/// (`up_proj`/`down_proj`, ReLU²) under `model.layers.N.mixer.experts.E.` and run in
/// the low-rank `moe_latent` space. Both layouts are read as one coalesced span (the
/// projections are contiguous on disk) and detected by the same scale-sidecar probe, so
/// the loader shares a single code path across arches.
#[derive(Clone, Copy)]
pub struct ExpertLayout {
    /// Mixer-block segment of the container name: `"mlp"` (GLM/MiniMax) or `"mixer"`
    /// (Nemotron-H).
    prefix: &'static str,
    /// Gateless: 2-tensor `up`/`down`, no `gate_proj` (Nemotron-H ReLU²). The
    /// [`Expert::gate`] is left empty and ignored by the ReLU² FFN.
    gateless: bool,
}

impl ExpertLayout {
    /// The container layout for `arch`. Nemotron-H is gateless under `.mixer.experts.`;
    /// every other arch is 3-tensor SwiGLU under `.mlp.experts.`.
    pub fn for_arch(arch: Arch) -> ExpertLayout {
        match arch {
            Arch::NemotronH => ExpertLayout { prefix: "mixer", gateless: true },
            _ => ExpertLayout { prefix: "mlp", gateless: false },
        }
    }

    /// The ordered projection suffixes for one expert: `[gate,up,down]` (SwiGLU) or
    /// `[up,down]` (gateless). This is the group the coalesced read fetches, in disk order.
    fn projs(&self) -> &'static [&'static str] {
        if self.gateless {
            &["up_proj", "down_proj"]
        } else {
            &["gate_proj", "up_proj", "down_proj"]
        }
    }

    /// Container weight name for one expert projection (`suf` ∈ [`ExpertLayout::projs`]).
    fn weight_name(&self, layer: usize, eid: usize, suf: &str) -> String {
        format!("model.layers.{layer}.{}.experts.{eid}.{suf}.weight", self.prefix)
    }
}

/// The outer (input/output) dimension of a routed expert: the model `hidden` for the
/// 3-tensor SwiGLU arches, but the low-rank `moe_latent` bottleneck for Nemotron-H,
/// whose experts run entirely in latent space (`up: moe_latent→moe_inter`,
/// `down: moe_inter→moe_latent`).
fn expert_outer_dim(cfg: &Config) -> usize {
    if matches!(cfg.arch, Arch::NemotronH) {
        cfg.moe_latent as usize
    } else {
        cfg.hidden as usize
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
    /// Concurrent readers each expert's ~18 MB read is chunked across (a single
    /// stream tops out far below the NVMe, which needs queue depth ~10 to saturate).
    /// `COLI_LOAD_THREADS` overrides; see [`default_read_threads`] for why the
    /// default is 2× cores rather than the core count.
    read_threads: usize,
    /// Container name/shape convention for this arch's experts (3-tensor SwiGLU vs
    /// 2-tensor gateless). Derived once from `cfg.arch` at construction.
    layout: ExpertLayout,
}

/// Read-thread count for on-demand expert streaming: `COLI_LOAD_THREADS` else
/// **twice the core count**.
///
/// This is an *I/O concurrency* knob, not a compute-parallelism one — the threads
/// spend nearly all their time blocked in `pread`, so each contributes at most one
/// outstanding request. Sizing them to cores (what this used to do, by borrowing
/// [`crate::preload::default_num_files`] — which is right for *shard* counts and
/// wrong here) leaves the NVMe queue under-fed.
///
/// Measured on GB10, 1 node, prompt 512, ngen 12, tokens byte-identical at every
/// setting:
///
/// | threads | ms/token |
/// |---------|----------|
/// | 12      | 4768 |
/// | 20 (= cores, the old default) | 3409–3422 |
/// | 32      | 2941–3238 |
/// | 40      | 2842 |
/// | 48      | 2886 |
///
/// The curve is steep below the core count and flat from ~32 to ~48, where the
/// differences sit inside run-to-run drift. 2× cores is chosen as a principled
/// point in that flat region rather than over-fitting to the single fastest
/// sample. The clamp keeps tiny boxes above a useful queue depth and stops
/// many-core hosts spawning hundreds of blocked threads for one drive.
fn default_read_threads() -> usize {
    std::env::var("COLI_LOAD_THREADS")
        .ok()
        .and_then(|s| s.parse().ok())
        .unwrap_or_else(|| crate::preload::default_num_files().saturating_mul(2).clamp(8, 64))
}

impl<'a> ShardsExpertProvider<'a> {
    /// Single-node provider: all `n_experts` are local.
    pub fn new(shards: &'a Shards, cfg: &Config, ebits: u32) -> ShardsExpertProvider<'a> {
        ShardsExpertProvider {
            shards,
            hidden: expert_outer_dim(cfg),
            moe_inter: cfg.moe_inter as usize,
            ebits,
            sharding: ExpertSharding::single(cfg.n_experts as u32),
            this_node: NodeId(0),
            read_threads: default_read_threads(),
            layout: ExpertLayout::for_arch(cfg.arch),
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
            hidden: expert_outer_dim(cfg),
            moe_inter: cfg.moe_inter as usize,
            ebits,
            sharding,
            this_node,
            read_threads: default_read_threads(),
            layout: ExpertLayout::for_arch(cfg.arch),
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

/// Load one routed expert directly from the shards. Shared by `ShardsExpertProvider`
/// and the direct parallel preloader. `layout` selects the container convention:
/// 3-tensor SwiGLU (`gate/up/down` under `.mlp.experts.`) or 2-tensor gateless
/// (`up/down` under `.mixer.experts.`, Nemotron-H) — the gateless expert's `gate` is
/// left empty. `hidden` is the expert's **outer** dim (the model hidden, or `moe_latent`
/// for Nemotron's latent-space experts).
pub fn load_expert(
    shards: &Shards,
    layout: ExpertLayout,
    hidden: usize,
    moe_inter: usize,
    ebits: u32,
    layer: usize,
    eid: usize,
    read_threads: usize,
) -> io::Result<Expert> {
    let projs = layout.projs();
    let first = layout.weight_name(layer, eid, projs[0]);
    // Container marker is `.qs` (int/e4m3 per-row scales) OR `.g` (NVFP4 global scale);
    // NVFP4 experts drop `.qs` entirely, so both must count as "pre-quantized container".
    let mut ex = if shards.has(&format!("{first}.qs")) || shards.has(&format!("{first}.g")) {
        // Pre-quantized container: the projections are contiguous on disk (~18 MB for the
        // 3-tensor case), so read the whole group (2 gateless / 3 SwiGLU) in ONE coalesced
        // read into a shared buffer the tensors view — instead of a separate read +
        // allocation per projection (the streaming bottleneck). The read is chunked across
        // `read_threads` cores so a single miss saturates the disk. Scales are tiny and
        // elsewhere; keep them as small per-tensor reads.
        let names: Vec<String> = projs.iter().map(|s| layout.weight_name(layer, eid, s)).collect();
        let name_refs: Vec<&str> = names.iter().map(String::as_str).collect();
        let ws = shards.read_raw_shared(&name_refs, read_threads)?;
        expert_from_views(shards, layout, hidden, moe_inter, layer, eid, &ws)?
    } else {
        // Full-tensor (runtime-quantized) path — the tiny oracle model.
        let load = |suf: &str, o: usize, i: usize| {
            crate::loader::qt_load(shards, &layout.weight_name(layer, eid, suf), o, i, ebits)
        };
        if layout.gateless {
            Expert {
                gate: QTensor::default(),
                up: load("up_proj", moe_inter, hidden)?,
                down: load("down_proj", hidden, moe_inter)?,
            }
        } else {
            Expert {
                gate: load("gate_proj", moe_inter, hidden)?,
                up: load("up_proj", moe_inter, hidden)?,
                down: load("down_proj", hidden, moe_inter)?,
            }
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

/// `COLI_EXPERT_NVFP4_SIM=1` (requires `COLI_EXPERT_FP8=1`): at load, round-trip each
/// e4m3 expert weight through the NVFP4 grid (e2m1 + per-16 ue4m3 block scale +
/// per-tensor global) and re-encode to e4m3, so the existing tiled/GEMV fp8 kernel runs
/// at NVFP4 *quality* while keeping e4m3 *speed and bytes*. This measures NVFP4's true
/// end-to-end perplexity cost (`coli ppl`) BEFORE committing to the NVFP4 container +
/// dedicated FP4 kernel — the reconstruction-error probe (9.4% rel-RMS) does not predict
/// perplexity. Slower to load (per-expert re-quantize on the reader threads); off by
/// default. Tokens WILL differ from the e4m3 baseline — that divergence is the signal.
fn expert_nvfp4_sim_enabled() -> bool {
    static ON: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *ON.get_or_init(|| std::env::var("COLI_EXPERT_NVFP4_SIM").ok().as_deref() == Some("1"))
}

/// Round-trip e4m3 codes (`o`×`i`, per-row `row_scale`) through NVFP4 and re-encode to
/// e4m3, returning `(new_codes, new_row_scales)`. Decode e4m3→f32, apply the validated
/// [`crate::convert::quantize_nvfp4_sim`] (per-tensor global + per-16 ue4m3 blocks +
/// e2m1), then re-quantize per-row to e4m3 (absmax/448). The re-encode adds negligible
/// error — e4m3's 8 bits are far finer than the NVFP4 grid the values now sit on.
fn nvfp4_sim_e4m3(codes: &[u8], o: usize, i: usize, row_scale: &[f32]) -> (Vec<u8>, Vec<f32>) {
    let mut w = vec![0f32; o * i];
    for r in 0..o {
        let s = row_scale[r];
        for c in 0..i {
            w[r * i + c] = colibri_core::dtype::f8e4m3_to_f32(codes[r * i + c]) * s;
        }
    }
    let recon = crate::convert::quantize_nvfp4_sim(&w, o, i);
    let mut out = vec![0u8; o * i];
    let mut ns = vec![0f32; o];
    for r in 0..o {
        let row = &recon[r * i..(r + 1) * i];
        let amax = row.iter().fold(0f32, |m, &x| m.max(x.abs()));
        let s = if amax > 0.0 { amax / 448.0 } else { 1.0 };
        let inv = 1.0 / s;
        for c in 0..i {
            out[r * i + c] = crate::convert::float_to_e4m3(row[c] * inv);
        }
        ns[r] = s;
    }
    (out, ns)
}

/// Build an `Expert` from its raw weight views (in [`ExpertLayout::projs`] order — 3
/// `gate,up,down` for SwiGLU or 2 `up,down` for gateless — each as returned by
/// [`Shards::read_raw_shared`]/`read_raw_shared_batched`), reading the tiny per-weight
/// scales separately. Shared by the single-expert and batched loaders. A gateless
/// expert's `gate` is left empty.
fn expert_from_views(
    shards: &Shards,
    layout: ExpertLayout,
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
        // NVFP4 experts: the weight blob is `nibbles ++ ue4m3 block-scales`, read as ONE
        // coalesced buffer together with gate/up/down (a separate `.bs` read cost one
        // uncoalesced random-seek pread per expert — 15x slower decode). Recognized by
        // the `.g` (per-tensor global scale) sidecar. Both halves are zero-copy views
        // into the shared buffer. See convert::requant_experts_nvfp4.
        let base = sname.strip_suffix(".qs").unwrap_or(&sname);
        let g_name = format!("{base}.g");
        if shards.has(&g_name) {
            let (buf, off, _len) = w;
            let nib_bytes = o * i.div_ceil(2);
            let bs_bytes = o * i.div_ceil(16);
            let mut g = [0f32; 1];
            shards.read_f32(&g_name, &mut g)?;
            return Ok(QTensor {
                fmt_code: 5,
                o: o as i32,
                i: i as i32,
                q4: Bytes::Shared { buf: buf.clone(), off: *off, len: nib_bytes },
                bs: Bytes::Shared { buf: buf.clone(), off: *off + nib_bytes, len: bs_bytes },
                g: g[0],
                ..Default::default()
            });
        }
        let (buf, off, len) = w;
        // int8/e4m3 (`o*i` bytes, told apart by `fp8`) vs int2 (`o*ceil(i/4)`).
        // int4 experts are no longer produced.
        let fmt = if *len == o * i { 1 } else { 3 };
        let mut s = vec![0f32; o];
        shards.read_f32(&sname, &mut s)?;
        let mut t = QTensor { fmt_code: fmt, o: o as i32, i: i as i32, s, ..Default::default() };
        let fp8 = expert_fp8_enabled();
        if fmt == 1 {
            // int8 goes in q8 (signed) — a copy. Skipped under `fp8`, where `fmt == 1`
            // means an e4m3 container (1 B/weight, length-indistinguishable from int8):
            // the block below replaces `q8` with a zero-copy view, so materializing it
            // here is 37.7 MB of allocate-and-copy per expert that is discarded unused.
            // At 8 experts × 75 layers that was 22.6 GB of dead single-threaded copying
            // per decoded token, with the drive idle throughout.
            if !fp8 {
                t.q8 = buf[*off..*off + *len].iter().map(|&b| b as i8).collect();
            }
        } else {
            t.q4 = Bytes::Shared { buf: buf.clone(), off: *off, len: *len };
        }
        if fp8 {
            if fmt == 1 {
                // e4m3 snapshot (COLI_XFP8 container): the bytes are already e4m3 —
                // 1 B/weight, indistinguishable by length from int8. Use them directly,
                // no conversion. Routed experts are never genuinely int8.
                t.q8 = Vec::new();
                if expert_nvfp4_sim_enabled() {
                    // NVFP4-quality probe: round-trip through the NVFP4 grid, re-encode
                    // to e4m3. `t.s` currently holds the container's per-row scales.
                    let (nc, ns) = nvfp4_sim_e4m3(&buf[*off..*off + *len], o, i, &t.s);
                    t.s = ns;
                    t.q4 = Bytes::Owned(nc);
                } else {
                    t.q4 = Bytes::Shared { buf: buf.clone(), off: *off, len: *len };
                }
                t.fmt_code = 4;
            }
        }
        Ok(t)
    };
    // `mk` reads the scale sidecar by name; pass the `.qs` name — it is also the carrier
    // the NVFP4 (`.g`) branch strips back to the base weight name to find the global.
    let qs = |suf: &str| format!("{}.qs", layout.weight_name(layer, eid, suf));
    if layout.gateless {
        // 2-tensor gateless (Nemotron-H): views = [up, down], no gate_proj. Experts run
        // in the `moe_latent` space, so `hidden` here is `moe_latent`.
        Ok(Expert {
            gate: QTensor::default(),
            up: mk(moe_inter, hidden, &views[0], qs("up_proj"))?,
            down: mk(hidden, moe_inter, &views[1], qs("down_proj"))?,
        })
    } else {
        Ok(Expert {
            gate: mk(moe_inter, hidden, &views[0], qs("gate_proj"))?,
            up: mk(moe_inter, hidden, &views[1], qs("up_proj"))?,
            down: mk(hidden, moe_inter, &views[2], qs("down_proj"))?,
        })
    }
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
    layout: ExpertLayout,
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
    let projs = layout.projs();
    // The pooled path applies only to the pre-quantized container (contiguous
    // projections + sidecar scales). Detect via the first expert's first projection
    // scale sidecar: `.qs` (int/e4m3) or `.g` (NVFP4, which has no `.qs`).
    let first = layout.weight_name(layer, eids[0], projs[0]);
    if !shards.has(&format!("{first}.qs")) && !shards.has(&format!("{first}.g")) {
        return eids
            .iter()
            .map(|&e| load_expert(shards, layout, hidden, moe_inter, ebits, layer, e, read_threads))
            .collect();
    }
    // One projection-name group per expert (2 gateless / 3 SwiGLU); keep the owned
    // strings alive so the borrowed &str slices handed to the reader stay valid.
    let names: Vec<Vec<String>> = eids
        .iter()
        .map(|&eid| projs.iter().map(|s| layout.weight_name(layer, eid, s)).collect())
        .collect();
    let groups: Vec<Vec<&str>> =
        names.iter().map(|g| g.iter().map(String::as_str).collect()).collect();
    let group_refs: Vec<&[&str]> = groups.iter().map(|g| g.as_slice()).collect();
    let views = shards.read_raw_shared_batched(&group_refs, read_threads)?;

    let mut out = Vec::with_capacity(eids.len());
    for (gi, &eid) in eids.iter().enumerate() {
        let mut ex = expert_from_views(shards, layout, hidden, moe_inter, layer, eid, &views[gi])?;
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
            self.layout,
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
                self.layout,
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
    // Fused GPU expert pipeline (one host round-trip) for resident weights. The SwiGLU
    // kernels apply the model's variant (set via gpu::set_activation), so that path is
    // correct for both GLM (SiLU) and MiniMax-M3 (swigluoai). Nemotron-H's gateless ReLU²
    // expert (`down(relu(up·x)²)`, no gate) routes to the dedicated NVFP4 relu² kernel.
    // Either falls back to the CPU reference below on decline (GPU unavailable, wrong
    // format, or out-of-range).
    #[cfg(feature = "cuda")]
    {
        if activation().relu2 {
            if crate::gpu::try_expert_ffn_relu2(up, down, x, nr, out) {
                return;
            }
        } else if crate::gpu::try_expert_ffn(gate, up, down, x, nr, out) {
            return;
        }
    }
    ffn_cpu(gate, up, down, x, nr, out);
}

/// CPU FFN reference / fallback. Two shapes selected by [`activation`]:
///   * SwiGLU (GLM/MiniMax): `down(act(gate·x) ⊙ up·x)`, `act` = SiLU or clamped
///     OpenAI-SwiGLU.
///   * gateless ReLU² (Nemotron-H, `relu2`): `down(relu(up·x)²)` — the `gate` argument
///     is unused (Nemotron experts ship no gate projection).
fn ffn_cpu(gate: &QTensor, up: &QTensor, down: &QTensor, x: &[f32], nr: usize, out: &mut [f32]) {
    let a = activation();
    if a.relu2 {
        // Gateless ReLU²: one up-projection, square the ReLU, one down-projection.
        let inter = up.o as usize;
        let mut uu = vec![0f32; nr * inter];
        matmul_qt(&mut uu, x, up, nr);
        for u in uu.iter_mut() {
            let r = u.max(0.0);
            *u = r * r;
        }
        matmul_qt(out, &uu, down, nr);
        return;
    }
    let inter = gate.o as usize; // moe_inter (or shared intermediate)
    let mut gg = vec![0f32; nr * inter];
    let mut uu = vec![0f32; nr * inter];
    matmul_qt(&mut gg, x, gate, nr);
    matmul_qt(&mut uu, x, up, nr);
    for (g, &u) in gg.iter_mut().zip(uu.iter()) {
        *g = if a.oai {
            crate::math::swiglu_oai(*g, u, a.alpha, a.limit)
        } else {
            silu(*g) * u
        };
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
        // Gateless ReLU² (Nemotron-H) has its own grouped kernel — the fp8 one is SwiGLU
        // and would read a gate tensor these experts don't ship.
        let grouped = if activation().relu2 {
            crate::gpu::try_expert_group_relu2(&active, activations, d, &mut out)
        } else {
            crate::gpu::try_expert_group(&active, activations, d, &mut out)
        };
        if grouped {
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

    // Split the routed experts into the driver's own shard and each peer's shard.
    let mut local: Option<(Vec<u32>, Vec<f32>)> = None;
    let mut remotes: Vec<(u32, Vec<u32>, Vec<f32>)> = Vec::new();
    for (node, cols) in by_node {
        let experts: Vec<u32> = cols.iter().map(|&ui| uniq[ui] as u32).collect();
        let weights = subcols(&w_mat, s_len, n_uniq, &cols);
        if NodeId(node) == me {
            local = Some((experts, weights));
        } else {
            remotes.push((node, experts, weights));
        }
    }

    // Overlap the nodes. The serial loop above computed the local shard, THEN blocked
    // shipping activations to each peer and waiting for its reply — so the nodes took
    // turns (each idle while the other loaded + computed) and the expert-parallel split
    // bought almost nothing (measured: 2-node expert-load halved but total prefill flat,
    // the savings absorbed into peer-wait). Here every peer request flies concurrently
    // while the local shard computes, so wall time is max(nodes) not sum(nodes). Partials
    // are folded in ascending node order, so the f32 sum is bit-identical to the serial
    // path (`Transport: Send + Sync` makes the concurrent exchange sound).
    let mut partials: Vec<(u32, Vec<f32>)> = Vec::with_capacity(remotes.len() + 1);
    let mut err: Option<io::Error> = None;
    std::thread::scope(|scope| {
        let handles: Vec<_> = remotes
            .iter()
            .map(|(node, experts, weights)| {
                let node = *node;
                let h = scope.spawn(move || {
                    let req = ExpertRequest {
                        experts: experts.clone(),
                        weights: weights.clone(),
                        activations: x.to_vec(),
                        n_tokens: s_len,
                        hidden: d,
                        layer: layer as u32,
                    };
                    transport.exchange(NodeId(node), &req)
                });
                (node, h)
            })
            .collect();
        // Local shard computes while the peer requests are in flight.
        if let Some((experts, weights)) = &local {
            match compute_experts_partial(provider, layer, experts, weights, x, s_len, d) {
                Ok(p) => partials.push((me.0, p)),
                Err(e) => err = Some(e),
            }
        }
        for (node, h) in handles {
            match h.join() {
                Ok(Ok(resp)) if resp.outputs.len() == s_len * d => partials.push((node, resp.outputs)),
                Ok(Ok(resp)) => {
                    err.get_or_insert_with(|| {
                        io::Error::new(
                            io::ErrorKind::InvalidData,
                            format!("node {node}: expected {} outputs, got {}", s_len * d, resp.outputs.len()),
                        )
                    });
                }
                Ok(Err(e)) => {
                    err.get_or_insert_with(|| io::Error::new(io::ErrorKind::Other, e.to_string()));
                }
                Err(_) => {
                    err.get_or_insert_with(|| {
                        io::Error::new(io::ErrorKind::Other, format!("node {node}: exchange thread panicked"))
                    });
                }
            }
        }
    });
    if let Some(e) = err {
        return Err(e);
    }
    // Fold in ascending node order → identical accumulation order to the serial path.
    partials.sort_by_key(|(n, _)| *n);
    for (_, p) in &partials {
        for (o, v) in out.iter_mut().zip(p.iter()) {
            *o += *v;
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

/// Nemotron-H latent-space MoE over `x[S, hidden]` into `out[S, hidden]` (`x` is the
/// block-normed input). Port of `NemotronHMoE.forward`:
///
/// ```text
///   idx, w  = route(gate·x)                  # router runs on hidden (sigmoid+bias top-k)
///   h_lat   = fc1_latent · x                 # hidden -> moe_latent
///   moe_lat = Σ_k w_k · expert_k(h_lat)      # routed experts run in latent space (ReLU²)
///   out     = fc2_latent · moe_lat           # moe_latent -> hidden
///   out    += shared_expert(x)               # gateless ReLU² on the original hidden
/// ```
///
/// The routed experts are **gateless ReLU²** (`down(relu(up·x)²)`) and operate entirely
/// in the `moe_latent` space, so [`compute_experts_partial`] is reused with
/// `hidden := moe_latent` and the `relu2` activation (set once by [`set_activation`],
/// which makes [`ffn`] ignore the unused gate). The shared expert reuses `l.up_proj`/
/// `l.down_proj` (its `gate_proj` is empty and ignored under ReLU²).
///
/// The routed experts are latent-space **2-tensor** (up/down, no gate) NVFP4 weights.
/// [`ShardsExpertProvider`] streams them via the gateless [`ExpertLayout`]
/// (`.mixer.experts.` naming, 2-projection coalesced read, empty `gate`); the forward
/// math here is provider-agnostic — any provider returning [`Expert`]s whose `up`/`down`
/// are the latent projections (any `gate`) computes the correct result.
pub fn nemotron_moe<P: ExpertProvider>(
    cfg: &Config,
    l: &Layer,
    layer: usize,
    x: &[f32],
    s_len: usize,
    out: &mut [f32],
    provider: &P,
) -> io::Result<()> {
    let d = cfg.hidden as usize;
    let dl = cfg.moe_latent as usize;
    let e_n = cfg.n_experts as usize;
    let k = (cfg.topk as usize).min(e_n);

    // ---- router (f32, on the hidden state) + top-K per position -----------
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

    // ---- fc1: hidden -> latent --------------------------------------------
    let fc1 = l.fc1_latent.as_ref().expect("nemotron MoE layer missing fc1_latent");
    let fc2 = l.fc2_latent.as_ref().expect("nemotron MoE layer missing fc2_latent");
    let mut h_lat = vec![0f32; s_len * dl];
    matmul_qt(&mut h_lat, x, fc1, s_len);

    // ---- routed experts (weighted sum, in latent space) -------------------
    let (uniq, w_mat) = union_and_weights(&idxs, &ws, s_len, k, e_n);
    let uniq_u32: Vec<u32> = uniq.iter().map(|&e| e as u32).collect();
    let moe_lat = compute_experts_partial(provider, layer, &uniq_u32, &w_mat, &h_lat, s_len, dl)?;

    // ---- fc2: latent -> hidden --------------------------------------------
    matmul_qt(out, &moe_lat, fc2, s_len);

    // ---- shared expert (gateless ReLU², on the original hidden) -----------
    let _st = std::time::Instant::now();
    let mut sh = vec![0f32; s_len * d];
    // `l.gate_proj` is empty for a Nemotron MoE layer and ignored under ReLU²; the
    // shared expert is `l.up_proj`/`l.down_proj` (hidden -> shared_inter -> hidden).
    ffn(&l.gate_proj, &l.up_proj, &l.down_proj, x, s_len, &mut sh);
    for (o, &sv) in out.iter_mut().zip(sh.iter()) {
        *o += sv;
    }
    if crate::forward::profile_on() {
        crate::forward::SHARED_US
            .fetch_add(_st.elapsed().as_micros() as u64, std::sync::atomic::Ordering::Relaxed);
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::quantize::qtensor_from_f32;
    use std::collections::HashMap;

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

    // MiniMax-M3 (and GLM) routing: sigmoid scoring, additive selection bias,
    // top-k by (sigmoid + bias), weights = normalized raw sigmoids × routed_scale.
    #[test]
    fn route_sigmoid_bias_topk_normalized_scaled() {
        let json = colibri_json::Json::parse(
            r#"{"hidden_size":4,"num_hidden_layers":1,"num_attention_heads":1,
                "n_routed_experts":4,"num_experts_per_tok":2,"moe_intermediate_size":3,
                "intermediate_size":4,"first_k_dense_replace":0,"q_lora_rank":2,
                "kv_lora_rank":2,"qk_nope_head_dim":2,"qk_rope_head_dim":2,"v_head_dim":2,
                "n_shared_experts":1,"vocab_size":8,"n_group":1,"topk_group":1,
                "norm_topk_prob":true,"rms_norm_eps":1e-5,"routed_scaling_factor":2.0,
                "rope_parameters":{"rope_theta":10000.0},"eos_token_id":[7],
                "index_topk":0,"index_n_heads":0,"index_head_dim":0}"#,
        )
        .unwrap();
        let cfg = Config::from_json(&json).unwrap();
        // Expert 0 has the lowest logit but a large selection bias → it must be
        // chosen; expert 1 has the highest logit. 2 and 3 lose on sigmoid+bias.
        let logits = [0.0f32, 2.0, -1.0, 0.3];
        let bias = [5.0f32, 0.0, 0.0, 0.0];
        let (idx, w) = route(&cfg, &logits, &bias);
        assert_eq!(idx.len(), 2);
        assert!(idx.contains(&0) && idx.contains(&1), "chosen {idx:?}");
        // Weights are the *raw* sigmoid(logit) (bias affects selection only),
        // normalized over the chosen set, then scaled by routed_scaling_factor.
        let (s0, s1) = (crate::math::sigmoid(0.0), crate::math::sigmoid(2.0));
        let sum = s0 + s1;
        let wmap: std::collections::HashMap<usize, f32> =
            idx.iter().copied().zip(w.iter().copied()).collect();
        assert!((wmap[&0] - s0 / sum * 2.0).abs() < 1e-5);
        assert!((wmap[&1] - s1 / sum * 2.0).abs() < 1e-5);
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

    /// End-to-end: write a real int2 `.weight` + f32 `.qs` shard for one expert,
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

        // hidden=1536 (2x) so the int2 weight span stays 2.25 MiB (0.75 MiB/tensor),
        // matching the int4-era byte layout so the 2-chunk split still lands in gate.
        let hidden = 1536usize;
        let moe_inter = 2048usize;
        let rb_gu = hidden.div_ceil(4); // int2 row bytes for gate/up [moe_inter, hidden]
        let rb_d = moe_inter.div_ceil(4); // for down [hidden, moe_inter]

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
        let glm = ExpertLayout::for_arch(Arch::GlmMoeDsa);
        let ex = load_expert(&shards, glm, hidden, moe_inter, 4, 0, 0, 8).unwrap();

        // (a) each Bytes::Shared view holds exactly its on-disk bytes + scales + dims.
        assert!(ex.gate.q4.as_slice() == gate_q4.as_slice(), "gate q4 mismatch");
        assert!(ex.up.q4.as_slice() == up_q4.as_slice(), "up q4 mismatch");
        assert!(ex.down.q4.as_slice() == down_q4.as_slice(), "down q4 mismatch");
        assert_eq!(ex.gate.s, gate_s);
        assert_eq!(ex.up.s, up_s);
        assert_eq!(ex.down.s, down_s);
        assert_eq!((ex.gate.fmt_code, ex.gate.o, ex.gate.i), (3, moe_inter as i32, hidden as i32));
        assert_eq!((ex.down.fmt_code, ex.down.o, ex.down.i), (3, hidden as i32, moe_inter as i32));

        // (b) the shared views dequant identically to an owned reference through the
        // real matmul kernel (proves the QTensor is usable, not just byte-equal).
        let check = |loaded: &QTensor, q4: &[u8], s: &[f32], o: usize, i: usize| {
            let reference = QTensor {
                fmt_code: 3,
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
            // 1e-5 still catches what this test is for: a mis-decoded int2 field or a
            // dropped bias moves a value by ~2*scale, i.e. orders of magnitude, not
            // epsilons.
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

    /// Nemotron-H routed experts are 2-tensor **gateless** (up_proj + down_proj, no
    /// gate_proj), NVFP4, latent-space, and live under `.mixer.experts.`. Build one
    /// faithful to what `convert.rs` emits — U8 weight = packed e2m1 nibbles ++ ue4m3
    /// block-scales, plus an F32 `.g` global; codes block first then the floats, exactly
    /// `convert_snapshot`'s on-disk order — stream it through `ShardsExpertProvider` (both
    /// the single-expert and the pooled batched read), and assert the loaded `Expert`
    /// (empty gate; correct 2-tensor NVFP4 up/down in `moe_latent` space) computes
    /// `down(relu(up·x)²)` identically to an in-memory expert built from the same blobs.
    #[test]
    fn shards_provider_loads_gateless_nemotron_expert() {
        use std::fs::File;
        use std::io::Write;
        use std::path::PathBuf;
        use std::sync::atomic::{AtomicU64, Ordering};

        // Expert outer dim (moe_latent) = 16 = one NVFP4 block; moe_inter = 32 = two.
        const LATENT: usize = 16;
        const MOE_INTER: usize = 32;
        const NR: usize = 2; // tokens

        fn temp_dir() -> PathBuf {
            static N: AtomicU64 = AtomicU64::new(0);
            let base = std::env::var("TMPDIR").unwrap_or_else(|_| "/tmp".into());
            let p = PathBuf::from(base).join(format!(
                "colibri-nemo-gateless-{}-{}",
                std::process::id(),
                N.fetch_add(1, Ordering::Relaxed)
            ));
            std::fs::create_dir_all(&p).unwrap();
            p
        }

        // Deterministic smooth weights so the NVFP4 encoding is well-conditioned.
        let wv = |n: usize, seed: usize| -> Vec<f32> {
            (0..n).map(|k| ((k + seed) as f32 * 0.19).sin() * 0.5).collect()
        };
        let w_up = wv(MOE_INTER * LATENT, 3); // up:   [MOE_INTER, LATENT]
        let w_down = wv(LATENT * MOE_INTER, 7); // down: [LATENT, MOE_INTER]
        let (nib_u, bsc_u, g_u) = crate::convert::quantize_nvfp4(&w_up, MOE_INTER, LATENT);
        let (nib_d, bsc_d, g_d) = crate::convert::quantize_nvfp4(&w_down, LATENT, MOE_INTER);
        let mut blob_u = nib_u.clone();
        blob_u.extend_from_slice(&bsc_u); // weight = nibbles ++ block-scales
        let mut blob_d = nib_d.clone();
        blob_d.extend_from_slice(&bsc_d);

        // safetensors shard: the two U8 weight blobs contiguous (so the loader's coalesced
        // 2-tensor read fires), then the F32 `.g` globals — convert's codes-then-floats
        // order. NO gate_proj, NO `.qs`.
        let dir = temp_dir();
        let p = |suf: &str| format!("model.layers.0.mixer.experts.0.{suf}");
        let (uw, dw, ug, dg) =
            (p("up_proj.weight"), p("down_proj.weight"), p("up_proj.weight.g"), p("down_proj.weight.g"));
        let entries: [(&str, &str, Vec<u8>); 4] = [
            (uw.as_str(), "U8", blob_u.clone()),
            (dw.as_str(), "U8", blob_d.clone()),
            (ug.as_str(), "F32", g_u.to_le_bytes().to_vec()),
            (dg.as_str(), "F32", g_d.to_le_bytes().to_vec()),
        ];
        let mut hjson = String::from("{");
        let mut off = 0usize;
        for (i, (name, dtype, b)) in entries.iter().enumerate() {
            if i > 0 {
                hjson.push(',');
            }
            let numel = if *dtype == "F32" { b.len() / 4 } else { b.len() };
            hjson.push_str(&format!(
                "\"{name}\":{{\"dtype\":\"{dtype}\",\"shape\":[{numel}],\"data_offsets\":[{off},{}]}}",
                off + b.len()
            ));
            off += b.len();
        }
        hjson.push('}');
        let hbytes = hjson.as_bytes();
        let mut f = File::create(dir.join("model.safetensors")).unwrap();
        f.write_all(&(hbytes.len() as u64).to_le_bytes()).unwrap();
        f.write_all(hbytes).unwrap();
        for (_, _, b) in &entries {
            f.write_all(b).unwrap();
        }
        drop(f);

        // Tiny 1-layer Nemotron-H config: `arch` drives the gateless `.mixer.experts.`
        // layout and the `moe_latent` expert outer dim.
        let json = colibri_json::Json::parse(&format!(
            r#"{{"model_type":"nemotron_h","hidden_size":8,"num_hidden_layers":1,
                "num_attention_heads":2,"num_key_value_heads":1,"head_dim":2,"vocab_size":8,
                "hybrid_override_pattern":"E","n_routed_experts":4,"num_experts_per_tok":2,
                "moe_intermediate_size":{MOE_INTER},"moe_latent_size":{LATENT},
                "moe_shared_expert_intermediate_size":6,"norm_topk_prob":false,
                "routed_scaling_factor":1.0,"mlp_hidden_act":"relu2","ssm_state_size":2,
                "conv_kernel":2,"mamba_num_heads":2,"mamba_head_dim":2,"n_groups":1,
                "chunk_size":2,"layer_norm_epsilon":1e-5}}"#
        ))
        .unwrap();
        let cfg = Config::from_json(&json).unwrap();
        assert_eq!(cfg.arch, Arch::NemotronH);
        assert_eq!((cfg.moe_latent, cfg.moe_inter), (LATENT as i32, MOE_INTER as i32));

        let shards = Shards::open(&dir).unwrap();
        let provider = ShardsExpertProvider::new(&shards, &cfg, 8);
        // The provider must have picked up the gateless/latent layout from the arch.
        assert!(provider.layout.gateless, "Nemotron provider must use the gateless layout");
        assert_eq!(provider.hidden, LATENT, "expert outer dim must be moe_latent");

        // In-memory reference expert built straight from the same NVFP4 blobs.
        let ref_qt = |o: usize, i: usize, nib: &[u8], bsc: &[u8], g: f32| QTensor {
            fmt_code: 5,
            o: o as i32,
            i: i as i32,
            q4: Bytes::Owned(nib.to_vec()),
            bs: Bytes::Owned(bsc.to_vec()),
            g,
            ..Default::default()
        };
        let up_ref = ref_qt(MOE_INTER, LATENT, &nib_u, &bsc_u, g_u);
        let down_ref = ref_qt(LATENT, MOE_INTER, &nib_d, &bsc_d, g_d);

        // Gateless ReLU² FFN `down(relu(up·x)²)` over `[NR, LATENT]` (mirrors the `relu2`
        // branch of `ffn_cpu`), computed directly so this test never touches the
        // process-global activation the SwiGLU unit tests rely on.
        let relu2 = |up: &QTensor, down: &QTensor, x: &[f32]| -> Vec<f32> {
            let mut u = vec![0f32; NR * MOE_INTER];
            matmul_qt(&mut u, x, up, NR);
            for v in u.iter_mut() {
                let r = v.max(0.0);
                *v = r * r;
            }
            let mut y = vec![0f32; NR * LATENT];
            matmul_qt(&mut y, &u, down, NR);
            y
        };
        let x: Vec<f32> = (0..NR * LATENT).map(|k| 0.4 - 0.03 * (k % 9) as f32).collect();
        let want = relu2(&up_ref, &down_ref, &x);
        assert!(want.iter().any(|v| v.abs() > 1e-4), "reference relu2 output is all-zero");

        // Both provider entry points: the single-expert load and the pooled batched read
        // (a 2-tensor group through `read_raw_shared_batched`).
        for (label, ex) in [
            ("expert", provider.expert(0, 0).unwrap()),
            ("experts_batch", provider.experts_batch(0, &[0]).unwrap().remove(0)),
        ] {
            // Gateless: gate is empty; up/down are NVFP4 (fmt 5) with the latent-space dims.
            assert!(
                ex.gate.q4.is_empty() && ex.gate.o == 0 && ex.gate.i == 0,
                "{label}: gateless expert must have an empty gate"
            );
            assert_eq!(
                (ex.up.fmt_code, ex.up.o, ex.up.i),
                (5, MOE_INTER as i32, LATENT as i32),
                "{label}: up dims/format"
            );
            assert_eq!(
                (ex.down.fmt_code, ex.down.o, ex.down.i),
                (5, LATENT as i32, MOE_INTER as i32),
                "{label}: down dims/format"
            );
            // The coalesced `.g` read materialized exactly the on-disk nibble + block-scale
            // halves and the global — byte-for-byte, at the right offsets.
            assert_eq!(ex.up.q4.as_slice(), nib_u.as_slice(), "{label}: up nibbles");
            assert_eq!(ex.up.bs.as_slice(), bsc_u.as_slice(), "{label}: up block-scales");
            assert_eq!(ex.up.g, g_u, "{label}: up global");
            assert_eq!(ex.down.q4.as_slice(), nib_d.as_slice(), "{label}: down nibbles");
            assert_eq!(ex.down.bs.as_slice(), bsc_d.as_slice(), "{label}: down block-scales");
            assert_eq!(ex.down.g, g_d, "{label}: down global");

            // Computes `down(relu(up·x)²)` identically to the in-memory reference.
            let got = relu2(&ex.up, &ex.down, &x);
            for (k, (&a, &b)) in got.iter().zip(&want).enumerate() {
                let tol = 1e-5 * a.abs().max(b.abs()).max(1.0);
                assert!((a - b).abs() <= tol, "{label} row {k}: loaded {a} vs reference {b}");
            }
        }

        std::fs::remove_dir_all(&dir).ok();
    }
}
