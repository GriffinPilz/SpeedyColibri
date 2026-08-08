//! GPU matmul dispatch (feature `cuda`) — routes eligible `matmul_qt` calls to
//! the resident CUDA (Blackwell) backend.
//!
//! `coli_cuda_matmul` uploads a weight into a device slot on first use and reuses
//! it thereafter, so we keep a per-weight slot keyed by the weight's data
//! pointer. Only [`QTensor::gpu_eligible`] tensors (dense weights + preloaded
//! experts) are cached — their buffers live for the run, so the address key is
//! stable. Streaming experts (fresh buffers, reused addresses) stay on the CPU.
//!
//! The forward pass is single-threaded, so the slot registry is a `thread_local`
//! and needs no synchronization.

use colibri_backend::cuda::{self, ColiCudaTensor};
use colibri_core::tier::lfru_score;
use colibri_core::QTensor;
use std::cell::{OnceCell, RefCell};
use std::collections::HashMap;
use std::os::raw::c_void;

/// One GPU-resident FFN weight (expert / shared / dense) + LFRU bookkeeping.
struct GpuEntry {
    tensor: cuda::ResidentTensor, // frees the device slot on drop
    bytes: u64,
    heat: u32,
    last: u32,
}

/// Budget-bounded cache of GPU-resident FFN weights, keyed by CPU data pointer.
/// Evicts the coldest (LFRU) when over the VRAM budget so the full expert set
/// never exhausts device memory. Hot weights (shared expert, dense MLP — touched
/// every token) survive; cold routed experts are dropped and re-uploaded on use.
struct GpuFfnCache {
    entries: HashMap<usize, GpuEntry>,
    bytes: u64,
    budget: u64,
    clock: u32,
    evictions: u64,
}

impl GpuFfnCache {
    fn new() -> GpuFfnCache {
        GpuFfnCache {
            entries: HashMap::new(),
            bytes: 0,
            budget: ffn_budget(),
            clock: 0,
            evictions: 0,
        }
    }

    /// Evict coldest entries until resident bytes are at or under `budget`,
    /// never evicting a `protect`ed key (the tensors the current op still needs).
    /// If everything left is protected, it stops (holding the minimum working set
    /// even if that exceeds the nominal budget).
    ///
    /// This is the same O(entries)-per-victim shape that cost the *expert* cache 1839 ms
    /// on a profiled GLM run — worse here, in fact, since `protect.contains` linearly
    /// scans a slice inside the inner loop. It is left alone deliberately: this cache
    /// holds tens of entries (GLM 78 resident / 0 evictions, K3 1 / 143), not the expert
    /// cache's ~2051, so the product is negligible and changing it would be an unmeasured
    /// edit. Revisit **if this cache ever grows** — dropping the redundant device copy on
    /// GB10, or Nemotron-style preloading, would make it the same problem.
    /// See `State::evict_to_protecting` for the rank-once fix and its equivalence test.
    fn evict_to(&mut self, budget: u64, protect: &[usize]) {
        while self.bytes > budget {
            let clock = self.clock;
            let victim = self
                .entries
                .iter()
                .filter(|(k, _)| !protect.contains(k))
                .min_by_key(|(_, e)| lfru_score(e.heat, e.last, clock))
                .map(|(&k, _)| k);
            match victim {
                Some(k) => {
                    if let Some(e) = self.entries.remove(&k) {
                        self.bytes -= e.bytes; // ResidentTensor::drop frees the VRAM
                        self.evictions += 1;
                    }
                }
                None => break,
            }
        }
    }
}

/// GPU-resident expert VRAM budget: `COLI_VRAM_GB` if set, else free device
/// memory minus a reserve for the dense weights + working buffers.
fn ffn_budget() -> u64 {
    if let Ok(gb) = std::env::var("COLI_VRAM_GB") {
        if let Ok(g) = gb.parse::<u64>() {
            return g << 30;
        }
    }
    match cuda::mem_info(0) {
        Some((free, _total)) => (free as u64).saturating_sub(14u64 << 30), // ~dense+working reserve
        None => u64::MAX,
    }
}

/// CUDA availability, probed **once per process** — deliberately not thread-local.
///
/// `probe()` calls `coli_cuda_init`, which builds process-global state: `g_nctx`, the
/// `DeviceContext` array, and each device's stream. A thread-local `OnceCell` meant every
/// thread that first touched the GPU re-ran it, resetting a context other threads were
/// launching kernels on. That is the `invalid resource handle` the CUDA suite hit under
/// cargo's default parallelism — deterministic single-threaded, flaky otherwise, with the
/// victim test varying run to run. `coli_cuda_init` is now idempotent as well, so this is
/// belt and braces: the FFI is safe however it is called, and callers stop calling twice.
static AVAIL: std::sync::OnceLock<bool> = std::sync::OnceLock::new();

thread_local! {
    // Whether device 0 can read pageable host memory directly (coherent unified
    // memory). When true, FFN weights are wrapped (zero-copy) instead of copied.
    static PAGEABLE: OnceCell<bool> = const { OnceCell::new() };
    static RESIDENT: RefCell<HashMap<usize, *mut ColiCudaTensor>> =
        RefCell::new(HashMap::new());
    // budget-bounded GPU FFN cache (experts + shared + dense MLP), copy path
    static RESIDENT_FFN: RefCell<GpuFfnCache> = RefCell::new(GpuFfnCache::new());
    static GPU_MATMULS: std::cell::Cell<u64> = const { std::cell::Cell::new(0) };
    static GPU_FFN: std::cell::Cell<u64> = const { std::cell::Cell::new(0) };
    static GPU_ATTN: std::cell::Cell<u64> = const { std::cell::Cell::new(0) };
    // When set, routed NVFP4 relu² experts run per-row gemv even at S>1, so a multi-row
    // (collided-expert) call is bit-identical to S sequential decode calls. The MTP verify
    // forward sets it via `ExactExpertsGuard`; see forward.rs.
    static EXACT_EXPERTS: std::cell::Cell<bool> = const { std::cell::Cell::new(false) };
}

/// Whether routed NVFP4 relu² experts must run the bit-exact per-row gemv path (verify).
pub fn exact_experts() -> bool {
    EXACT_EXPERTS.with(|c| c.get())
}

/// RAII scope that forces (or restores) the bit-exact routed-expert path. Set around the
/// MTP verify forward so its S>1 expert logits match sequential decode to the bit.
pub struct ExactExpertsGuard(bool);
impl ExactExpertsGuard {
    pub fn new(on: bool) -> ExactExpertsGuard {
        let prev = EXACT_EXPERTS.with(|c| c.replace(on));
        ExactExpertsGuard(prev)
    }
}
impl Drop for ExactExpertsGuard {
    fn drop(&mut self) {
        EXACT_EXPERTS.with(|c| c.set(self.0));
    }
}

/// How resident dense weights reach the GPU.
///
/// `try_matmul_qt` used to always UPLOAD: `cudaMalloc` + `cudaMemcpy` into a device
/// buffer cached by host pointer. On GB10 "VRAM" is the same physical RAM as the host, so
/// that is a second copy of every resident weight rather than a move — and it is charged
/// against the same 121 GB the expert cache lives in.
///
/// The trade is real in both directions, so this is a choice rather than a default:
///   * **Upload** — the kernel reads device memory (~273 GB/s measured) but the model's
///     resident bytes are spent twice. Right when they fit.
///   * **ZeroCopy** — `coli_cuda_tensor_wrap` points the kernel at the host buffer
///     (~51 GB/s, the same path the streamed experts already take). Slower per access,
///     but costs nothing. Right when Upload would not fit — and vastly better than the
///     alternative of falling off the GPU entirely, which measured **6.8x** slower on
///     nemotron decode.
///
/// Kimi-K3 is the case that forces it: ~63 GB resident of a 121 GB box, so uploading
/// leaves no room for the expert cache and earlyoom kills the process mid-forward.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum WeightResidency {
    /// Device copy, cached by host pointer.
    Upload,
    /// Host buffer read in place — no device allocation.
    ZeroCopy,
}

thread_local! {
    static RESIDENCY: std::cell::Cell<WeightResidency> =
        const { std::cell::Cell::new(WeightResidency::Upload) };
}

/// Choose how resident weights reach the GPU. Set once at load, before any forward pass.
pub fn set_weight_residency(m: WeightResidency) {
    RESIDENCY.with(|c| c.set(m));
    if available() {
        cuda::set_weight_zerocopy(m == WeightResidency::ZeroCopy && zerocopy());
    }
}

/// The current mode; `ZeroCopy` degrades to `Upload` if the device cannot read pageable
/// host memory (nothing else in the tree could work either, but be explicit).
pub fn weight_residency() -> WeightResidency {
    let m = RESIDENCY.with(|c| c.get());
    if m == WeightResidency::ZeroCopy && !zerocopy() {
        return WeightResidency::Upload;
    }
    m
}

/// Whether the zero-copy wrap path is usable: a CUDA device is available and it
/// can read pageable host memory directly (probed once). `COLI_NO_ZEROCOPY=1`
/// forces the copy path for A/B comparison.
pub fn zerocopy() -> bool {
    if !available() {
        return false;
    }
    if std::env::var("COLI_NO_ZEROCOPY").ok().as_deref() == Some("1") {
        return false;
    }
    PAGEABLE.with(|c| *c.get_or_init(|| cuda::pageable_access(0)))
}

/// GPU FFN cache stats: `(resident_count, resident_bytes, evictions, budget)`.
pub fn ffn_cache_stats() -> (usize, u64, u64, u64) {
    RESIDENT_FFN.with(|r| {
        let c = r.borrow();
        (c.entries.len(), c.bytes, c.evictions, c.budget)
    })
}

/// Whether a CUDA device is usable (probed once per PROCESS; honors `COLI_CUDA=0`).
pub fn available() -> bool {
    *AVAIL.get_or_init(|| cuda::CudaBackend::probe().is_some())
}

/// Tell the CUDA backend which SwiGLU variant the FFN kernels should apply
/// (`oai` = clamped OpenAI-SwiGLU for MiniMax-M3, else SiLU). Set once at load.
pub fn set_activation(oai: bool, alpha: f32, limit: f32) {
    cuda::set_activation(oai, alpha, limit);
}

/// Standard GQA prefill attention on the GPU (MiniMax-M3 dense core). `ctx`/`q` are
/// `[S, H, D]`; `k`/`v` are the full causal cache `[T, Hkv, D]`. `mode` picks the
/// kernel: 0 = scalar (f32, reference), 1 = WMMA flash (fp16, ~faster). Returns false
/// (→ the CPU core) when CUDA is unavailable or the dims are outside the kernel's range.
///
/// `win > 0` restricts each query to the last `win` keys — Maple's sliding layers. Pass 0
/// for the unwindowed causal core every other model uses.
#[allow(clippy::too_many_arguments)]
pub fn try_gqa_attn(
    ctx: &mut [f32],
    q: &[f32],
    k: &[f32],
    v: &[f32],
    s: usize,
    h: usize,
    hkv: usize,
    d: usize,
    t: usize,
    scale: f32,
    mode: u32,
    win: usize,
) -> bool {
    if !available() {
        return false;
    }
    // SAFETY: the caller (attention_gqa) sizes ctx/q as [S,H,D] and k/v as [T,Hkv,D].
    unsafe {
        cuda::gqa_attn_raw(
            ctx.as_mut_ptr(),
            q.as_ptr(),
            k.as_ptr(),
            v.as_ptr(),
            s as i32,
            h as i32,
            hkv as i32,
            d as i32,
            t as i32,
            scale,
            mode as i32,
            win as i32,
        )
    }
}

/// Whether the Mamba2 selective-scan decode step runs on the GPU (default-on when CUDA
/// is available). `COLI_MAMBA_CPU=1` forces the CPU `selective_scan` — an A/B switch to
/// confirm the GPU kernel is token-identical with everything else unchanged.
pub fn mamba_scan_gpu_enabled() -> bool {
    static ON: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *ON.get_or_init(|| std::env::var("COLI_MAMBA_CPU").ok().as_deref() != Some("1"))
}

/// GPU Nemotron-H Mamba2 selective-scan for one decode token (`seq == 1`). Runs the
/// per-token recurrent update `ssm = ssm*dA + dt*B*x; y = Σ ssm*C + x*D` on the GPU,
/// parallelized over `(head, head_dim)` with each thread looping `d_state`. `state`
/// (`[n_heads*head_dim*d_state]`) is uploaded, updated in place on the device, and
/// downloaded back; `y` (`[n_heads*head_dim]`) is the scan output. `dt_h`/`da_h`
/// (`[n_heads]`) are the per-head step + decay precomputed on the host by
/// [`crate::mamba2::step_head_scalars`] (so softplus/exp match the CPU exactly); the
/// kernel does only fma-free `f32` multiply/adds, so the result is bit-identical to
/// the CPU [`crate::mamba2::selective_scan`]. Returns false — leaving `state`/`y`
/// untouched — when CUDA is unavailable, so the caller runs the CPU scan.
#[allow(clippy::too_many_arguments)]
pub fn try_mamba2_scan(
    state: &mut [f32],
    y: &mut [f32],
    hidden: &[f32],
    b: &[f32],
    c: &[f32],
    dt_h: &[f32],
    da_h: &[f32],
    d: &[f32],
    n_heads: usize,
    head_dim: usize,
    d_state: usize,
    n_groups: usize,
) -> bool {
    if !available() {
        return false;
    }
    // Guard the buffer sizes so the kernel's indexing stays in bounds.
    if state.len() != n_heads * head_dim * d_state
        || y.len() != n_heads * head_dim
        || hidden.len() != n_heads * head_dim
        || b.len() != n_groups * d_state
        || c.len() != n_groups * d_state
        || dt_h.len() != n_heads
        || da_h.len() != n_heads
        || d.len() != n_heads
        || n_groups == 0
        || n_heads % n_groups != 0
    {
        return false;
    }
    // SAFETY: all slice lengths are checked above to match the (nh, hd, ds, ng) the
    // kernel indexes with; state/y are host in/out, the rest read-only.
    unsafe {
        cuda::mamba2_scan_raw(
            state.as_mut_ptr(),
            y.as_mut_ptr(),
            hidden.as_ptr(),
            b.as_ptr(),
            c.as_ptr(),
            dt_h.as_ptr(),
            da_h.as_ptr(),
            d.as_ptr(),
            n_heads as i32,
            head_dim as i32,
            d_state as i32,
            n_groups as i32,
        )
    }
}

/// Whole-sequence (prefill, `S > 1`) twin of [`try_mamba2_scan`]. `hidden`/`y` are
/// `[seq, n_heads*head_dim]`, `b`/`c` `[seq, n_groups, d_state]`, `dt_h`/`da_h`
/// `[seq, n_heads]` (from [`crate::mamba2::seq_head_scalars`], so the transcendentals
/// stay Rust-side), `d` `[n_heads]`.
///
/// ⚠️ **Token-identical, not bit-identical** — unlike [`try_mamba2_scan`]. The per-element
/// state recurrence keeps the CPU's operand order exactly, but `y = Σ_nn ss*C` is
/// tree-reduced instead of summed in strict `nn` order (~1 ULP on a 128-term f32 sum).
/// The bit-exact formulation exposes only `n_heads*head_dim` threads and measured no
/// faster than the CPU scan on GB10; this one exposes `n_heads*head_dim*d_state`.
/// Deterministic across runs, so it is reproducible — just not byte-equal to the CPU.
/// **Gate correctness on token identity, not on comparing floats to `selective_scan`.**
///
/// Returns false — leaving `state`/`y` untouched — when the GPU path is unavailable,
/// the sizes disagree, or the backend declines (`d_state` must fit a CUDA block). The
/// caller then runs the CPU [`crate::mamba2::selective_scan`].
#[allow(clippy::too_many_arguments)]
pub fn try_mamba2_scan_seq(
    state: &mut [f32],
    y: &mut [f32],
    hidden: &[f32],
    b: &[f32],
    c: &[f32],
    dt_h: &[f32],
    da_h: &[f32],
    d: &[f32],
    n_heads: usize,
    head_dim: usize,
    d_state: usize,
    n_groups: usize,
    seq: usize,
    exact: bool,
) -> bool {
    if !available() || !mamba_scan_gpu_enabled() || seq == 0 {
        return false;
    }
    // Guard every buffer the kernel indexes; a mismatch means a caller bug, and reading
    // out of bounds on the device is far harder to diagnose than declining here.
    if state.len() != n_heads * head_dim * d_state
        || y.len() != seq * n_heads * head_dim
        || hidden.len() != seq * n_heads * head_dim
        || b.len() != seq * n_groups * d_state
        || c.len() != seq * n_groups * d_state
        || dt_h.len() != seq * n_heads
        || da_h.len() != seq * n_heads
        || d.len() != n_heads
        || n_groups == 0
        || n_heads % n_groups != 0
    {
        return false;
    }
    // SAFETY: all slice lengths are checked above to match the (seq, nh, hd, ds, ng) the
    // kernel indexes with; state/y are host in/out, the rest read-only.
    unsafe {
        cuda::mamba2_scan_seq_raw(
            state.as_mut_ptr(),
            y.as_mut_ptr(),
            hidden.as_ptr(),
            b.as_ptr(),
            c.as_ptr(),
            dt_h.as_ptr(),
            da_h.as_ptr(),
            d.as_ptr(),
            n_heads as i32,
            head_dim as i32,
            d_state as i32,
            n_groups as i32,
            seq as i32,
            exact,
        )
    }
}

/// How many matmuls actually ran on the GPU this thread (proof the path fired).
pub fn matmul_count() -> u64 {
    GPU_MATMULS.with(|c| c.get())
}

/// How many fused expert FFNs ran on the GPU this thread.
pub fn ffn_count() -> u64 {
    GPU_FFN.with(|c| c.get())
}

/// How many MLA attention cores ran on the GPU this thread.
pub fn attn_count() -> u64 {
    GPU_ATTN.with(|c| c.get())
}

/// Per-layer device-side shadow of the compressed KV cache, so decode uploads
/// only the new row per token instead of re-sending the whole cache. Mirrors the
/// C engine's `kv_dev_L`/`kv_dev_R` + `kv_dev_valid`.
pub struct DeviceKv {
    layers: Vec<DevLayer>,
    max_t: usize,
}

struct DevLayer {
    latent: *mut c_void, // [max_t * kv_lora] f32
    rope: *mut c_void,   // [max_t * qk_rope] f32
    valid: usize,        // rows already on device
}

impl DeviceKv {
    pub fn new(n_layers: usize, max_t: usize) -> DeviceKv {
        DeviceKv {
            layers: (0..n_layers)
                .map(|_| DevLayer {
                    latent: std::ptr::null_mut(),
                    rope: std::ptr::null_mut(),
                    valid: 0,
                })
                .collect(),
            max_t,
        }
    }

    /// Ensure device rows `[0, tk)` for `layer` match the host cache, uploading
    /// only what's missing. Returns device `(latent, rope)` base pointers.
    /// Rewrites at `pos_base < valid` invalidate the stale tail.
    #[allow(clippy::too_many_arguments)]
    pub fn sync(
        &mut self,
        layer: usize,
        host_latent: &[f32],
        host_rope: &[f32],
        kvl: usize,
        r: usize,
        pos_base: usize,
        tk: usize,
    ) -> Option<(*const f32, *const f32)> {
        let max_t = self.max_t;
        let l = &mut self.layers[layer];
        if l.latent.is_null() {
            l.latent = cuda::pipe_alloc(0, max_t * kvl * 4)?;
            l.rope = cuda::pipe_alloc(0, max_t * r * 4)?;
            l.valid = 0;
        }
        if pos_base < l.valid {
            l.valid = pos_base; // rewritten rows are stale
        }
        if tk > l.valid {
            let from = l.valid;
            let n = tk - from;
            // SAFETY: device buffers hold max_t rows; host slices cover [from, tk).
            let ok = unsafe {
                cuda::pipe_upload(
                    0,
                    (l.latent as *mut f32).add(from * kvl) as *mut c_void,
                    host_latent[from * kvl..tk * kvl].as_ptr() as *const c_void,
                    n * kvl * 4,
                ) && cuda::pipe_upload(
                    0,
                    (l.rope as *mut f32).add(from * r) as *mut c_void,
                    host_rope[from * r..tk * r].as_ptr() as *const c_void,
                    n * r * 4,
                )
            };
            if !ok {
                return None;
            }
            l.valid = tk;
        }
        Some((l.latent as *const f32, l.rope as *const f32))
    }
}

impl Drop for DeviceKv {
    fn drop(&mut self) {
        for l in &self.layers {
            if !l.latent.is_null() {
                unsafe {
                    cuda::pipe_free(0, l.latent);
                    cuda::pipe_free(0, l.rope);
                }
            }
        }
    }
}

/// Single-token (S=1) GPU attention reading the KV cache from device memory.
/// `latent_dev`/`rope_dev` come from [`DeviceKv::sync`].
#[allow(clippy::too_many_arguments)]
pub fn try_attention_absorb_kvdev(
    kv_b: &QTensor,
    ctx: &mut [f32],
    q: &[f32],
    latent_dev: *const f32,
    rope_dev: *const f32,
    h: usize,
    qk_nope: usize,
    qk_rope: usize,
    v_head: usize,
    kv_lora: usize,
    t: usize,
    scale: f32,
) -> bool {
    if !available() || !kv_b.gpu_eligible {
        return false;
    }
    let Some(handle) = upload_ffn(kv_b, &[]) else {
        return false;
    };
    // SAFETY: handle resident; latent/rope device pointers valid for [T,K]/[T,R];
    // ctx/q host sized [H*V]/[H*qh].
    let ok = unsafe {
        cuda::attention_absorb_kvdev_raw(
            handle,
            ctx.as_mut_ptr(),
            q.as_ptr(),
            latent_dev,
            rope_dev,
            h as i32,
            qk_nope as i32,
            qk_rope as i32,
            v_head as i32,
            kv_lora as i32,
            t as i32,
            scale,
        )
    };
    if ok {
        GPU_ATTN.with(|c| c.set(c.get() + 1));
    }
    ok
}

/// Try the MLA weight-absorption attention core on the GPU: `ctx[S, H*V]` from
/// the query and the compressed KV cache, using resident `kv_b`. Returns `true`
/// if it ran there. Equivalent to the CPU `absorb_core`.
#[allow(clippy::too_many_arguments)]
pub fn try_attention_absorb(
    kv_b: &QTensor,
    ctx: &mut [f32],
    q: &[f32],
    latent: &[f32],
    rope: &[f32],
    s: usize,
    h: usize,
    qk_nope: usize,
    qk_rope: usize,
    v_head: usize,
    kv_lora: usize,
    t: usize,
    scale: f32,
) -> bool {
    if !available() || !kv_b.gpu_eligible {
        return false;
    }
    let Some(handle) = upload_ffn(kv_b, &[]) else {
        return false;
    };
    // SAFETY: handle resident on device 0; ctx/q/latent/rope sized by the dims.
    let ok = unsafe {
        cuda::attention_absorb_batch_raw(
            handle,
            ctx.as_mut_ptr(),
            q.as_ptr(),
            latent.as_ptr(),
            rope.as_ptr(),
            s as i32,
            h as i32,
            qk_nope as i32,
            qk_rope as i32,
            v_head as i32,
            kv_lora as i32,
            t as i32,
            scale,
        )
    };
    if ok {
        GPU_ATTN.with(|c| c.set(c.get() + 1));
    }
    ok
}

/// DSA sparse attention on the GPU — the [`try_attention_absorb`] twin that attends
/// only to each query's indexer selection. `sel[q]` holds the query's chosen cache
/// positions (relative to the latent's first row; the DSA path runs at `st0 == 0`, so
/// these are the absolute positions). An empty `sel[q]` is the is_dense case. Falls
/// back (returns false) when the GPU is unavailable, so the caller uses the CPU
/// `reconstruct_core`.
#[allow(clippy::too_many_arguments)]
/// DSA lightning-indexer scores on the GPU — the indexer's hot loop
/// (`score[s][t] = wsc·Σ_h hw[h]·relu(rs·dot(qi[h], key[t]))`, ~25.8 GFLOP per FULL
/// layer on CPU). Fills `scores[nsp, t_len]`; row `si` (query `s0+si`) is valid for
/// `t < pos_base+s0+si+1`. The kernel accumulates each head's dot in ascending `i`
/// exactly like the CPU reference, so the scores — and therefore the top-k selection
/// — match. Returns false (caller keeps the CPU path) when unavailable or `nh > 32`.
#[allow(clippy::too_many_arguments)]
pub fn try_dsa_indexer_scores(
    scores: &mut [f32],
    qi: &[f32],
    hw: &[f32],
    keys: &[f32],
    nsp: usize,
    s0: usize,
    nh: usize,
    hd: usize,
    t_len: usize,
    pos_base: usize,
) -> bool {
    if !available() || nsp == 0 || nh == 0 || nh > 32 || hd == 0 || t_len == 0 {
        return false;
    }
    if qi.len() < nsp * nh * hd
        || hw.len() < nsp * nh
        || keys.len() < t_len * hd
        || scores.len() < nsp * t_len
    {
        return false;
    }
    // SAFETY: sizes checked above; device 0 is the engine's single GPU.
    unsafe {
        cuda::dsa_indexer_scores_raw(
            scores.as_mut_ptr(),
            qi.as_ptr(),
            hw.as_ptr(),
            keys.as_ptr(),
            nsp as i32,
            s0 as i32,
            nh as i32,
            hd as i32,
            t_len as i32,
            pos_base as i32,
            0,
        )
    }
}

/// DeepSeek-V4's sparse attention core on the GPU.
///
/// This path was measured at **48% of V4 decode** (144 of 300 ms/token) while running as a
/// scalar Rust loop, with `coli gen` reporting `0 attention cores` — V4 attention had never
/// touched the GPU at all. Same contract as `dsv4::attention_dsv4_sparse`: `-1` masks a
/// slot, duplicate indices accumulate twice on purpose, sink in the denominator only.
///
/// The kernel accumulates in f32 where the CPU path uses f64, so results differ in the last
/// bits and tokens can diverge on a near-tie. That is a different-but-valid numeric path,
/// not a regression — `COLI_DSV4_GPU_ATTN=0` selects the CPU one for an exact A/B.
#[allow(clippy::too_many_arguments)]
pub fn try_dsv4_sparse_attn(
    out: &mut [f32],
    q: &[f32],
    kv: &[f32],
    sink: &[f32],
    idxs: &[i32],
    s: usize,
    h: usize,
    hd: usize,
    topk: usize,
) -> bool {
    static ON: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    if !*ON.get_or_init(|| std::env::var("COLI_DSV4_GPU_ATTN").ok().as_deref() != Some("0")) {
        return false;
    }
    if !available() || s == 0 || h == 0 || hd == 0 || topk == 0 || hd % 32 != 0 {
        return false;
    }
    let rows = kv.len() / hd;
    if rows == 0
        || q.len() < s * h * hd
        || out.len() < s * h * hd
        || sink.len() < h
        || idxs.len() < s * topk
    {
        return false;
    }
    let scale = (hd as f32).powf(-0.5);
    // SAFETY: every length checked above; the kernel bails on shared-memory overflow.
    let ok = unsafe {
        cuda::dsv4_sparse_attn_raw(
            q.as_ptr(),
            kv.as_ptr(),
            sink.as_ptr(),
            idxs.as_ptr(),
            s as i32,
            h as i32,
            hd as i32,
            topk as i32,
            rows as i32,
            scale,
            out.as_mut_ptr(),
        )
    };
    if ok {
        GPU_ATTN.with(|c| c.set(c.get() + 1));
    }
    ok
}

/// `h0`/`hc` select the head slice `[h0, h0+hc)` to compute (tensor-parallel
/// attention); the full-attention call passes `(0, h)`. A partial slice writes only
/// its `ctx` head-columns and the kernel zeroes the rest, so summing the slices'
/// o-projections reconstructs full attention. `h` stays the full head count (the
/// `[s, h, ·]` stride of `q`/`ctx`).
#[allow(clippy::too_many_arguments)]
pub fn try_attention_absorb_sparse(
    kv_b: &QTensor,
    ctx: &mut [f32],
    q: &[f32],
    latent: &[f32],
    rope: &[f32],
    sel: &[Vec<u32>],
    index_topk: usize,
    h0: usize,
    hc: usize,
    s: usize,
    h: usize,
    qk_nope: usize,
    qk_rope: usize,
    v_head: usize,
    kv_lora: usize,
    t: usize,
    scale: f32,
) -> bool {
    if !available() || !kv_b.gpu_eligible || sel.len() != s || index_topk == 0 {
        return false;
    }
    if h0 + hc > h || hc == 0 {
        return false;
    }
    let Some(handle) = upload_ffn(kv_b, &[]) else {
        return false;
    };
    // Flatten into fixed-stride [s, maxsel] indices + per-query counts. A query with an
    // empty selection keeps count 0 → the kernel attends causally (is_dense).
    let maxsel = index_topk;
    let mut sel_idx = vec![0i32; s * maxsel];
    let mut sel_cnt = vec![0i32; s];
    for (qi, positions) in sel.iter().enumerate() {
        let n = positions.len().min(maxsel);
        sel_cnt[qi] = n as i32;
        for (j, &p) in positions.iter().take(maxsel).enumerate() {
            sel_idx[qi * maxsel + j] = p as i32;
        }
    }
    // SAFETY: handle resident; ctx/q/latent/rope sized by the dims; sel_idx has
    // s*maxsel ints and sel_cnt has s ints (allocated just above).
    let ok = unsafe {
        cuda::attention_absorb_sparse_raw(
            handle,
            ctx.as_mut_ptr(),
            q.as_ptr(),
            latent.as_ptr(),
            rope.as_ptr(),
            sel_idx.as_ptr(),
            sel_cnt.as_ptr(),
            maxsel as i32,
            h0 as i32,
            hc as i32,
            s as i32,
            h as i32,
            qk_nope as i32,
            qk_rope as i32,
            v_head as i32,
            kv_lora as i32,
            t as i32,
            scale,
        )
    };
    if ok {
        GPU_ATTN.with(|c| c.set(c.get() + 1));
    }
    ok
}

fn weight_ptr(w: &QTensor) -> *const c_void {
    match w.fmt_code {
        0 => w.qf.as_ptr() as *const c_void,
        1 => w.q8.as_ptr() as *const c_void,
        _ => w.q4.as_ptr() as *const c_void,
    }
}

/// Zero-copy wrap of `w`: a **fresh, owned** descriptor pointing at the live RAM
/// buffers (no device allocation, no cache). The caller holds it for the duration
/// of one kernel call and drops it after. Crucially *not* cached by pointer — an
/// expert can be evicted (its RAM freed) and its address reused by another expert,
/// so a pointer-keyed descriptor cache would hand the kernel stale memory. Wrapping
/// is ~free (a `calloc` + storing pointers), so per-call is fine.
///
/// # Safety
/// `weight_ptr(w)`/`w.s` must stay valid until the returned tensor is dropped —
/// true while the caller holds the `Arc<Expert>` across the kernel call. The wrapped
/// weights stay in their on-disk layout. Only valid when `zerocopy()`.
fn wrap_fresh(w: &QTensor) -> Option<cuda::ResidentTensor> {
    // NVFP4 carries three host buffers (nibbles + ue4m3 block scales + f32 global)
    // rather than weights + per-row scale; wrap all three zero-copy.
    if w.fmt_code == 5 {
        return unsafe {
            cuda::ResidentTensor::wrap_raw_nvfp4(
                w.q4.as_ptr() as *const c_void,
                w.bs.as_ptr() as *const c_void,
                w.g,
                w.i,
                w.o,
                0,
            )
        };
    }
    // MXFP4 (Kimi-K3): same three-buffer shape as NVFP4 — nibbles + block scales +
    // global — but the format code selects the per-32 stride and the E8M0 decode.
    if w.fmt_code == 6 {
        return unsafe {
            cuda::ResidentTensor::wrap_raw_mxfp4(
                w.q4.as_ptr() as *const c_void,
                w.bs.as_ptr() as *const c_void,
                w.g,
                w.i,
                w.o,
                0,
            )
        };
    }
    unsafe { cuda::ResidentTensor::wrap_raw(weight_ptr(w), w.s.as_ptr(), w.fmt_code, w.i, w.o, 0) }
}

/// Upload `w` to the GPU (once) and return its resident handle, caching by data
/// pointer under the VRAM budget (the copy path — device copy). `protect` lists the
/// current op's other tensor keys so eviction never
/// drops a tensor still needed this op. The zero-copy path uses [`wrap_fresh`]
/// instead; this is only reached when zero-copy is unavailable/disabled.
fn upload_ffn(w: &QTensor, protect: &[usize]) -> Option<*mut ColiCudaTensor> {
    let key = weight_ptr(w) as usize;
    RESIDENT_FFN.with(|r| {
        let mut c = r.borrow_mut();
        c.clock = c.clock.wrapping_add(1);
        let clock = c.clock;
        if let Some(e) = c.entries.get_mut(&key) {
            e.heat = e.heat.saturating_add(1);
            e.last = clock;
            return Some(e.tensor.as_raw());
        }
        // Miss: make room (estimate from the CPU size), protecting this op's other
        // tensors, then upload + insert.
        let budget = c.budget;
        c.evict_to(budget.saturating_sub(w.bytes() as u64), protect);
        // SAFETY: weight_ptr/scales point at the live QTensor buffers, sized by
        // the tensor's [O,I]/fmt.
        let rt = unsafe {
            cuda::ResidentTensor::upload_raw(weight_ptr(w), w.s.as_ptr(), w.fmt_code, w.i, w.o, 0)
        }?;
        let raw = rt.as_raw();
        let bytes = rt.bytes() as u64; // actual device bytes
        c.bytes += bytes;
        c.entries.insert(
            key,
            GpuEntry {
                tensor: rt,
                bytes,
                heat: 1,
                last: clock,
            },
        );
        Some(raw)
    })
}

/// Try the fused expert FFN `out = down(silu(gate·x) ⊙ up·x)` on the GPU (one
/// upload/download instead of three GEMMs). Returns `true` if it ran there.
pub fn try_expert_ffn(
    gate: &QTensor,
    up: &QTensor,
    down: &QTensor,
    x: &[f32],
    nr: usize,
    out: &mut [f32],
) -> bool {
    if !available() || !gate.gpu_eligible || !up.gpu_eligible || !down.gpu_eligible {
        return false;
    }
    if zerocopy() {
        // Fresh, owned descriptors held only for this call — see `wrap_fresh`. Safe
        // under cache eviction (no stale pointer-keyed descriptors).
        let (Some(g), Some(u), Some(d)) = (wrap_fresh(gate), wrap_fresh(up), wrap_fresh(down))
        else {
            return false;
        };
        // SAFETY: g/u/d live until end of scope, covering the synchronous kernel +
        // download in expert_mlp_raw; out/x sized [nr, O]/[nr, I] by ffn().
        let ok = unsafe {
            if gate.fmt_code == 5 {
                cuda::expert_mlp_nvfp4_raw(
                    g.as_raw(),
                    u.as_raw(),
                    d.as_raw(),
                    out.as_mut_ptr(),
                    x.as_ptr(),
                    nr as i32,
                )
            } else if gate.fmt_code == 4 {
                cuda::expert_mlp_fp8_raw(
                    g.as_raw(),
                    u.as_raw(),
                    d.as_raw(),
                    out.as_mut_ptr(),
                    x.as_ptr(),
                    nr as i32,
                )
            } else if gate.fmt_code == 1 && tile_i8_enabled() {
                cuda::expert_mlp_i8a16_raw(
                    g.as_raw(),
                    u.as_raw(),
                    d.as_raw(),
                    out.as_mut_ptr(),
                    x.as_ptr(),
                    nr as i32,
                )
            } else {
                cuda::expert_mlp_raw(
                    g.as_raw(),
                    u.as_raw(),
                    d.as_raw(),
                    out.as_mut_ptr(),
                    x.as_ptr(),
                    nr as i32,
                )
            }
        };
        if ok {
            GPU_FFN.with(|c| c.set(c.get() + 1));
        }
        return ok;
    }
    // NVFP4 has no device-copy path (the block-scale/global plumbing is zero-copy only);
    // fall back to the CPU decode (matmul_qt fmt=5) when zero-copy is unavailable.
    if gate.fmt_code == 5 {
        return false;
    }
    // Copy path: cached device uploads. all three must stay resident together for
    // the fused kernel — protect them from eviction.
    let keys = [
        weight_ptr(gate) as usize,
        weight_ptr(up) as usize,
        weight_ptr(down) as usize,
    ];
    let (Some(g), Some(u), Some(d)) = (
        upload_ffn(gate, &keys),
        upload_ffn(up, &keys),
        upload_ffn(down, &keys),
    ) else {
        return false;
    };
    // SAFETY: handles are resident on device 0; out/x sized [nr, O]/[nr, I] by ffn().
    let ok = unsafe {
        if gate.fmt_code == 4 {
            cuda::expert_mlp_fp8_raw(g, u, d, out.as_mut_ptr(), x.as_ptr(), nr as i32)
        } else if gate.fmt_code == 1 && tile_i8_enabled() {
            cuda::expert_mlp_i8a16_raw(g, u, d, out.as_mut_ptr(), x.as_ptr(), nr as i32)
        } else {
            cuda::expert_mlp_raw(g, u, d, out.as_mut_ptr(), x.as_ptr(), nr as i32)
        }
    };
    if ok {
        GPU_FFN.with(|c| c.set(c.get() + 1));
    }
    ok
}

/// MXFP4 expert FFN with the engine's configured SwiGLU — DeepSeek-V4's experts.
///
/// V4's experts are MXFP4 but gated SwiGLU, which fits neither existing path: the `_situ`
/// twin below computes K3's activation, and `coli_cuda_expert_mlp` reads BLOCK scales as
/// a per-row f32 array and took an illegal memory access on them. So every one of V4's
/// 258 routed experts per token ran the scalar CPU loop.
///
/// Zero-copy only, like the other nibble/block-scale paths.
pub fn try_expert_ffn_mxfp4(
    gate: &QTensor,
    up: &QTensor,
    down: &QTensor,
    x: &[f32],
    nr: usize,
    out: &mut [f32],
) -> bool {
    if !available() || !zerocopy() {
        return false;
    }
    if gate.fmt_code != 6 || up.fmt_code != 6 || down.fmt_code != 6 {
        return false;
    }
    let (Some(g), Some(u), Some(d)) = (wrap_fresh(gate), wrap_fresh(up), wrap_fresh(down)) else {
        return false;
    };
    // SAFETY: g/u/d live to end of scope, covering the synchronous kernel + download;
    // x/out are sized [nr, gate.i] / [nr, down.o] by `ffn`.
    let ok = unsafe {
        cuda::expert_mlp_mxfp4_raw(
            g.as_raw(),
            u.as_raw(),
            d.as_raw(),
            out.as_mut_ptr(),
            x.as_ptr(),
            nr as i32,
        )
    };
    if ok {
        GPU_FFN.with(|c| c.set(c.get() + 1));
    }
    ok
}

/// Try the gateless ReLU² expert FFN `out = down(relu(up·x)²)` on the GPU (Nemotron-H's
/// two-tensor expert — no gate projection). NVFP4-only: reuses the same zero-copy NVFP4
/// decode as [`try_expert_ffn`], with a relu² activation between the up and down GEMMs.
/// Returns `true` if it ran there; the caller falls back to the CPU reference otherwise.
/// Kimi-K3 MXFP4 expert FFN with the situ activation: `y = down(situ(gate.x, up.x))`.
///
/// Exists because K3 could otherwise not use the GPU for experts at all — `ffn`
/// deliberately declines the fused SwiGLU path when situ is set (those kernels apply
/// oai-or-SiLU and would return success having computed a DIFFERENT activation), so K3's
/// routed experts ran the scalar CPU loop at 85-99% of a measured forward pass.
///
/// Zero-copy only, like the NVFP4 paths: the nibble/block-scale plumbing has no
/// device-copy variant.
pub fn try_expert_ffn_mxfp4_situ(
    gate: &QTensor,
    up: &QTensor,
    down: &QTensor,
    x: &[f32],
    nr: usize,
    out: &mut [f32],
    beta: f32,
    linear_beta: f32,
) -> bool {
    if !available() || !zerocopy() {
        return false;
    }
    if gate.fmt_code != 6 || up.fmt_code != 6 || down.fmt_code != 6 {
        return false;
    }
    let (Some(g), Some(u), Some(d)) = (wrap_fresh(gate), wrap_fresh(up), wrap_fresh(down)) else {
        return false;
    };
    // SAFETY: g/u/d live to end of scope, covering the synchronous kernel + download;
    // x/out are sized [nr, gate.i] / [nr, down.o] by `ffn`.
    let ok = unsafe {
        cuda::expert_mlp_mxfp4_situ_raw(
            g.as_raw(),
            u.as_raw(),
            d.as_raw(),
            out.as_mut_ptr(),
            x.as_ptr(),
            nr as i32,
            beta,
            linear_beta,
        )
    };
    if ok {
        GPU_FFN.with(|c| c.set(c.get() + 1));
    }
    ok
}

pub fn try_expert_ffn_relu2(
    up: &QTensor,
    down: &QTensor,
    x: &[f32],
    nr: usize,
    out: &mut [f32],
) -> bool {
    if !available() || !up.gpu_eligible || !down.gpu_eligible {
        return false;
    }
    // NVFP4 (fmt==5) is zero-copy only — the block-scale/global plumbing has no
    // device-copy path. Bail to the CPU reference for any other format or when
    // zero-copy is unavailable.
    if up.fmt_code != 5 || down.fmt_code != 5 || !zerocopy() {
        return false;
    }
    // Fresh, owned descriptors held only for this call — see `wrap_fresh`. Safe under
    // cache eviction (no stale pointer-keyed descriptors).
    let (Some(u), Some(d)) = (wrap_fresh(up), wrap_fresh(down)) else {
        return false;
    };
    // SAFETY: u/d live until end of scope, covering the synchronous kernel + download in
    // expert_mlp_nvfp4_relu2_raw; out/x sized [nr, up.I]/[nr, up.I] by ffn() (latent-space).
    let ok = unsafe {
        cuda::expert_mlp_nvfp4_relu2_raw(
            u.as_raw(),
            d.as_raw(),
            out.as_mut_ptr(),
            x.as_ptr(),
            nr as i32,
            exact_experts(),
        )
    };
    if ok {
        GPU_FFN.with(|c| c.set(c.get() + 1));
    }
    ok
}

/// `COLI_EXPERT_GROUP=1` batches a layer's routed experts through the grouped async
/// kernel (one H2D/D2H per ≤64-expert chunk) instead of a synchronous call per expert
/// — attacks the per-expert round-trip that dominates moe-compute.
/// Is the gateless ReLU² grouped expert path on? **Default ON**, unlike the fp8
/// `expert_group_enabled()` opt-in: this path declines outside decode (`rows==1`), so it
/// cannot hit the prefill devcopy gap that kept the fp8 one opt-in. MEASURED on Nemotron-H,
/// interleaved with a discarded warmup: decode 8.11 -> 8.31 tok/s (+2.5%, non-overlapping
/// ranges), prefill 38.5 -> 38.6 s (unchanged), tokens byte-identical. `COLI_EXPERT_GROUP=0`
/// turns it off.
/// `COLI_EXPERT_GROUP_PREFILL=1` lets the grouped gateless-ReLU² path take multi-row
/// (prefill) groups too. **Off by default** — see the recorded 4.7% prefill regression at
/// the gate in [`try_expert_group_relu2`]. Exists so that number can be re-measured on a
/// current binary instead of inherited.
/// `COLI_EXPERT_SEG=1` routes multi-row (prefill) expert groups through the SEGMENTED
/// kernel — one grid for the whole layer instead of three launches per expert.
///
/// ⚠️ **MEASURED NEUTRAL — leave off.** Built to test an occupancy hypothesis that turned
/// out to be wrong. Token-identical [17054], and ~1% slower:
///     off  prefill 31084 ms, moe 26194      on  prefill 31354 ms, moe 26378
///
/// The hypothesis: the per-expert path launches dim3((I+63)/64,(S+15)/16) = 86 blocks at
/// ~25 rows/expert, 453 launches back-to-back on one stream (which cannot overlap), so it
/// looked starved. The segmented form issues ~39,000 blocks in ONE grid. **86 blocks and
/// 39,000 blocks perform the same**, so occupancy was never the constraint.
///
/// What actually drives the cost — solving the 512- vs 2048-token scaling (8.916 s vs
/// 12.966 s, experts 1.13x, rows 4x) for its two components:
///     weight-read  7.91 s  (89%)   47.2 GB at 5.97 GB/s
///     row-compute  1.01 s  (11%)
/// So gpu-ffn is ~90% weight streaming. The 2.75x that the 2048-token prompt showed was
/// WEIGHT AMORTIZATION per row (118 -> 33 KB/row as rows/expert go 25 -> 88), not better
/// occupancy — and a segmented GEMM changes launch structure, not that ratio.
///
/// The real lever is therefore the weight path: 47 GB moving at ~6 GB/s. Making a layer's
/// experts device-resident before the GEMMs (1.3 GB/layer to VRAM, then ~TB/s reads) is
/// the untried option; note the existing per-expert `COLI_FFN_DEVCOPY` is NOT that — it
/// stages one expert at a time from PAGEABLE memory and measured slightly negative.
///
/// Kept because it is correct, it is the definitive disproof of the occupancy theory, and
/// it is the scaffolding a device-resident version would reuse.
fn expert_seg_enabled() -> bool {
    static ON: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *ON.get_or_init(|| std::env::var("COLI_EXPERT_SEG").ok().as_deref() == Some("1"))
}

/// Also take the segmented path when every expert has a SINGLE row — i.e. decode
/// (`COLI_EXPERT_SEG_DECODE=1`).
///
/// The seg gate below requires some expert to have >1 row, so decode never reaches it: at
/// one row per expert it issues a launch trio per expert instead of one for the layer.
/// An nsys profile of Nemotron decode shows what that costs — **26,400 `nvfp4_gemv`
/// launches averaging 15.4 us, i.e. ~407 ms, which is the whole of gpu-ffn**. Each warp
/// handles one output row and reads ~512 B, so the kernels are latency-bound, not
/// bandwidth-bound: widening their reads to 512 B/warp left the per-call time unchanged
/// (15349 -> 15688 ns).
///
/// **MEASURED: a 2.4x REGRESSION. Leave off** — but the reason recorded here was WRONG,
/// and the conclusion drawn from it was wrong too. Nemotron decode, ABBA, 12 tokens,
/// tokens identical in every arm: moe 1220 ms off vs 2963 ms on.
///
/// The original explanation was that `nvfp4_matmul_seg` tiles 16 rows, so a 1-row expert
/// wastes 15/16 of the MMA and that redundant compute swamps the ~44x reduction in
/// launches — hence "treat launch-batching for 1-row experts as settled".
///
/// **2026-08-03: that is falsified.** A true segmented GEMV (one row per expert, many
/// experts per grid, tiling nothing) was written and reproduced the SAME ~2.5x penalty.
/// The real cause is that both paths read zero-copy HOST weight pointers out of the
/// `sg_uw`/`sg_dw` DEVICE arrays; passing the identical pointers in kernel PARAMETER space
/// instead makes the identical kernel 2.8x faster. Launch batching was never the problem —
/// with parameter-space pointers it is a **1.19x WIN** on Nemotron decode
/// (11.28 -> 13.43 tok/s), which is what `COLI_NVFP4_SEG_GEMV` now ships by default.
/// See `SegP` in backend_cuda.cu for the three-arm experiment.
///
/// Kept as a knob because "why not just batch the launches?" is a question this profile
/// will keep provoking, and this is the answer with a number.
///
/// Caveat for whoever measures next: `gpu-ffn` reports ~438 ms in BOTH arms here while
/// `moe` differs by 1740 ms, so that timer does not capture this path's GPU work. Use
/// `moe`, or nsys.
fn expert_seg_decode_enabled() -> bool {
    static ON: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *ON.get_or_init(|| std::env::var("COLI_EXPERT_SEG_DECODE").ok().as_deref() == Some("1"))
}

fn group_prefill_enabled() -> bool {
    static ON: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *ON.get_or_init(|| std::env::var("COLI_EXPERT_GROUP_PREFILL").ok().as_deref() == Some("1"))
}

pub fn expert_group_relu2_enabled() -> bool {
    static ON: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *ON.get_or_init(|| std::env::var("COLI_EXPERT_GROUP").ok().as_deref() != Some("0"))
}

pub fn expert_group_enabled() -> bool {
    static ON: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *ON.get_or_init(|| std::env::var("COLI_EXPERT_GROUP").ok().as_deref() == Some("1"))
}

/// Routes int8 experts/MLPs (the shared expert + dense layers) through the tiled `i8a16`
/// tensor-core kernel instead of the naive `quant_matmul` — nsys found that kernel is 60%
/// of GPU time (its S-fold weight re-reads). Default-on; set `COLI_TILE_I8=0` to disable.
/// Measured @512 tok: attn 60.3→19.9 s (3.0×), prefill 386→334 s, tokens bit-identical.
pub fn tile_i8_enabled() -> bool {
    static ON: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *ON.get_or_init(|| std::env::var("COLI_TILE_I8").ok().as_deref() != Some("0"))
}

/// Batched routed-expert FFN. `active` is one `(expert, its token rows, its per-row
/// weights)` per active expert. Gathers all rows into one buffer (grouped by expert),
/// wraps each expert zero-copy, computes them in ≤64-expert grouped calls (one H2D/D2H
/// each), then scatters the weighted results into `out` `[n_tokens, d]`. Returns false
/// — leaving `out` untouched — if unavailable/ineligible, so the caller falls back
/// per-expert. FP8-only for now (the grouped kernel's e4m3 branch).
pub fn try_expert_group(
    active: &[(std::sync::Arc<crate::moe::Expert>, Vec<usize>, Vec<f32>)],
    activations: &[f32],
    d: usize,
    out: &mut [f32],
) -> bool {
    if !available() || !zerocopy() {
        return false;
    }
    if active.is_empty() {
        return true; // nothing routed — `out` unchanged
    }
    if !active.iter().all(|(ex, _, _)| {
        ex.gate.gpu_eligible
            && ex.gate.fmt_code == 4
            && ex.up.fmt_code == 4
            && ex.down.fmt_code == 4
    }) {
        return false;
    }
    let total: usize = active.iter().map(|(_, r, _)| r.len()).sum();
    // Gather activations, rows grouped by expert; remember each global row's dest token+weight.
    let mut x_all = vec![0f32; total * d];
    let mut token_of = vec![0usize; total];
    let mut weight_of = vec![0f32; total];
    let mut g = 0usize;
    for (_, rows, rw) in active {
        for (r, &t) in rows.iter().enumerate() {
            x_all[g * d..(g + 1) * d].copy_from_slice(&activations[t * d..(t + 1) * d]);
            token_of[g] = t;
            weight_of[g] = rw[r];
            g += 1;
        }
    }
    let mut y_all = vec![0f32; total * d];
    // Grouped calls in chunks of ≤64 experts (the C-side GroupDesc cap).
    let mut row_off = 0usize;
    let mut ci = 0usize;
    while ci < active.len() {
        let c1 = (ci + 64).min(active.len());
        let (mut gs, mut us, mut ds, mut rows_i) = (Vec::new(), Vec::new(), Vec::new(), Vec::new());
        let mut keep = Vec::new(); // hold descriptors alive across the synchronous call
        let mut chunk_rows = 0usize;
        for (ex, rows, _) in &active[ci..c1] {
            let (Some(gt), Some(ut), Some(dt)) = (
                wrap_fresh(&ex.gate),
                wrap_fresh(&ex.up),
                wrap_fresh(&ex.down),
            ) else {
                return false;
            };
            gs.push(gt.as_raw());
            us.push(ut.as_raw());
            ds.push(dt.as_raw());
            keep.push(gt);
            keep.push(ut);
            keep.push(dt);
            rows_i.push(rows.len() as i32);
            chunk_rows += rows.len();
        }
        let off = row_off * d;
        // SAFETY: gs/us/ds stay resident until `keep` drops (after the synchronous
        // group call); the x/y sub-slices hold chunk_rows*d floats each.
        let ok = unsafe {
            cuda::expert_group_raw(
                &gs,
                &us,
                &ds,
                &rows_i,
                y_all[off..off + chunk_rows * d].as_mut_ptr(),
                x_all[off..off + chunk_rows * d].as_ptr(),
            )
        };
        drop(keep);
        if !ok {
            return false;
        }
        row_off += chunk_rows;
        ci = c1;
    }
    // Scatter weighted results into the destination tokens.
    for gg in 0..total {
        let (t, wgt) = (token_of[gg], weight_of[gg]);
        let ys = &y_all[gg * d..(gg + 1) * d];
        let os = &mut out[t * d..(t + 1) * d];
        for dd in 0..d {
            os[dd] += wgt * ys[dd];
        }
    }
    true
}

/// Reusable per-thread gather/scatter scratch for [`try_expert_group_relu2`].
///
/// The grouped path packs every routed row of a layer into one contiguous buffer before
/// the call and scatters the results back after. At Nemotron's 557-token prefill that is
/// ~12.3k rows x 1024 floats = ~50 MB for `x_all` and another ~50 MB for `y_all`, freshly
/// allocated and zeroed on every one of 40 layers — ~4 GB of allocate-and-zero per
/// prefill. The identical pattern cost 8.5% of a warm prefill in the Mamba mixer (see
/// `MambaScratch` in forward.rs), which is why it is the first suspect for why grouping
/// measures ~1.05x SLOWER than the per-expert path at prefill.
///
/// Same contract as `MambaScratch`: single-threaded forward pass, buffers only grow, and
/// callers slice to the current length because a stale tail is always live.
#[derive(Default)]
struct GroupScratch {
    x_all: Vec<f32>,
    y_all: Vec<f32>,
    token_of: Vec<usize>,
    weight_of: Vec<f32>,
}

thread_local! {
    static GROUP_SCRATCH: std::cell::Cell<GroupScratch> = const {
        std::cell::Cell::new(GroupScratch {
            x_all: Vec::new(), y_all: Vec::new(),
            token_of: Vec::new(), weight_of: Vec::new(),
        })
    };
}

fn fit_f32(v: &mut Vec<f32>, n: usize) -> &mut [f32] {
    if v.len() < n {
        v.resize(n, 0.0);
    }
    &mut v[..n]
}

fn fit_usize(v: &mut Vec<usize>, n: usize) -> &mut [usize] {
    if v.len() < n {
        v.resize(n, 0);
    }
    &mut v[..n]
}

/// Batched gateless ReLU² routed-expert FFN (Nemotron-H) — the [`try_expert_group`] shape
/// for the two-tensor NVFP4 expert. Same gather/scatter, but each expert is `down(relu(up·x)²)`
/// with no gate tensor (`ex.gate` is empty and never touched).
///
/// Why this exists: the decode profile puts routed experts at 47.5 ms/token (43.4%), and the
/// cause is call count, not kernel speed — 22 experts × 40 layers = 880 separate
/// `expert_mlp_nvfp4_relu2` calls per token, each with its own H2D/D2H round-trip (~54 µs
/// per call, only ~10 µs of it real work). One grouped call per layer pays the round-trip
/// 40 times instead of 880. The per-expert kernels and their accumulation order are
/// unchanged, so results match the per-expert path exactly.
///
/// The weighted scatter stays on the CPU: it is 1.9% of decode, and moving it would mean
/// shipping row→token maps to the device for no measurable gain.
///
/// Returns false — leaving `out` untouched — if unavailable/ineligible, so the caller falls
/// back to the per-expert loop.
/// Experts staged per transfer in the grouped NVFP4 SwiGLU path (`COLI_GROUP_CHUNK`;
/// `0` = the whole layer in one go).
///
/// **32, measured.** M2.7, 128-token prefill, one binary, tokens identical ([517]) at
/// every point:
///
/// | experts/transfer | 1 | 8 | **32** | 128 | whole layer |
/// |---|---|---|---|---|---|
/// | wall | 77 s | 59 s | **56 s** | 66 s | 66 s |
///
/// It is a real optimum, not a monotone curve, which is why it is worth pinning rather
/// than defaulting to "as big as possible". Small chunks pay the per-transfer latency once
/// per expert and never let the copy engine get going; whole-layer chunks need a pinned
/// buffer and a device arena sized to every expert the layer routed, and stall the first
/// GEMM behind the last byte of the last expert. 32 amortizes the transfer while still
/// letting compute start early.
const GROUP_CHUNK_DEFAULT: usize = 32;

fn group_chunk_experts() -> Option<usize> {
    static N: std::sync::OnceLock<Option<usize>> = std::sync::OnceLock::new();
    *N.get_or_init(|| {
        match std::env::var("COLI_GROUP_CHUNK")
            .ok()
            .and_then(|s| s.parse::<usize>().ok())
        {
            Some(0) => None, // explicit 0 = whole layer
            Some(n) => Some(n),
            None => Some(GROUP_CHUNK_DEFAULT),
        }
    })
}

/// Grouped NVFP4 **SwiGLU** experts (MiniMax-M2.7 / M3 / GLM-5.2).
///
/// These models had no grouped path at all: `activation().relu2` is false for them, so the
/// dispatcher offered the *fp8* group, which declines on `fmt_code == 5` and dropped every
/// one of them to a per-expert call — ~15872 per prefill on M2.7, each with its own H2D,
/// weight staging, D2H, sync and scratch-mutex acquire.
///
/// **Grouping alone is not the point.** Its ceiling was already measured at ~3% (see the
/// relu2 twin below), and a segmented GEMM was measured and disproved as well. What this
/// unlocks is per-layer bulk residency inside the kernel: one pinned transfer of the whole
/// group's weights into a device arena, versus `COLI_FFN_DEVCOPY` staging one expert at a
/// time out of pageable memory at 1.0-2.2 GB/s. Staging is 93.6% of expert-call GPU time
/// on M2.7, so that transfer is the target — not the launches.
pub fn try_expert_group_nvfp4(
    active: &[(std::sync::Arc<crate::moe::Expert>, Vec<usize>, Vec<f32>)],
    activations: &[f32],
    d: usize,
    out: &mut [f32],
) -> bool {
    try_expert_group_packed(active, activations, d, out, 5)
}

/// `COLI_INT2_GROUP=0` restores the per-expert path for int2 experts.
///
/// A MEASUREMENT CONTROL, not a tuning knob — the same reason `COLI_NVFP4_GROUP` exists.
/// Making a grouped path unconditional deletes the only in-binary comparison against the
/// path it replaced, and the arms differ in the thing that actually changed (one launch
/// triple per layer versus one per expert), with the zero-copy weight read held fixed.
pub fn int2_group_enabled() -> bool {
    static ON: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *ON.get_or_init(|| std::env::var("COLI_INT2_GROUP").ok().as_deref() != Some("0"))
}

/// Grouped int2 (ternary) experts for the DECODE shape: every active expert consumes the
/// same single token row, so one launch triple serves the whole layer.
///
/// Declines — leaving the caller on its per-expert path — unless every expert is a
/// gpu-eligible three-tensor int2 SwiGLU of the right shape AND every one of them has
/// exactly one row, all the same token. That row test is the gate, deliberately: it is the
/// quantity at the decision point rather than a phase flag, and prefill (where each expert
/// sees a different row set, and where the per-dispatch overhead is ~12% rather than ~69%)
/// is simply not modelled here.
///
/// Nothing is staged. The weights are read in place exactly as the per-expert path reads
/// them, and the routed-weight combine below is byte-for-byte the loop the per-expert path
/// runs — same expert order, same `+=` — so the result is bit-identical.
pub fn try_expert_group_int2_decode(
    active: &[(std::sync::Arc<crate::moe::Expert>, Vec<usize>, Vec<f32>)],
    activations: &[f32],
    d: usize,
    out: &mut [f32],
) -> bool {
    if !available() || !int2_group_enabled() || active.is_empty() || active.len() > 16 {
        return false;
    }
    // Decode shape only: one row per expert, and the same row for all of them.
    let t = match active[0].1.first() {
        Some(&t) => t,
        None => return false,
    };
    if !active
        .iter()
        .all(|(_, rows, _)| rows.len() == 1 && rows[0] == t)
    {
        return false;
    }
    let inter = active[0].0.gate.o as usize;
    if !active.iter().all(|(ex, _, _)| {
        ex.gate.fmt_code == 3
            && ex.up.fmt_code == 3
            && ex.down.fmt_code == 3
            && ex.gate.gpu_eligible
            && ex.up.gpu_eligible
            && ex.down.gpu_eligible
            && ex.gate.i as usize == d
            && ex.up.i as usize == d
            && ex.down.o as usize == d
            && ex.gate.o as usize == inter
            && ex.up.o as usize == inter
            && ex.down.i as usize == inter
            && ex.gate.s.len() == inter
            && ex.up.s.len() == inter
            && ex.down.s.len() == d
    }) {
        return false;
    }
    let k = active.len();
    let (mut gw, mut uw, mut dw) = (Vec::with_capacity(k), Vec::with_capacity(k), Vec::with_capacity(k));
    let (mut gs, mut us, mut ds) = (Vec::with_capacity(k), Vec::with_capacity(k), Vec::with_capacity(k));
    for (ex, _, _) in active {
        gw.push(ex.gate.q4.as_ptr() as *const c_void);
        uw.push(ex.up.q4.as_ptr() as *const c_void);
        dw.push(ex.down.q4.as_ptr() as *const c_void);
        gs.push(ex.gate.s.as_ptr());
        us.push(ex.up.s.as_ptr());
        ds.push(ex.down.s.as_ptr());
    }
    let mut y = vec![0f32; k * d];
    // SAFETY: x is the token's `[d]` activation row; y is `[k, d]`; the six arrays each
    // hold `k` pointers into the experts' live weight/scale buffers, which outlive the call.
    let ok = unsafe {
        cuda::expert_group_int2_raw(
            y.as_mut_ptr(),
            activations[t * d..(t + 1) * d].as_ptr(),
            &gw,
            &uw,
            &dw,
            &gs,
            &us,
            &ds,
            k as i32,
            d as i32,
            inter as i32,
        )
    };
    if !ok {
        return false;
    }
    // Identical to the per-expert scatter: same expert order, same accumulation.
    for (kk, (_, _, rw)) in active.iter().enumerate() {
        let wgt = rw[0];
        for dd in 0..d {
            out[t * d + dd] += wgt * y[kk * d + dd];
        }
    }
    true
}

/// Grouped SwiGLU experts for a packed 4-bit format — NVFP4 (`fmt` 5) or MXFP4 (`fmt` 6).
///
/// Parameterised rather than duplicated: the gather/scatter, the chunking, and the RAM
/// ledger accounting are identical between the two, and only the format check and the raw
/// entry point differ. DeepSeek-V4 is MXFP4 and had NO grouped arm at all, so every expert
/// took the per-expert path — 301 dispatches per decode token, each paying its own stream
/// synchronise.
#[allow(clippy::too_many_arguments)]
pub fn try_expert_group_packed(
    active: &[(std::sync::Arc<crate::moe::Expert>, Vec<usize>, Vec<f32>)],
    activations: &[f32],
    d: usize,
    out: &mut [f32],
    fmt: i32,
) -> bool {
    if !available() || !zerocopy() {
        return false;
    }
    if active.is_empty() {
        return true;
    }
    // Every expert must be a gpu-eligible three-tensor SwiGLU of `fmt` and the expected
    // shape, or decline and let the caller run per-expert.
    if !active.iter().all(|(ex, _, _)| {
        ex.gate.gpu_eligible
            && ex.gate.fmt_code == fmt
            && ex.gate.i as usize == d
            && ex.up.gpu_eligible
            && ex.down.gpu_eligible
            && ex.up.fmt_code == fmt
            && ex.down.fmt_code == fmt
            && ex.up.i as usize == d
            && ex.down.o as usize == d
    }) {
        return false;
    }
    let total: usize = active.iter().map(|(_, r, _)| r.len()).sum();
    let mut gsc = GROUP_SCRATCH.with(|c| c.take());
    let x_all = fit_f32(&mut gsc.x_all, total * d);
    let token_of = fit_usize(&mut gsc.token_of, total);
    let weight_of = fit_f32(&mut gsc.weight_of, total);
    let mut g = 0usize;
    for (_, rows, rw) in active {
        for (r, &t) in rows.iter().enumerate() {
            x_all[g * d..(g + 1) * d].copy_from_slice(&activations[t * d..(t + 1) * d]);
            token_of[g] = t;
            weight_of[g] = rw[r];
            g += 1;
        }
    }
    let y_all = fit_f32(&mut gsc.y_all, total * d);

    // How many experts are staged per transfer (`COLI_GROUP_CHUNK`, 0 = the whole layer).
    //
    // This is the knob the residency win actually turns on, so it is worth a sweep rather
    // than a guess. Bigger chunks amortize the per-transfer latency over more weight and
    // give the copy engine a longer run; smaller chunks need a smaller pinned buffer and a
    // smaller device arena, and start the first GEMM sooner. The arena is sized to the
    // chunk, so this also bounds the device memory the path holds.
    // Size the chunk to the RAM ledger, and charge it.
    //
    // On GB10 the staging is real system memory TWICE OVER — one pinned host buffer plus
    // one device arena, both out of the same 121 GB pool — and neither was visible to the
    // ledger. GLM-5.2 has the largest dense tier (34 GB with its device duplicate) and the
    // least slack, so it was the one that tipped: earlyoom SIGTERMed it at 106.3 GiB with
    // no tokens produced. This is the same accounting hole as the dense device duplicate,
    // reintroduced by a new allocation, which is exactly why the rule has to be enforced
    // where memory is taken rather than remembered per call site.
    //
    // Degrading is cheap, so prefer it to failing: the chunk sweep put 8 experts within 5%
    // of 32 (59 s vs 56 s), so a short-on-memory model gives up very little by staging
    // less at a time. If even one expert will not fit, decline and let the caller run the
    // per-expert path, which allocates nothing beyond one expert's scratch.
    let per_expert: u64 = active
        .iter()
        .map(|(ex, _, _)| ex.bytes())
        .max()
        .unwrap_or(0);
    let staging_per_expert = per_expert.saturating_mul(2); // pinned host + device arena
    let mut chunk = group_chunk_experts().unwrap_or(active.len()).max(1);
    if staging_per_expert > 0 {
        // Quarter of headroom: KV, activations and the read buffers draw on the same
        // remainder, and a staging buffer that consumed all of it would simply move the
        // kill to the next allocator.
        let budget = crate::ram::manager()
            .map(|m| m.headroom() / 4)
            .unwrap_or(u64::MAX);
        let fits = (budget / staging_per_expert) as usize;
        if fits == 0 {
            GROUP_SCRATCH.with(|c| c.set(gsc));
            return false;
        }
        chunk = chunk.min(fits);
        crate::ram::set_usage(
            crate::ram::Class::Scratch,
            staging_per_expert.saturating_mul(chunk as u64),
        );
    }
    let mut ok = true;
    let mut done = 0usize; // rows consumed so far, to slice x_all/y_all per chunk
    for part in active.chunks(chunk) {
        let part_rows: usize = part.iter().map(|(_, r, _)| r.len()).sum();
        let mut gs: Vec<*mut cuda::ColiCudaTensor> = Vec::with_capacity(part.len());
        let mut us: Vec<*mut cuda::ColiCudaTensor> = Vec::with_capacity(part.len());
        let mut ds: Vec<*mut cuda::ColiCudaTensor> = Vec::with_capacity(part.len());
        let mut rws: Vec<i32> = Vec::with_capacity(part.len());
        let mut keep = Vec::with_capacity(part.len() * 3);
        let mut all_wrapped = true;
        for (ex, rows, _) in part {
            let (Some(gt), Some(ut), Some(dt)) = (
                wrap_fresh(&ex.gate),
                wrap_fresh(&ex.up),
                wrap_fresh(&ex.down),
            ) else {
                all_wrapped = false;
                break;
            };
            gs.push(gt.as_raw());
            us.push(ut.as_raw());
            ds.push(dt.as_raw());
            rws.push(rows.len() as i32);
            keep.push(gt);
            keep.push(ut);
            keep.push(dt);
        }
        if !all_wrapped {
            drop(keep);
            ok = false;
            break;
        }
        // SAFETY: the wrapped tensors stay alive in `keep` until after the call, which is
        // synchronous through its own D2H; the sub-slices hold `part_rows * d` floats each.
        let called = unsafe {
            let (yp, xp) = (
                y_all[done * d..(done + part_rows) * d].as_mut_ptr(),
                x_all[done * d..(done + part_rows) * d].as_ptr(),
            );
            if fmt == 6 {
                cuda::expert_group_mxfp4_raw(&gs, &us, &ds, &rws, yp, xp)
            } else {
                cuda::expert_group_nvfp4_raw(&gs, &us, &ds, &rws, yp, xp)
            }
        };
        drop(keep);
        if !called {
            ok = false;
            break;
        }
        done += part_rows;
    }
    if ok {
        for gg in 0..total {
            let (t, wgt) = (token_of[gg], weight_of[gg]);
            let ys = &y_all[gg * d..(gg + 1) * d];
            let os = &mut out[t * d..(t + 1) * d];
            for dd in 0..d {
                os[dd] += wgt * ys[dd];
            }
        }
    }
    GROUP_SCRATCH.with(|c| c.set(gsc));
    ok
}

pub fn try_expert_group_relu2(
    active: &[(std::sync::Arc<crate::moe::Expert>, Vec<usize>, Vec<f32>)],
    activations: &[f32],
    d: usize,
    out: &mut [f32],
) -> bool {
    // NVFP4 (fmt==5) is zero-copy only — the block-scale/global plumbing has no
    // device-copy path (same guard as `try_expert_ffn_relu2`).
    if !available() || !zerocopy() {
        return false;
    }
    if active.is_empty() {
        return true; // nothing routed — `out` unchanged
    }
    // Decode-only BY DEFAULT — the prefill regression is REAL, but the reason recorded here
    // originally was WRONG, so read this before trying to "just group them".
    //
    // Grouping saves the per-expert H2D/D2H round-trip: a decode win (+5% end-to-end,
    // measured interleaved). At prefill it LOSES, re-measured 2026-07-26 on a current
    // binary via `COLI_EXPERT_GROUP_PREFILL=1`, 2 reps, token-identical [17054]:
    //     off  31404 / 31098 ms      on  32273 / 32895 ms      ~1.05x SLOWER
    //
    // The old explanation — that grouping forfeits the `COLI_FFN_DEVCOPY` weight staging
    // the per-expert path gets at S>=16 — does not survive measurement: devcopy is not
    // helping. A/B'd on the per-expert path, turning it OFF was FASTER (gpu-ffn 8911 ->
    // 8375 ms). So the loss is not about where the weights live.
    //
    // ⚠️ AND GROUPING CANNOT WIN HERE — the ceiling is ~3%. CUDA-event timing of the
    // per-expert path splits its 9060 ms as: H2D activations 72 ms, D2H+sync 184 ms, host
    // memcpy 18 ms, and 7575 ms (84%) inside the kernel window. The per-expert round-trip
    // that grouping eliminates is therefore ~256 ms, under 3% of the cost. Grouping
    // optimizes the wrong 3%.
    //
    // Two rounds of evidence. The path's own gather buffers (~50 MB x_all + ~50 MB y_all
    // per layer at 557 tokens) were freshly allocated per layer; hoisting them into
    // `GroupScratch` moved grouping from ~1.05x slower to ~1.013x slower — so most of the
    // original regression was allocation churn, the same thing that cost 8.5% in the Mamba
    // mixer. But it is STILL a loss, 5 reps each, token-identical [17054]:
    //     off  31394/31129/31195/30917/31263   median 31195 ms
    //     on   32023/31224/31772/31602/31440   median 31602 ms
    // because grouping still launches 3 kernels PER EXPERT (nvfp4_matmul / relu2 /
    // nvfp4_matmul in a `for c<count` loop): it removes transfers, not launches.
    //
    // So do not revisit grouping for prefill. The 84% lives in the kernel window — ~47 GB
    // of expert weight traffic per prefill (453 experts x 40 layers x ~2.95 MB) moving at
    // ~6 GB/s against a ~51 GB/s zero-copy ceiling, at an average of only ~25 rows per
    // call. The fix is a fused SEGMENTED GEMM: one launch per layer with per-expert tile
    // ranges, so the weight stream gets real memory-level parallelism.
    //
    // `COLI_EXPERT_GROUP_PREFILL=1` lifts the restriction so this stays re-measurable.
    if !group_prefill_enabled()
        && !expert_seg_enabled()
        && active.iter().any(|(_, rows, _)| rows.len() != 1)
    {
        return false;
    }
    // `d` is the expert input width (the MoE latent for Nemotron-H, not the model hidden);
    // the kernel derives D from up.I, so decline rather than mis-stride if they disagree.
    if !active.iter().all(|(ex, _, _)| {
        ex.up.gpu_eligible
            && ex.down.gpu_eligible
            && ex.up.fmt_code == 5
            && ex.down.fmt_code == 5
            && ex.up.i as usize == d
            && ex.down.o as usize == d
    }) {
        return false;
    }
    let total: usize = active.iter().map(|(_, r, _)| r.len()).sum();
    // Gather activations, rows grouped by expert; remember each global row's dest token+weight.
    // Reused across layers — see `GroupScratch`. Moved out and back rather than held
    // borrowed, so the body below is unchanged apart from the slice types.
    let mut gsc = GROUP_SCRATCH.with(|c| c.take());
    let x_all = fit_f32(&mut gsc.x_all, total * d);
    let token_of = fit_usize(&mut gsc.token_of, total);
    let weight_of = fit_f32(&mut gsc.weight_of, total);
    let mut g = 0usize;
    for (_, rows, rw) in active {
        for (r, &t) in rows.iter().enumerate() {
            x_all[g * d..(g + 1) * d].copy_from_slice(&activations[t * d..(t + 1) * d]);
            token_of[g] = t;
            weight_of[g] = rw[r];
            g += 1;
        }
    }
    let y_all = fit_f32(&mut gsc.y_all, total * d);

    // SEGMENTED fast path: one grid for the whole layer. Only worth it when experts carry
    // real row counts (prefill); at decode every expert has a single row and the grouped
    // chunk path already amortizes the round-trip.
    if (expert_seg_enabled() && active.iter().any(|(_, rows, _)| rows.len() > 1))
        || (expert_seg_decode_enabled() && active.len() > 1)
    {
        let mut us: Vec<*mut cuda::ColiCudaTensor> = Vec::with_capacity(active.len());
        let mut ds: Vec<*mut cuda::ColiCudaTensor> = Vec::with_capacity(active.len());
        let mut rws: Vec<i32> = Vec::with_capacity(active.len());
        let mut keep = Vec::with_capacity(active.len() * 2);
        let mut all_wrapped = true;
        for (ex, rows, _) in active {
            let (Some(ut), Some(dt)) = (wrap_fresh(&ex.up), wrap_fresh(&ex.down)) else {
                all_wrapped = false;
                break;
            };
            us.push(ut.as_raw());
            ds.push(dt.as_raw());
            rws.push(rows.len() as i32);
            keep.push(ut);
            keep.push(dt);
        }
        if all_wrapped {
            // SAFETY: us/ds stay resident until `keep` drops (after the synchronous call,
            // which includes the D2H); x_all/y_all hold `total * d` floats each.
            let ok = unsafe {
                cuda::expert_seg_nvfp4_relu2_raw(&us, &ds, &rws, y_all.as_mut_ptr(), x_all.as_ptr())
            };
            drop(keep);
            if ok {
                for gg in 0..total {
                    let (t, wgt) = (token_of[gg], weight_of[gg]);
                    let ys = &y_all[gg * d..(gg + 1) * d];
                    let os = &mut out[t * d..(t + 1) * d];
                    for dd in 0..d {
                        os[dd] += wgt * ys[dd];
                    }
                }
                GROUP_SCRATCH.with(|c| c.set(gsc));
                return true;
            }
        } else {
            drop(keep);
        }
        // fall through to the chunked grouped path on any decline
    }
    // Chunked at 64 experts to share the shape of the fp8 grouped path (Nemotron routes 22
    // per layer, so this is one chunk in practice).
    let mut row_off = 0usize;
    let mut ci = 0usize;
    while ci < active.len() {
        let c1 = (ci + 64).min(active.len());
        let (mut us, mut ds, mut rows_i) = (Vec::new(), Vec::new(), Vec::new());
        let mut keep = Vec::new(); // hold descriptors alive across the synchronous call
        let mut chunk_rows = 0usize;
        for (ex, rows, _) in &active[ci..c1] {
            // Fresh, owned descriptors held only for this call — see `wrap_fresh`. Safe
            // under cache eviction (no stale pointer-keyed descriptors).
            let (Some(ut), Some(dt)) = (wrap_fresh(&ex.up), wrap_fresh(&ex.down)) else {
                // Put the scratch back before declining, or the next call reallocates.
                GROUP_SCRATCH.with(|c| c.set(gsc));
                return false;
            };
            us.push(ut.as_raw());
            ds.push(dt.as_raw());
            keep.push(ut);
            keep.push(dt);
            rows_i.push(rows.len() as i32);
            chunk_rows += rows.len();
        }
        let off = row_off * d;
        // SAFETY: us/ds stay resident until `keep` drops (after the synchronous group call,
        // which includes the D2H); the x/y sub-slices hold chunk_rows*d floats each.
        let ok = unsafe {
            cuda::expert_group_nvfp4_relu2_raw(
                &us,
                &ds,
                &rows_i,
                y_all[off..off + chunk_rows * d].as_mut_ptr(),
                x_all[off..off + chunk_rows * d].as_ptr(),
            )
        };
        drop(keep);
        if !ok {
            GROUP_SCRATCH.with(|c| c.set(gsc));
            return false;
        }
        row_off += chunk_rows;
        ci = c1;
    }
    // Scatter weighted results into the destination tokens (CPU — 1.9% of decode).
    for gg in 0..total {
        let (t, wgt) = (token_of[gg], weight_of[gg]);
        let ys = &y_all[gg * d..(gg + 1) * d];
        let os = &mut out[t * d..(t + 1) * d];
        for dd in 0..d {
            os[dd] += wgt * ys[dd];
        }
    }
    GROUP_SCRATCH.with(|c| c.set(gsc));
    true
}

/// Try to run `y[S,O] = x[S,I] @ W^T` on the GPU. Returns `true` if it ran there;
/// `false` (do it on the CPU) when CUDA is unavailable or `w` isn't eligible.
pub fn try_matmul_qt(y: &mut [f32], x: &[f32], w: &QTensor, s: usize) -> bool {
    if !w.gpu_eligible || !available() {
        return false;
    }
    // This dense-upload GPU matmul handles the resident formats f32 (0) and int8 (1).
    // Packed formats store fewer bytes than a dense `o*i` buffer, so uploading them here
    // reads out of bounds: NVFP4 is handled just above via its own wrap+dispatch, and
    // e4m3 has the fused expert kernel (`try_expert_ffn`) or the CPU reference.
    // NVFP4 (fmt 5) has its own entry point: the weight is packed (nibbles + block scales),
    // so the dense `o*i` upload below would read out of bounds. The device kernels are the
    // same general ones the expert path uses — only the wrapping differs.
    if w.fmt_code == 5 {
        let key = w.q4.as_ptr() as usize;
        return RESIDENT.with(|r| {
            let mut map = r.borrow_mut();
            let slot = map.entry(key).or_insert(std::ptr::null_mut());
            // SAFETY: y/x sized by matmul_qt's asserts; q4/bs are this QTensor's live
            // buffers (o*ceil(i/2) and o*ceil(i/16)); the slot persists in the registry.
            let ok = unsafe {
                cuda::matmul_nvfp4_raw(
                    slot,
                    y.as_mut_ptr(),
                    x.as_ptr(),
                    w.q4.as_ptr() as *const c_void,
                    w.bs.as_ptr() as *const c_void,
                    w.g,
                    s as i32,
                    w.i,
                    w.o,
                    0,
                )
            };
            if ok {
                GPU_MATMULS.with(|c| c.set(c.get() + 1));
            }
            ok
        });
    }
    let (wptr, key): (*const c_void, usize) = match w.fmt_code {
        0 => (w.qf.as_ptr() as *const c_void, w.qf.as_ptr() as usize),
        1 => (w.q8.as_ptr() as *const c_void, w.q8.as_ptr() as usize),
        // int2 (fmt 3) — Maple's ternary attention projections. This arm was missing, and
        // the omission is the `gpu_eligible` trap in its quietest form: the weight IS
        // marked eligible, so nothing looks wrong, and `matmul_qt` just falls through to
        // the single-threaded CPU int2 loop. Same failure shape as the M3 q/k/v
        // projections (84% of a prefill) and V4's O-LoRA (58% of decode).
        //
        // Safe despite the "packed formats read out of bounds" note above, which is about
        // NVFP4 specifically: the C side sizes both the alloc and the copy by
        // `row_bytes(fmt, I) * O`, and `row_bytes` already returns `ceil(I/4)` for fmt 3.
        // What actually disqualifies NVFP4 here is its SEPARATE block-scale array, which
        // `coli_cuda_matmul` has no parameter for — hence its own entry point above. int2
        // has no sidecar: its scales are the same per-row f32 vector int8 passes, and
        // `weight_at`/`quant_matmul` have decoded fmt 3 all along.
        3 => (w.q4.as_ptr() as *const c_void, w.q4.as_ptr() as usize),
        // bf16 (fmt 2) — the IO tier. Same reasoning as int2 above: the C side sizes the
        // alloc and the copy by `row_bytes(fmt, I) * O`, and `row_bytes` returns `I*2`
        // here. Unlike int2 there is no scale sidecar at all, which is why the upload
        // path's scale copy had to stop testing `fmt != 0`.
        2 => (w.q4.as_ptr() as *const c_void, w.q4.as_ptr() as usize),
        _ => return false,
    };
    let sptr = w.s.as_ptr();
    RESIDENT.with(|r| {
        let mut map = r.borrow_mut();
        let slot = map.entry(key).or_insert(std::ptr::null_mut());
        // SAFETY: y/x sized by the caller (matmul_qt asserts); slot persists in
        // the registry; wptr/sptr point at the live QTensor buffers.
        let ok = unsafe {
            cuda::matmul_raw(
                slot,
                y.as_mut_ptr(),
                x.as_ptr(),
                wptr,
                sptr,
                w.fmt_code,
                s as i32,
                w.i,
                w.o,
                0,
            )
        };
        if ok {
            GPU_MATMULS.with(|c| c.set(c.get() + 1));
        }
        ok
    })
}

/// Dense f32 matmul `y[s,o] = x[s,i] @ w[o,i]^T` on the GPU, full f32 precision.
/// For the MoE router projection: a single-threaded CPU `matmul_f32` there was
/// measured at ~40% of moe-compute (~248 s @4096 tok) while the GPU sat idle. The
/// router weight is numerically sensitive so it stays f32 (fmt=0 — no scales). The
/// weight is resident-cached by its pointer, so it uploads once. Returns false to
/// fall back to the CPU path when CUDA is unavailable.
pub fn try_matmul_f32(y: &mut [f32], x: &[f32], w: &[f32], s: usize, i: usize, o: usize) -> bool {
    if !available() || w.len() != o * i {
        return false;
    }
    let key = w.as_ptr() as usize;
    RESIDENT.with(|r| {
        let mut map = r.borrow_mut();
        let slot = map.entry(key).or_insert(std::ptr::null_mut());
        // SAFETY: y sized [s,o], x sized [s,i] by the caller; w is [o,i] f32 and
        // outlives the resident tensor (a model weight). fmt=0 ⇒ scales unused, so
        // a null scales pointer is valid (see coli_cuda_tensor_upload / quant_matmul).
        let ok = unsafe {
            cuda::matmul_raw(
                slot,
                y.as_mut_ptr(),
                x.as_ptr(),
                w.as_ptr() as *const c_void,
                std::ptr::null(),
                0,
                s as i32,
                i as i32,
                o as i32,
                0,
            )
        };
        if ok {
            GPU_MATMULS.with(|c| c.set(c.get() + 1));
        }
        ok
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::linear::matmul_qt;
    use crate::quantize::qtensor_from_f32;

    /// A deterministic MXFP4 weight [o, i]: nibble k of row r cycles the e2m1 codebook and
    /// the E8M0 block scale varies per block AND per row, so a wrong nibble order or a
    /// wrong block stride (32, not NVFP4's 16) cannot coincide with the right answer.
    /// Same construction as `linear::tests::matmul_qt_reconstructs_mxfp4`, which pins the
    /// CPU arm this test uses as its reference.
    fn mxfp4_weight(o: usize, i: usize, seed: usize) -> QTensor {
        let nb = i / 32;
        let mut q4 = vec![0u8; o * i / 2];
        let mut bs = vec![0u8; o * nb];
        for r in 0..o {
            for k in 0..i {
                let nib = ((k + r * 3 + seed) % 16) as u8;
                let idx = r * (i / 2) + (k >> 1);
                if k & 1 == 1 {
                    q4[idx] |= nib << 4;
                } else {
                    q4[idx] |= nib;
                }
            }
            for b in 0..nb {
                bs[r * nb + b] = (126 + ((b + r + seed) % 3) as i32) as u8;
            }
        }
        QTensor {
            fmt_code: 6,
            q4: colibri_core::Bytes::Owned(q4),
            bs: colibri_core::Bytes::Owned(bs),
            g: 0.5,
            o: o as i32,
            i: i as i32,
            ..Default::default()
        }
    }

    /// The MXFP4 expert FFN on the GPU must agree with the CPU reference at every S, and
    /// this is the ONLY check that covers it: the read-pattern and weight-stationary
    /// kernels sum in a different order than the kernels they replace, so cross-kernel
    /// token identity is the wrong gate — agreement with `matmul_qt` is the right one.
    ///
    /// S is chosen to enter each arm of `coli_cuda_expert_mlp_mxfp4`: 1 = the decode GEMV
    /// dispatcher, 4/16/32 = the three weight-stationary MT buckets, 33 = the WMMA tile
    /// the WSMM launcher declines into. A kernel that quietly truncates rows past its
    /// bucket, or one that never runs at all, fails here rather than in a benchmark.
    ///
    /// The shapes matter as much as S. `mxfp4_wsmm` sweeps K in KT=128 tiles, so a K of
    /// exactly 128 runs that loop ONCE and proves nothing about accumulating across tiles
    /// or about a short final tile — which is the shape every real expert has (D=4096,
    /// I=2048). 320x160 gives gate/up K=320 (two full tiles + 64) and down K=160 (one full
    /// tile + 32); 128x64 keeps the exact-multiple case alongside it.
    ///
    /// Sets the CUDA activation globals; run CUDA tests with `--test-threads=1` (task #57).
    #[test]
    fn mxfp4_expert_ffn_gpu_matches_cpu_at_every_s() {
        if !available() || !zerocopy() {
            eprintln!("skip: no zero-copy CUDA device");
            return;
        }
        set_activation(false, 0.0, 0.0); // plain SiLU: act_mul -> silu_mul

        for &(d, inter) in &[(128usize, 64usize), (320, 160)] {
            let gate = mxfp4_weight(inter, d, 0);
            let up = mxfp4_weight(inter, d, 5);
            let down = mxfp4_weight(d, inter, 9);

            for &s in &[1usize, 4, 16, 32, 33] {
                let x: Vec<f32> =
                    (0..s * d).map(|k| ((k % 11) as f32 - 5.0) * 0.05).collect();

                // CPU reference — exactly `moe::ffn_cpu`'s plain-SwiGLU arm.
                let mut gg = vec![0f32; s * inter];
                let mut uu = vec![0f32; s * inter];
                matmul_qt(&mut gg, &x, &gate, s);
                matmul_qt(&mut uu, &x, &up, s);
                for (g, &u) in gg.iter_mut().zip(uu.iter()) {
                    *g = crate::math::silu(*g) * u;
                }
                let mut want = vec![0f32; s * d];
                matmul_qt(&mut want, &gg, &down, s);

                let mut got = vec![0f32; s * d];
                assert!(
                    try_expert_ffn_mxfp4(&gate, &up, &down, &x, s, &mut got),
                    "[{d}x{inter}] S={s}: the GPU MXFP4 expert declined — this test would \
                     otherwise pass by comparing the CPU reference against a zero buffer"
                );
                let tol = 1e-3 * want.iter().fold(1.0f32, |m, v| m.max(v.abs()));
                for (j, (&a, &b)) in got.iter().zip(want.iter()).enumerate() {
                    assert!(
                        (a - b).abs() <= tol,
                        "[{d}x{inter}] S={s} elem {j}: gpu {a} vs cpu {b} (tol {tol})"
                    );
                }
            }
        }
    }

    // GPU vs CPU-NEON matmul at GLM-scale sizes.
    // `cargo test -p colibri-engine --features cuda --release -- --ignored --nocapture bench_matmul`
    #[test]
    #[ignore]
    fn bench_matmul_gpu_vs_cpu() {
        if !available() {
            eprintln!("skip: no CUDA device");
            return;
        }
        // o_proj-scale int8 weight [O, I]
        let (o, i) = (8192usize, 6144usize);
        let wf: Vec<f32> = (0..o * i).map(|k| ((k % 13) as f32 - 6.0) * 0.01).collect();
        let mut w = qtensor_from_f32(&wf, o, i, 8);
        for &s in &[1usize, 32] {
            let x = vec![0.01f32; s * i];
            let mut y = vec![0f32; s * o];
            let iters = 1000u64;
            w.gpu_eligible = true;
            matmul_qt(&mut y, &x, &w, s); // warm upload
            let t = std::time::Instant::now();
            for _ in 0..iters {
                matmul_qt(&mut y, &x, &w, s);
            }
            let gpu = t.elapsed().as_secs_f64();
            w.gpu_eligible = false; // force CPU (NEON int8)
            let t = std::time::Instant::now();
            for _ in 0..iters {
                matmul_qt(&mut y, &x, &w, s);
            }
            let cpu = t.elapsed().as_secs_f64();
            let flops = iters as f64 * s as f64 * o as f64 * i as f64 * 2.0;
            eprintln!(
                "matmul [{o},{i}] S={s} x{iters}: GPU {:.3}s ({:.0} GFLOP/s) | CPU-NEON {:.3}s ({:.0} GFLOP/s) | {:.2}x",
                gpu, flops / gpu / 1e9, cpu, flops / cpu / 1e9, cpu / gpu
            );
        }
    }
}
