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

thread_local! {
    static AVAIL: OnceCell<bool> = const { OnceCell::new() };
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

/// Whether a CUDA device is usable (probed once; honors `COLI_CUDA=0`).
pub fn available() -> bool {
    AVAIL.with(|c| *c.get_or_init(|| cuda::CudaBackend::probe().is_some()))
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
    if qi.len() < nsp * nh * hd || hw.len() < nsp * nh || keys.len() < t_len * hd
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
                cuda::expert_mlp_nvfp4_raw(g.as_raw(), u.as_raw(), d.as_raw(), out.as_mut_ptr(), x.as_ptr(), nr as i32)
            } else if gate.fmt_code == 4 {
                cuda::expert_mlp_fp8_raw(g.as_raw(), u.as_raw(), d.as_raw(), out.as_mut_ptr(), x.as_ptr(), nr as i32)
            } else if gate.fmt_code == 1 && tile_i8_enabled() {
                cuda::expert_mlp_i8a16_raw(g.as_raw(), u.as_raw(), d.as_raw(), out.as_mut_ptr(), x.as_ptr(), nr as i32)
            } else {
                cuda::expert_mlp_raw(g.as_raw(), u.as_raw(), d.as_raw(), out.as_mut_ptr(), x.as_ptr(), nr as i32)
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
    let (Some(g), Some(u), Some(d)) =
        (upload_ffn(gate, &keys), upload_ffn(up, &keys), upload_ffn(down, &keys))
    else {
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

/// Try the gateless ReLU² expert FFN `out = down(relu(up·x)²)` on the GPU (Nemotron-H's
/// two-tensor expert — no gate projection). NVFP4-only: reuses the same zero-copy NVFP4
/// decode as [`try_expert_ffn`], with a relu² activation between the up and down GEMMs.
/// Returns `true` if it ran there; the caller falls back to the CPU reference otherwise.
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
            u.as_raw(), d.as_raw(), out.as_mut_ptr(), x.as_ptr(), nr as i32, exact_experts(),
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
        ex.gate.gpu_eligible && ex.gate.fmt_code == 4 && ex.up.fmt_code == 4 && ex.down.fmt_code == 4
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
        let (mut gs, mut us, mut ds, mut rows_i) =
            (Vec::new(), Vec::new(), Vec::new(), Vec::new());
        let mut keep = Vec::new(); // hold descriptors alive across the synchronous call
        let mut chunk_rows = 0usize;
        for (ex, rows, _) in &active[ci..c1] {
            let (Some(gt), Some(ut), Some(dt)) =
                (wrap_fresh(&ex.gate), wrap_fresh(&ex.up), wrap_fresh(&ex.down))
            else {
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
    if !group_prefill_enabled() && !expert_seg_enabled() && active.iter().any(|(_, rows, _)| rows.len() != 1) {
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
    if expert_seg_enabled() && active.iter().any(|(_, rows, _)| rows.len() > 1) {
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
                cuda::expert_seg_nvfp4_relu2_raw(
                    &us,
                    &ds,
                    &rws,
                    y_all.as_mut_ptr(),
                    x_all.as_ptr(),
                )
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
