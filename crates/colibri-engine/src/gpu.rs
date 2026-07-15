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
    // Zero-copy FFN descriptors on unified memory: each just points at the RAM
    // QTensor, so there is no device memory to budget or evict. Keyed by CPU
    // weight pointer; the `ResidentTensor` frees only the tiny descriptor on drop.
    static WRAPPED_FFN: RefCell<HashMap<usize, cuda::ResidentTensor>> =
        RefCell::new(HashMap::new());
    static GPU_MATMULS: std::cell::Cell<u64> = const { std::cell::Cell::new(0) };
    static GPU_FFN: std::cell::Cell<u64> = const { std::cell::Cell::new(0) };
    static GPU_ATTN: std::cell::Cell<u64> = const { std::cell::Cell::new(0) };
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
    let Some(handle) = upload_ffn(kv_b, &[], false) else {
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
    let Some(handle) = upload_ffn(kv_b, &[], false) else {
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

fn weight_ptr(w: &QTensor) -> *const c_void {
    match w.fmt_code {
        0 => w.qf.as_ptr() as *const c_void,
        1 => w.q8.as_ptr() as *const c_void,
        _ => w.q4.as_ptr() as *const c_void,
    }
}

/// Zero-copy wrap of `w` on unified memory: cache a descriptor pointing at the
/// live RAM buffers (no device allocation, no eviction, no budget). Keyed by CPU
/// data pointer so repeated tokens reuse the descriptor.
fn wrap_ffn(w: &QTensor) -> Option<*mut ColiCudaTensor> {
    let key = weight_ptr(w) as usize;
    WRAPPED_FFN.with(|r| {
        let mut m = r.borrow_mut();
        if let Some(rt) = m.get(&key) {
            return Some(rt.as_raw());
        }
        // SAFETY: weight_ptr/scales point at the live QTensor buffers (owned by the
        // Arc<Expert> in the resident cache), sized by [O,I]/fmt; int4 stays
        // offset-binary (the kernel reads it with off=1). Only called when
        // `zerocopy()` (device supports pageable host access).
        let rt = unsafe {
            cuda::ResidentTensor::wrap_raw(weight_ptr(w), w.s.as_ptr(), w.fmt_code, w.i, w.o, 0)
        }?;
        let raw = rt.as_raw();
        m.insert(key, rt);
        Some(raw)
    })
}

/// Upload `w` to the GPU (once) and return its resident handle. When `zc` and the
/// device supports it, this wraps the RAM copy in place (zero-copy, offset-binary
/// int4); otherwise it copies to device memory (converting int4 to signed),
/// caching by data pointer under the VRAM budget. `protect` lists the current op's
/// other tensor keys so eviction never drops a tensor still needed this op.
///
/// `zc` must be **false** for weights whose kernel reads int4 with the signed
/// interpretation (the attention absorb kernels): those have no offset-binary path,
/// so they need the converted device copy. Expert/MLP FFN kernels pass `zc=true`.
fn upload_ffn(w: &QTensor, protect: &[usize], zc: bool) -> Option<*mut ColiCudaTensor> {
    if zc && zerocopy() {
        return wrap_ffn(w);
    }
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
    // all three must stay resident together for the fused kernel — protect them
    let keys = [
        weight_ptr(gate) as usize,
        weight_ptr(up) as usize,
        weight_ptr(down) as usize,
    ];
    let (Some(g), Some(u), Some(d)) = (
        upload_ffn(gate, &keys, true),
        upload_ffn(up, &keys, true),
        upload_ffn(down, &keys, true),
    ) else {
        return false;
    };
    // SAFETY: handles are resident on device 0; out/x sized [nr, O]/[nr, I] by ffn().
    let ok = unsafe { cuda::expert_mlp_raw(g, u, d, out.as_mut_ptr(), x.as_ptr(), nr as i32) };
    if ok {
        GPU_FFN.with(|c| c.set(c.get() + 1));
    }
    ok
}

/// Try to run `y[S,O] = x[S,I] @ W^T` on the GPU. Returns `true` if it ran there;
/// `false` (do it on the CPU) when CUDA is unavailable or `w` isn't eligible.
pub fn try_matmul_qt(y: &mut [f32], x: &[f32], w: &QTensor, s: usize) -> bool {
    if !w.gpu_eligible || !available() {
        return false;
    }
    // weight bytes + a stable address key, per format
    let (wptr, key): (*const c_void, usize) = match w.fmt_code {
        0 => (w.qf.as_ptr() as *const c_void, w.qf.as_ptr() as usize),
        1 => (w.q8.as_ptr() as *const c_void, w.q8.as_ptr() as usize),
        _ => (w.q4.as_ptr() as *const c_void, w.q4.as_ptr() as usize),
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
        // o_proj-scale int4 weight [O, I]
        let (o, i) = (8192usize, 6144usize);
        let wf: Vec<f32> = (0..o * i).map(|k| ((k % 13) as f32 - 6.0) * 0.01).collect();
        let mut w = qtensor_from_f32(&wf, o, i, 4);
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
            w.gpu_eligible = false; // force CPU (NEON int4)
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
