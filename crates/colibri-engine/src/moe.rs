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
use colibri_cluster::{ExpertSharding, NodeId};
use colibri_core::{Bytes, Config, QTensor};
use colibri_safetensors::Shards;
use std::io;
use std::sync::Arc;

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
/// once. Default: **on** when the zero-copy path is available (unified memory) —
/// it's memory-safe and ~2× the copy path. `COLI_GPU_EXPERTS=1`/`=0` forces it;
/// `=1` also enables the copy path (needs `COLI_VRAM_GB` caps, see [`load_expert`]).
fn gpu_experts_enabled() -> bool {
    use std::sync::OnceLock;
    static ON: OnceLock<bool> = OnceLock::new();
    *ON.get_or_init(|| match std::env::var("COLI_GPU_EXPERTS").ok().as_deref() {
        Some("1") => true,
        Some("0") => false,
        _ => {
            #[cfg(feature = "cuda")]
            {
                crate::gpu::zerocopy()
            }
            #[cfg(not(feature = "cuda"))]
            {
                false
            }
        }
    })
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
        let mk = |o: usize, i: usize, w: &(std::sync::Arc<[u8]>, usize, usize), sname: String| -> io::Result<QTensor> {
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
            Ok(t)
        };
        Expert {
            gate: mk(moe_inter, hidden, &ws[0], format!("{gate_w}.qs"))?,
            up: mk(moe_inter, hidden, &ws[1], format!("{up_w}.qs"))?,
            down: mk(hidden, moe_inter, &ws[2], format!("{down_w}.qs"))?,
        }
    } else {
        // Full-tensor (runtime-quantized) path — the tiny oracle model.
        Expert {
            gate: crate::loader::qt_load(shards, &gate_w, moe_inter, hidden, ebits)?,
            up: crate::loader::qt_load(shards, &up_w, moe_inter, hidden, ebits)?,
            down: crate::loader::qt_load(shards, &down_w, hidden, moe_inter, ebits)?,
        }
    };
    // Route streamed experts through the GPU fused-FFN path. On unified memory
    // (the GB10) this uses the zero-copy wrap — the kernel reads the RAM copy in
    // place, so there is no VRAM double-store, no eviction, and no OOM — and it is
    // ~2× the copy path. Default-on there; see [`gpu_experts_enabled`] for the
    // `COLI_GPU_EXPERTS` override and the copy-path caveats on other devices.
    if gpu_experts_enabled() {
        ex.mark_gpu_eligible();
    }
    Ok(ex)
}

impl ExpertProvider for ShardsExpertProvider<'_> {
    fn expert(&self, layer: usize, eid: usize) -> io::Result<Arc<Expert>> {
        // Expert-parallel ownership: local experts load from disk; non-local ones
        // would be fetched over the RDMA transport (not wired — single node now).
        if !self.sharding.is_local(self.this_node, eid as u32) {
            return Err(io::Error::new(
                io::ErrorKind::Unsupported,
                format!("expert {eid} owned by another node; RDMA transport not wired"),
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
    let d = cfg.hidden as usize;
    let e_n = cfg.n_experts as usize;
    let k = (cfg.topk as usize).min(e_n);

    // ---- router (f32) + top-K per position --------------------------------
    let mut logits = vec![0f32; s_len * e_n];
    matmul_f32(&mut logits, x, &l.router, s_len, d, e_n);

    let mut idxs = vec![0usize; s_len * k];
    let mut ws = vec![0f32; s_len * k];
    for s in 0..s_len {
        let (idx, w) = route(cfg, &logits[s * e_n..(s + 1) * e_n], &l.router_bias);
        idxs[s * k..s * k + k].copy_from_slice(&idx);
        ws[s * k..s * k + k].copy_from_slice(&w);
    }

    for v in out.iter_mut() {
        *v = 0.0;
    }

    // ---- union of experts across the batch --------------------------------
    let mut seen = vec![false; e_n];
    let mut uniq = Vec::new();
    for &e in &idxs {
        if !seen[e] {
            seen[e] = true;
            uniq.push(e);
        }
    }

    // Fetch this layer's experts disk→RAM in parallel before computing. Serial
    // per-expert loading is otherwise the decode bottleneck (~74% of MoE time).
    if crate::forward::profile_on() {
        let t = std::time::Instant::now();
        provider.prefetch(layer, &uniq)?;
        crate::forward::LOAD_US
            .fetch_add(t.elapsed().as_micros() as u64, std::sync::atomic::Ordering::Relaxed);
    } else {
        provider.prefetch(layer, &uniq)?;
    }

    // ---- apply each unique expert to the positions that route to it -------
    for &e in &uniq {
        let mut rows = Vec::new();
        let mut rw = Vec::new();
        for s in 0..s_len {
            for kk in 0..k {
                if idxs[s * k + kk] == e {
                    rows.push(s);
                    rw.push(ws[s * k + kk]);
                    break;
                }
            }
        }
        let nr = rows.len();
        let mut xg = vec![0f32; nr * d];
        for (r, &s) in rows.iter().enumerate() {
            xg[r * d..(r + 1) * d].copy_from_slice(&x[s * d..(s + 1) * d]);
        }
        let ex = provider.expert(layer, e)?;
        let mut hh = vec![0f32; nr * d];
        ffn(&ex.gate, &ex.up, &ex.down, &xg, nr, &mut hh);
        for (r, &s) in rows.iter().enumerate() {
            let wgt = rw[r];
            for dd in 0..d {
                out[s * d + dd] += wgt * hh[r * d + dd];
            }
        }
    }

    // ---- shared expert (weight 1.0, all positions) ------------------------
    if with_shared {
        let mut sh = vec![0f32; s_len * d];
        ffn(&l.sh_gate, &l.sh_up, &l.sh_down, x, s_len, &mut sh);
        for (o, &s) in out.iter_mut().zip(sh.iter()) {
            *o += s;
        }
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
}
