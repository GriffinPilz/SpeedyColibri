//! The assembled forward pass and greedy decode loop — port of `layer_forward`
//! / `layers_forward` / `forward_all` / `generate` in `c/glm.c`.
//!
//! Per layer (CPU path): `in_ln` RMSNorm → MLA attention → residual add →
//! `post_ln` RMSNorm → MoE (or dense MLP for the first `first_k_dense_replace`
//! layers) → residual add. Then a final RMSNorm and the `lm_head` produce
//! logits, and greedy decoding feeds the argmax back in one token at a time.

use crate::attention::{attention_gqa, attention_sharded, attention_with, AttnCore};
use crate::linear::{embed_row, matmul_qt};
use crate::mamba2::{causal_conv1d_silu, gated_rmsnorm, selective_scan, MambaDims};
use crate::math::rmsnorm;
use crate::model::{KvCache, Layer, Model};
use crate::moe::{cluster_ctx, dense_mlp, moe, ExpertProvider};
use crate::sampling::argmax;
use colibri_core::{Arch, Config, LayerKind};
use std::io;
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::OnceLock;

/// `COLI_TP_ATTN=1` enables tensor-parallel attention: split the heads across cluster
/// nodes so every box's GPU runs part of the (dominant) attention core, instead of the
/// driver computing all heads while peers idle. Off by default; only takes effect in a
/// multi-node cluster during single-shot prefill (see [`layer_forward`]).
fn tp_attn_enabled() -> bool {
    static ON: OnceLock<bool> = OnceLock::new();
    *ON.get_or_init(|| std::env::var("COLI_TP_ATTN").ok().as_deref() == Some("1"))
}

/// COLI_PROFILE=1 accumulates per-section wall time (microseconds) across the
/// forward pass so `generate_greedy` can print a breakdown. Off by default.
pub(crate) fn profile_on() -> bool {
    static ON: OnceLock<bool> = OnceLock::new();
    *ON.get_or_init(|| std::env::var("COLI_PROFILE").ok().as_deref() == Some("1"))
}
static ATTN_US: AtomicU64 = AtomicU64::new(0);
static MOE_US: AtomicU64 = AtomicU64::new(0);
static DENSE_US: AtomicU64 = AtomicU64::new(0);
/// Nemotron-H Mamba2 mixer wall time (its analog of `ATTN_US` for attention layers).
static MAMBA_US: AtomicU64 = AtomicU64::new(0);
/// Sub-totals of `MAMBA_US`: the selective scan (GPU kernel or CPU `selective_scan`)
/// and the in_proj + out_proj matmuls. The remainder is conv1d + splits + gated norm.
static MAMBA_SCAN_US: AtomicU64 = AtomicU64::new(0);
static MAMBA_PROJ_US: AtomicU64 = AtomicU64::new(0);
/// Further sub-totals of `MAMBA_US`, split out of what the profile used to report as one
/// "conv+norm" remainder: the causal depthwise conv1d+silu and the gated per-group RMSNorm.
static MAMBA_CONV_US: AtomicU64 = AtomicU64::new(0);
static MAMBA_NORM_US: AtomicU64 = AtomicU64::new(0);
static EMBED_US: AtomicU64 = AtomicU64::new(0);
/// Time spent fetching experts through the provider (disk→RAM on a cache miss).
/// A sub-total of `MOE_US`. Incremented from `moe`.
pub(crate) static LOAD_US: AtomicU64 = AtomicU64::new(0);
/// Sub-totals of `MOE_US` (compute side, excludes LOAD_US): CPU row-gather into the
/// per-expert activation buffer, the GPU FFN call incl. sync/transfers, and the
/// weighted scatter back into the output. Incremented from `compute_experts_partial`.
pub(crate) static GATHER_US: AtomicU64 = AtomicU64::new(0);
pub(crate) static GPUFFN_US: AtomicU64 = AtomicU64::new(0);
pub(crate) static SCATTER_US: AtomicU64 = AtomicU64::new(0);
/// Sub-totals of `MOE_US` outside the routed-expert loop: the CPU router projection
/// (`matmul_f32`) and the shared-expert FFN. Incremented from `moe`.
pub(crate) static ROUTER_US: AtomicU64 = AtomicU64::new(0);
pub(crate) static SHARED_US: AtomicU64 = AtomicU64::new(0);
/// Sub-totals of `ATTN_US`: q/kv projections, RoPE + latent-cache write, the DSA
/// lightning indexer, the attention core (sparse/dense), and the output projection.
/// Incremented from `attention_with`.
pub(crate) static ATTN_PROJ_US: AtomicU64 = AtomicU64::new(0);
pub(crate) static ATTN_ROPE_US: AtomicU64 = AtomicU64::new(0);
pub(crate) static ATTN_INDEX_US: AtomicU64 = AtomicU64::new(0);
pub(crate) static ATTN_CORE_US: AtomicU64 = AtomicU64::new(0);
pub(crate) static ATTN_OPROJ_US: AtomicU64 = AtomicU64::new(0);

/// Monotonic forward-pass counter — one per `forward` call, i.e. per decode token
/// (prefill is a single step over the whole prompt). Used only to key the optional
/// expert-routing log so a token's per-layer expert sequence is reconstructable.
static FWD_STEP: AtomicU64 = AtomicU64::new(0);

/// The current forward step (see [`FWD_STEP`]).
pub fn current_step() -> u64 {
    FWD_STEP.load(Ordering::Relaxed)
}

/// Tokens to speculatively draft per forward via the MTP head: `DRAFT=n`.
///
/// **Defaults to 0 (off)** — same as the C's `g_draft`, where speculation is
/// opt-in because the win is workload- and acceptance-dependent. Capped at 63
/// (the C's `draft[64]`).
pub(crate) fn draft_budget() -> usize {
    static N: OnceLock<usize> = OnceLock::new();
    *N.get_or_init(|| {
        std::env::var("DRAFT").ok().and_then(|v| v.parse::<usize>().ok()).unwrap_or(0).min(63)
    })
}
/// Time `f` into `acc` when profiling is enabled (else just run it).
#[inline]
fn timed<T>(acc: &AtomicU64, f: impl FnOnce() -> T) -> T {
    if !profile_on() {
        return f();
    }
    let t = std::time::Instant::now();
    let r = f();
    acc.fetch_add(t.elapsed().as_micros() as u64, Ordering::Relaxed);
    r
}

/// Run ONE layer over `x[S * hidden]` in place (positions
/// `pos_base..pos_base+S`), updating `kv[li]`. Port of `layer_forward`.
///
/// `nrm`/`tmp` are caller-owned scratch, each `[S * hidden]`, so a hot loop can
/// reuse them. Shared by the main stack and by the MTP head, which runs its own
/// block at `li = n_layers`.
pub fn layer_forward<P: ExpertProvider>(
    model: &Model,
    kv: &mut KvCache,
    provider: &P,
    l: &Layer,
    li: usize,
    x: &mut [f32],
    s: usize,
    pos_base: usize,
    nrm: &mut [f32],
    tmp: &mut [f32],
    dsa_sel: &mut Option<Vec<Vec<u32>>>,
) -> io::Result<()> {
    layer_forward_kind(model, kv, provider, l, li, None, x, s, pos_base, nrm, tmp, dsa_sel)
}

/// [`layer_forward`] with an explicit mixer `kind`, for layers that are NOT in
/// `cfg.layer_kind`.
///
/// The only such layers are the MTP head's sublayers: they run at `li >= n_layers`, past
/// the end of `layer_kind` (which is `num_hidden_layers` long and describes the main
/// stack), so indexing it there would panic. `kind: None` means "look it up", which is
/// what every main-stack call does and what keeps the GLM/M3 path byte-identical —
/// those arches ignore `kind` entirely.
#[allow(clippy::too_many_arguments)]
pub fn layer_forward_kind<P: ExpertProvider>(
    model: &Model,
    kv: &mut KvCache,
    provider: &P,
    l: &Layer,
    li: usize,
    kind: Option<LayerKind>,
    x: &mut [f32],
    s: usize,
    pos_base: usize,
    nrm: &mut [f32],
    tmp: &mut [f32],
    dsa_sel: &mut Option<Vec<Vec<u32>>>,
) -> io::Result<()> {
    let cfg = &model.cfg;
    // Nemotron-H is a hybrid single-sublayer-per-layer stack (Mamba2 / GQA attention /
    // latent-MoE) dispatched by `layer_kind`, with no post-attention norm — so it takes
    // its own path rather than the two-sublayer GLM/M3 driver below. `dsa_sel` is unused.
    if cfg.arch == Arch::NemotronH {
        let kind = kind.unwrap_or_else(|| cfg.layer_kind[li]);
        return nemotron_layer_forward(model, kv, provider, l, li, kind, x, s, pos_base, nrm, tmp);
    }
    let d = cfg.hidden as usize;
    // in_ln -> attention -> residual
    for si in 0..s {
        rmsnorm(&mut nrm[si * d..(si + 1) * d], &x[si * d..(si + 1) * d], &l.in_ln, cfg.eps);
    }
    if cfg.arch.is_gqa() {
        // MiniMax-M3: grouped-query attention (no MLA latent, no DSA indexer).
        timed(&ATTN_US, || attention_gqa(cfg, l, li, kv, nrm, s, pos_base, tmp));
    } else {
        // GLM DSA selection sharing: a FULL indexer layer (idx_type == true) computes
        // its own top-k selection; SHARED layers (false) reuse the most recent FULL
        // layer's — the `indexer_types` pattern. `attention_with` returns the selection
        // it computed so we can carry it forward across the stack.
        let is_full = cfg.idx_type.get(li).copied().unwrap_or(false);
        let reused = if is_full { None } else { dsa_sel.as_deref() };
        let computed = timed(&ATTN_US, || -> io::Result<Option<Vec<Vec<u32>>>> {
            // Tensor-parallel attention: split the heads across nodes so every box's GPU
            // runs part of the core. Only for a multi-node cluster during single-shot
            // prefill (`pos_base == 0`, `s > 1`) — peers build a fresh KV from the shipped
            // activations, so there is no cross-step state. Decode and the single-node
            // build keep the driver computing all heads via `attention_with`.
            if pos_base == 0 && s > 1 && tp_attn_enabled() {
                if let Some(cc) = cluster_ctx() {
                    if cc.sharding.num_nodes() > 1 {
                        return attention_sharded(
                            cfg, l, li, kv, nrm, s, pos_base, tmp, reused, &cc.sharding, &*cc.transport,
                        );
                    }
                }
            }
            Ok(attention_with(cfg, l, li, kv, nrm, s, pos_base, tmp, AttnCore::Reconstruct, reused))
        })?;
        if is_full {
            *dsa_sel = computed;
        }
    }
    for j in 0..s * d {
        x[j] += tmp[j];
    }
    // post_ln -> MoE/dense -> residual
    for si in 0..s {
        rmsnorm(&mut nrm[si * d..(si + 1) * d], &x[si * d..(si + 1) * d], &l.post_ln, cfg.eps);
    }
    if l.sparse {
        // with_shared only when the model actually has a shared expert (GLM/M3 do;
        // MiniMax-M2 has none — n_shared 0, shared_intermediate_size 0).
        timed(&MOE_US, || moe(cfg, l, li, nrm, s, tmp, cfg.n_shared > 0, provider))?;
    } else {
        timed(&DENSE_US, || dense_mlp(l, nrm, s, tmp));
    }
    for j in 0..s * d {
        x[j] += tmp[j];
    }
    Ok(())
}

/// Run ONE Nemotron-H layer over `x[S * hidden]` in place. Nemotron-H blocks have a
/// single sublayer — `in_ln` (the only norm; there is no post-norm) → the mixer named by
/// `kind` → residual add — unlike the two-sublayer GLM/M3 [`layer_forward`]. `kind` is
/// passed in rather than read from `cfg.layer_kind[li]` because the MTP head's sublayers
/// run at `li >= n_layers`, outside that vector (see [`layer_forward_kind`]).
/// Mamba2 and attention layers update their per-layer state in `kv`; MoE layers stream
/// their routed experts through `provider`.
#[allow(clippy::too_many_arguments)]
fn nemotron_layer_forward<P: ExpertProvider>(
    model: &Model,
    kv: &mut KvCache,
    provider: &P,
    l: &Layer,
    li: usize,
    kind: LayerKind,
    x: &mut [f32],
    s: usize,
    pos_base: usize,
    nrm: &mut [f32],
    tmp: &mut [f32],
) -> io::Result<()> {
    let cfg = &model.cfg;
    let d = cfg.hidden as usize;
    // in_ln (the block's only norm) -> mixer -> residual.
    for si in 0..s {
        rmsnorm(&mut nrm[si * d..(si + 1) * d], &x[si * d..(si + 1) * d], &l.in_ln, cfg.eps);
    }
    match kind {
        LayerKind::Mamba => timed(&MAMBA_US, || mamba2_mixer(cfg, l, kv, li, nrm, s, tmp)),
        // NoPE GQA attention (no rotary, no QK-norm — see `attention_gqa`).
        LayerKind::Attn => timed(&ATTN_US, || attention_gqa(cfg, l, li, kv, nrm, s, pos_base, tmp)),
        LayerKind::Moe => {
            timed(&MOE_US, || crate::moe::nemotron_moe(cfg, l, li, nrm, s, tmp, provider))?
        }
    }
    for j in 0..s * d {
        x[j] += tmp[j];
    }
    Ok(())
}

/// Reusable per-thread scratch for [`mamba2_mixer`].
///
/// The mixer used to allocate every intermediate fresh: at Nemotron's 557-token prefill
/// that is ~128 MB of zeroed `Vec` per layer (`proj` alone is 41 MB), 40 layers deep, so
/// ~5 GB of allocate-and-zero per prefill. It measured **8.5% of a warm prefill**, hidden
/// inside what the profile reported as the "conv+norm" remainder until that field was
/// split (see `MAMBA_CONV_US`).
///
/// The forward pass is single-threaded — the same assumption `gpu.rs` already makes for
/// its GPU slot registry — so a `thread_local` needs no locking. Buffers only ever GROW,
/// and every consumer must slice to the CURRENT length rather than trusting `.len()`:
/// after a 557-token prefill the buffers stay 557 wide while decode calls with `s == 1`,
/// so a stale tail is always live. `causal_conv1d_silu` and `selective_scan` both assert
/// their input lengths, which turns a slicing mistake into an immediate panic rather than
/// silent corruption — that is what makes this safe to do by hand.
#[derive(Default)]
struct MambaScratch {
    proj: Vec<f32>,
    gate: Vec<f32>,
    hbc: Vec<f32>,
    dt: Vec<f32>,
    aug: Vec<f32>,
    h: Vec<f32>,
    b: Vec<f32>,
    c: Vec<f32>,
}

thread_local! {
    static MAMBA_SCRATCH: std::cell::Cell<MambaScratch> =
        const { std::cell::Cell::new(MambaScratch {
            proj: Vec::new(), gate: Vec::new(), hbc: Vec::new(), dt: Vec::new(),
            aug: Vec::new(), h: Vec::new(), b: Vec::new(), c: Vec::new(),
        }) };
}

/// Grow `v` to at least `n` and hand back exactly the first `n` elements. New space is
/// zeroed; previously-used space keeps whatever was there, which is why callers must
/// fully overwrite the slice they take.
fn fit(v: &mut Vec<f32>, n: usize) -> &mut [f32] {
    if v.len() < n {
        v.resize(n, 0.0);
    }
    &mut v[..n]
}

/// Nemotron-H Mamba2 mixer over the block-normed input `x[S, hidden]`, writing
/// `out[S, hidden]`. Carries the layer's recurrent conv + ssm state in `kv`, so a
/// single-shot prefill (`S = prompt`) and a run of one-token decode steps (`S = 1`)
/// produce identical outputs. Port of `NemotronHMamba2Mixer.torch_forward`:
///
/// ```text
///   proj        = in_proj · x                        # [S, d_inner + conv_dim + n_heads]
///   gate,hBC,dt = split(proj, [d_inner, conv_dim, n_heads])   # d_mlp == 0
///   hBC         = silu(causal_conv1d(hBC))           # depthwise, carries conv state
///   h,B,C       = split(hBC, [d_inner, G·N, G·N])
///   y           = selective_scan(h, B, C, dt; A, D, dt_bias)  # carries ssm state
///   y           = gated_rmsnorm(y, gate)             # per-group, silu gate
///   out         = out_proj · y                        # [S, hidden]
/// ```
pub fn mamba2_mixer(
    cfg: &Config,
    l: &Layer,
    kv: &mut KvCache,
    layer: usize,
    x: &[f32],
    s: usize,
    out: &mut [f32],
) {
    let nh = cfg.mamba_n_heads as usize;
    let hd = cfg.mamba_head_dim as usize;
    let ds = cfg.mamba_d_state as usize;
    let ng = cfg.mamba_n_groups as usize;
    let d_inner = cfg.mamba_inter as usize; // n_heads * head_dim
    let gn = ng * ds; // per-group B/C width (n_groups * d_state)
    let conv_dim = d_inner + 2 * gn;
    let kk = cfg.mamba_d_conv as usize;
    let proj_out = d_inner + conv_dim + nh;

    let in_proj = l.mamba_in_proj.as_ref().expect("mamba layer missing in_proj");
    let out_proj = l.mamba_out_proj.as_ref().expect("mamba layer missing out_proj");

    // Reused across layers and calls — see `MambaScratch`. Moved out and back rather than
    // held borrowed, so the body below is unchanged apart from the slice types.
    let mut sc = MAMBA_SCRATCH.with(|c| c.take());

    // ---- in_proj, then split gate | hidden_B_C | dt ----------------------
    let proj = fit(&mut sc.proj, s * proj_out);
    timed(&MAMBA_PROJ_US, || matmul_qt(proj, x, in_proj, s));
    let gate = fit(&mut sc.gate, s * d_inner);
    let hbc = fit(&mut sc.hbc, s * conv_dim);
    let dt = fit(&mut sc.dt, s * nh);
    for t in 0..s {
        let base = t * proj_out;
        gate[t * d_inner..(t + 1) * d_inner].copy_from_slice(&proj[base..base + d_inner]);
        hbc[t * conv_dim..(t + 1) * conv_dim]
            .copy_from_slice(&proj[base + d_inner..base + d_inner + conv_dim]);
        dt[t * nh..(t + 1) * nh].copy_from_slice(&proj[base + d_inner + conv_dim..base + proj_out]);
    }

    // ---- causal depthwise conv1d + silu, carrying the conv state ---------
    // Prepend the saved (k-1) history columns so the conv at each new token sees its real
    // preceding context (all-zero at a fresh sequence start), then keep the last k input
    // columns as the next state. `causal_conv1d_silu`'s left zero-pad only touches the
    // discarded history rows, so this reduces to a plain prefill when the state is zero,
    // and its per-token outputs are identical whether the sequence arrives whole or split.
    let hist = kk - 1;
    let aug_len = hist + s;
    let aug = fit(&mut sc.aug, aug_len * conv_dim);
    {
        // conv state is [k, conv_dim] time-major (row k-1 = most recent); the (k-1)
        // history columns before this chunk are rows [1, k).
        let cs = kv.mamba_conv_row(layer);
        aug[..hist * conv_dim].copy_from_slice(&cs[conv_dim..kk * conv_dim]);
    }
    aug[hist * conv_dim..].copy_from_slice(&hbc[..s * conv_dim]);
    let conv_aug = timed(&MAMBA_CONV_US, || {
        causal_conv1d_silu(aug, &l.mamba_conv_w, &l.mamba_conv_b, aug_len, conv_dim, kk)
    });
    let conv_out = &conv_aug[hist * conv_dim..aug_len * conv_dim]; // [s, conv_dim]
    // Next state = the last k input columns of the augmented buffer.
    kv.mamba_conv_row_mut(layer)
        .copy_from_slice(&aug[(aug_len - kk) * conv_dim..aug_len * conv_dim]);

    // ---- split conv output into h | B | C, then the selective scan -------
    let h = fit(&mut sc.h, s * d_inner);
    let b = fit(&mut sc.b, s * gn);
    let c = fit(&mut sc.c, s * gn);
    for t in 0..s {
        let base = t * conv_dim;
        h[t * d_inner..(t + 1) * d_inner].copy_from_slice(&conv_out[base..base + d_inner]);
        b[t * gn..(t + 1) * gn].copy_from_slice(&conv_out[base + d_inner..base + d_inner + gn]);
        c[t * gn..(t + 1) * gn].copy_from_slice(&conv_out[base + d_inner + gn..base + conv_dim]);
    }
    let dims = MambaDims {
        n_heads: nh,
        head_dim: hd,
        d_state: ds,
        n_groups: ng,
        dt_min: cfg.mamba_dt_min,
    };
    // The recurrent scan routes to the GPU at EVERY sequence length: the per-head
    // step/decay are precomputed here (bit-identical softplus/exp to the CPU), then the
    // device kernel does the fma-free multiply/add recurrence and updates the persisted
    // ssm state in place. Any GPU-unavailable case, or a backend decline, falls to the
    // CPU `selective_scan`.
    //
    // ⚠️ The two kernels carry DIFFERENT contracts. S==1 (decode) is bit-identical to the
    // CPU. S>1 (prefill) is only TOKEN-identical: it threads per (head, head-dim row,
    // state index) to get 128x the parallelism, which makes `y`'s sum over d_state a tree
    // rather than the CPU's strict order (~1 ULP). The bit-exact prefill form was built
    // first and measured no faster than the CPU (7.57 s vs ~7.50 s) because it exposes
    // only n_heads*head_dim threads. `COLI_MAMBA_CPU=1` forces the CPU scan for an A/B.
    //
    // Prefill used to fall to the CPU unconditionally, which cost 7.95 s of a warm 23.1 s
    // Nemotron prefill — 34%, the single largest block — because 40 Mamba layers ran a
    // sequential scalar scan across the whole prompt. `S == 1` and `S > 1` take different
    // kernels only because the sequence form stages the head's state in shared memory and
    // carries a t loop; both reproduce the CPU operand order exactly.
    let y = timed(&MAMBA_SCAN_US, || {
        #[cfg(feature = "cuda")]
        let gpu_y: Option<Vec<f32>> = if crate::gpu::available()
            && crate::gpu::mamba_scan_gpu_enabled()
        {
            let mut yv = vec![0f32; s * d_inner];
            if s == 1 {
                let (dt_h, da_h) =
                    crate::mamba2::step_head_scalars(dims, dt, &l.mamba_a_log, &l.mamba_dt_bias);
                let st = kv.mamba_ssm_mut(layer);
                crate::gpu::try_mamba2_scan(
                    &mut st.data, &mut yv, h, b, c, &dt_h, &da_h, &l.mamba_d, nh, hd, ds, ng,
                )
                .then_some(yv)
            } else {
                let (dt_h, da_h) = crate::mamba2::seq_head_scalars(
                    dims, dt, &l.mamba_a_log, &l.mamba_dt_bias, s,
                );
                let st = kv.mamba_ssm_mut(layer);
                crate::gpu::try_mamba2_scan_seq(
                    &mut st.data, &mut yv, h, b, c, &dt_h, &da_h, &l.mamba_d, nh, hd, ds, ng, s,
                )
                .then_some(yv)
            }
        } else {
            None
        };
        #[cfg(not(feature = "cuda"))]
        let gpu_y: Option<Vec<f32>> = None;
        match gpu_y {
            Some(v) => v,
            None => selective_scan(
                dims,
                kv.mamba_ssm_mut(layer),
                h,
                b,
                c,
                dt,
                &l.mamba_a_log,
                &l.mamba_d,
                &l.mamba_dt_bias,
                s,
            ),
        }
    });

    // ---- gated RMSNorm (per group, silu gate) then out_proj --------------
    let yn = timed(&MAMBA_NORM_US, || {
        gated_rmsnorm(&y, &gate, &l.mamba_norm, s, d_inner, ng, cfg.eps)
    });
    timed(&MAMBA_PROJ_US, || matmul_qt(out, &yn, out_proj, s));
    MAMBA_SCRATCH.with(|c| c.set(sc));
}

/// Run the transformer stack over `ids` (positions `pos_base..pos_base+S`),
/// updating `kv` and writing the final hidden states `[S * hidden]` to
/// `hidden_out`. Port of embed + `layers_forward`.
///
/// `hidden_out` is the **raw** hidden state — before `model.norm`. That is what
/// [`logits`] and the MTP head both expect as input (each applies `final_norm`
/// itself).
pub fn forward<P: ExpertProvider>(
    model: &Model,
    kv: &mut KvCache,
    provider: &P,
    ids: &[i32],
    pos_base: usize,
    hidden_out: &mut [f32],
) -> io::Result<()> {
    let cfg = &model.cfg;
    let d = cfg.hidden as usize;
    let s = ids.len();
    assert_eq!(hidden_out.len(), s * d);
    FWD_STEP.fetch_add(1, Ordering::Relaxed);

    // token embeddings
    let mut x = vec![0f32; s * d];
    timed(&EMBED_US, || {
        for (i, &tok) in ids.iter().enumerate() {
            embed_row(&model.embed, tok as usize, &mut x[i * d..(i + 1) * d]);
        }
    });

    // COLI_DEBUG_ACT=1: print the hidden-state L2 norm (first + last position) after
    // embedding and the first few layers, to localize where a forward pass degenerates.
    let dbg_act = std::env::var("COLI_DEBUG_ACT").ok().as_deref() == Some("1");
    let pnorm = |tag: &str, x: &[f32]| {
        if s == 0 {
            return;
        }
        let n = |r: &[f32]| r.iter().map(|v| v * v).sum::<f32>().sqrt();
        eprintln!(
            "[act] {tag}: |x[0]|={:.4} |x[{}]|={:.4} x[0][..4]={:?}",
            n(&x[..d]),
            s - 1,
            n(&x[(s - 1) * d..s * d]),
            &x[..4.min(d)]
        );
    };
    if dbg_act {
        pnorm("embed", &x);
    }

    let mut nrm = vec![0f32; s * d];
    let mut tmp = vec![0f32; s * d];
    // Carries the current DSA selection from each FULL indexer layer to the SHARED
    // layers that follow it. Fresh per forward pass; stays None when DSA is inactive
    // (short context or decode), so those layers run dense as before.
    let mut dsa_sel: Option<Vec<Vec<u32>>> = None;
    for li in 0..model.layers.len() {
        layer_forward(
            model,
            kv,
            provider,
            &model.layers[li],
            li,
            &mut x,
            s,
            pos_base,
            &mut nrm,
            &mut tmp,
            &mut dsa_sel,
        )?;
        if dbg_act && li < 5 {
            pnorm(&format!("layer{li}"), &x);
        }
    }

    hidden_out.copy_from_slice(&x);
    Ok(())
}

/// One transformer layer over an N-sequence decode batch: each sequence `si`
/// contributes one token at absolute position `positions[si]` with its own
/// `kvs[si]`. Attention runs **per sequence** (an S=1 decode against that
/// sequence's own KV history — you cannot express N different positions over N
/// different-length histories as one contiguous `pos_base` sweep), but the
/// post-attention MoE/dense block runs **once** over all N rows. That single MoE
/// call is the point: [`crate::moe::compute_experts_partial`] streams the union of
/// routed experts from disk exactly once and scatters the result to every
/// contributing token, so the (bytes-bound) expert reads amortize across the batch.
#[allow(clippy::too_many_arguments)]
fn layer_forward_batched<P: ExpertProvider>(
    model: &Model,
    kvs: &mut [KvCache],
    provider: &P,
    l: &Layer,
    li: usize,
    x: &mut [f32],
    positions: &[usize],
    nrm: &mut [f32],
    tmp: &mut [f32],
) -> io::Result<()> {
    let cfg = &model.cfg;
    let d = cfg.hidden as usize;
    let n = positions.len();
    // in_ln -> attention (per sequence) -> residual
    for si in 0..n {
        rmsnorm(&mut nrm[si * d..(si + 1) * d], &x[si * d..(si + 1) * d], &l.in_ln, cfg.eps);
    }
    timed(&ATTN_US, || {
        for si in 0..n {
            // Per sequence: S=1, its own KV, its own position — identical to a lone decode
            // step, so batching cannot change any sequence's output (only shares the MoE
            // expert reads below). MiniMax-M3 uses the GQA core; GLM uses MLA reconstruct
            // (decode never fires DSA at pos_base>0, so there is no selection to carry).
            if cfg.arch.is_gqa() {
                attention_gqa(
                    cfg,
                    l,
                    li,
                    &mut kvs[si],
                    &nrm[si * d..(si + 1) * d],
                    1,
                    positions[si],
                    &mut tmp[si * d..(si + 1) * d],
                );
            } else {
                attention_with(
                    cfg,
                    l,
                    li,
                    &mut kvs[si],
                    &nrm[si * d..(si + 1) * d],
                    1,
                    positions[si],
                    &mut tmp[si * d..(si + 1) * d],
                    AttnCore::Reconstruct,
                    None,
                );
            }
        }
    });
    for j in 0..n * d {
        x[j] += tmp[j];
    }
    // post_ln -> MoE/dense (ONCE over all N rows — the amortization) -> residual
    for si in 0..n {
        rmsnorm(&mut nrm[si * d..(si + 1) * d], &x[si * d..(si + 1) * d], &l.post_ln, cfg.eps);
    }
    if l.sparse {
        timed(&MOE_US, || moe(cfg, l, li, nrm, n, tmp, cfg.n_shared > 0, provider))?;
    } else {
        timed(&DENSE_US, || dense_mlp(l, nrm, n, tmp));
    }
    for j in 0..n * d {
        x[j] += tmp[j];
    }
    Ok(())
}

/// Advance **N independent sequences by one decode step each** in a single forward.
/// Sequence `si` feeds token `ids[si]` at absolute position `positions[si]` with its
/// own `kvs[si]`, and its raw hidden row is written to `hidden_out[si*d..]`.
///
/// The whole reason this exists: decode is **bytes-bound** — each step streams the
/// routed experts' weights from disk, which dwarfs the compute. Running N sequences'
/// steps through one MoE call makes each unique expert's weights load **once** for the
/// whole batch instead of once per sequence, so aggregate tok/s rises with N (until
/// the per-layer expert union saturates all 256 experts). Per-sequence output is
/// unchanged: attention is per-sequence and every matmul is per-row, so sequence
/// `si`'s row is identical to a lone [`forward`] of that token — batching only shares
/// the expert reads. See [`crate::moe`] `union_and_weights`/`compute_experts_partial`.
pub fn forward_batched<P: ExpertProvider>(
    model: &Model,
    kvs: &mut [KvCache],
    provider: &P,
    ids: &[i32],
    positions: &[usize],
    hidden_out: &mut [f32],
) -> io::Result<()> {
    let cfg = &model.cfg;
    let d = cfg.hidden as usize;
    let n = ids.len();
    assert_eq!(kvs.len(), n, "one KvCache per sequence");
    assert_eq!(positions.len(), n, "one position per sequence");
    assert_eq!(hidden_out.len(), n * d);
    // `layer_forward_batched` assumes the uniform transformer shape every layer is
    // `in_ln -> attention -> residual -> MoE`, with attention picked by `is_gqa()`.
    // A hybrid stack (`layer_kind` non-empty: Nemotron-H's Mamba2/Attn/MoE mix) has
    // neither property, so it used to fall through to the MLA branch and panic deep in
    // `matmul_qt` on an empty `q_a` ("x must be [S,I], left: 4096, right: 0"). Refuse
    // it here with a real error instead. Only `coli genbatch` reaches this — `serve`
    // and single-sequence `gen` never call it, so nothing user-facing regressed.
    if !cfg.layer_kind.is_empty() {
        return Err(io::Error::new(
            io::ErrorKind::Unsupported,
            format!(
                "batched decode is not implemented for hybrid architectures ({:?}); \
                 use single-sequence `coli gen`",
                cfg.arch
            ),
        ));
    }
    FWD_STEP.fetch_add(1, Ordering::Relaxed);

    let mut x = vec![0f32; n * d];
    timed(&EMBED_US, || {
        for (i, &tok) in ids.iter().enumerate() {
            embed_row(&model.embed, tok as usize, &mut x[i * d..(i + 1) * d]);
        }
    });
    let mut nrm = vec![0f32; n * d];
    let mut tmp = vec![0f32; n * d];
    for li in 0..model.layers.len() {
        layer_forward_batched(
            model,
            kvs,
            provider,
            &model.layers[li],
            li,
            &mut x,
            positions,
            &mut nrm,
            &mut tmp,
        )?;
    }
    hidden_out.copy_from_slice(&x);
    Ok(())
}

/// Logits for a single hidden-state row: final RMSNorm then `lm_head`. Port of
/// the tail of `forward_all`.
pub fn logits(model: &Model, hidden_row: &[f32]) -> Vec<f32> {
    let d = model.cfg.hidden as usize;
    let v = model.cfg.vocab as usize;
    let mut row = vec![0f32; d];
    rmsnorm(&mut row, hidden_row, &model.final_norm, model.cfg.eps);
    let mut lo = vec![0f32; v];
    matmul_qt(&mut lo, &row, &model.lm_head, 1);
    lo
}

/// Greedy generation: prefill the prompt, then decode up to `n_new` tokens by
/// argmax, feeding each back through the cache. Stops early on a config stop
/// token. Port of `generate` (greedy path, no speculation). Returns the full
/// sequence (prompt + continuation).
pub fn generate_greedy<P: ExpertProvider>(
    model: &Model,
    kv: &mut KvCache,
    provider: &P,
    prompt: &[i32],
    n_new: usize,
) -> io::Result<Vec<i32>> {
    let mut out = prompt.to_vec();
    generate_stream(model, kv, provider, prompt, n_new, |tok| {
        out.push(tok);
        true
    })?;
    Ok(out)
}

/// Streaming greedy generation: like [`generate_greedy`], but invokes `on_token`
/// with each newly decoded token id as it is produced (before the next forward
/// step), so a caller can stream output live. Returning `false` from `on_token`
/// stops generation early — used by the server to abort when a client
/// disconnects. A config stop token is delivered to `on_token` and then ends the
/// run. `generate_greedy` is a thin wrapper that collects the tokens.
pub fn generate_stream<P, F>(
    model: &Model,
    kv: &mut KvCache,
    provider: &P,
    prompt: &[i32],
    n_new: usize,
    on_token: F,
) -> io::Result<()>
where
    P: ExpertProvider,
    F: FnMut(i32) -> bool,
{
    let budget = if model.has_mtp { draft_budget() } else { 0 };
    generate_stream_drafting(model, kv, provider, prompt, n_new, budget, on_token)?;
    Ok(())
}

/// What a decode run did. `forwards < emitted` is exactly the speculation win;
/// `drafts_accepted / drafts_proposed` is the acceptance rate that decides
/// whether the head is earning its keep.
#[derive(Debug, Default, Clone, Copy, PartialEq, Eq)]
pub struct DecodeStats {
    pub emitted: usize,
    /// main-model forwards actually run
    pub forwards: u64,
    pub drafts_proposed: u64,
    pub drafts_accepted: u64,
}

/// [`generate_stream`] with an explicit speculation budget: `budget` tokens are
/// drafted by the MTP head per forward and verified against the main model.
/// `generate_stream` supplies `DRAFT` for it.
///
/// `budget == 0` disables speculation, and the loop then reduces exactly to the
/// plain one-token-per-forward path — which is the property the
/// "speculation does not change output" test relies on. Exposed separately
/// because `DRAFT` is read once per process and so cannot be varied in-process.
pub fn generate_stream_drafting<P, F>(
    model: &Model,
    kv: &mut KvCache,
    provider: &P,
    prompt: &[i32],
    n_new: usize,
    budget: usize,
    mut on_token: F,
) -> io::Result<DecodeStats>
where
    P: ExpertProvider,
    F: FnMut(i32) -> bool,
{
    let d = model.cfg.hidden as usize;
    assert!(!prompt.is_empty(), "prompt must be non-empty");
    assert!(
        kv.max_t >= prompt.len() + n_new,
        "Kv cache too small: max_t={} needs >= {}",
        kv.max_t,
        prompt.len() + n_new
    );

    // COLI_TIMING=1 prints per-step wall time (prefill + each decode token) to
    // stderr and a steady-state decode tok/s summary. Off by default so the
    // C-vs-Rust validation harness output stays clean.
    let timing = std::env::var("COLI_TIMING").ok().as_deref() == Some("1");

    // prefill
    let s = prompt.len();
    let mut hidden = vec![0f32; s * d];
    let t_pre = std::time::Instant::now();
    forward(model, kv, provider, prompt, 0, &mut hidden)?;
    if timing {
        let ms = t_pre.elapsed().as_secs_f64() * 1e3;
        eprintln!("[timing] prefill {s} tok: {ms:.1} ms ({:.1} tok/s)", s as f64 / (ms / 1e3));
    }
    let mut logits_us = 0u64;
    let mut logit = {
        let t = std::time::Instant::now();
        let l = logits(model, &hidden[(s - 1) * d..s * d]);
        logits_us += t.elapsed().as_micros() as u64;
        l
    };
    let mut pos = s;
    // Raw hidden at the position that produced `logit` — the MTP head's input.
    let mut hlast = hidden[(s - 1) * d..s * d].to_vec();

    // Speculation budget for this run. Starts at `DRAFT` (0 = off, matching the
    // C's `g_draft`) and is zeroed by the auto-off guard below. With `g == 0`
    // every step below degenerates to exactly the non-speculative loop, which is
    // what makes `DRAFT=0` byte-identical to `DRAFT=n`.
    let mut budget = if model.has_mtp { budget } else { 0 };
    let (mut proposed, mut accepted, mut forwards) = (0u64, 0u64, 0u64);

    let mut decode_ms: Vec<f64> = Vec::with_capacity(n_new);
    let mut emitted = 0usize;
    while emitted < n_new {
        let next = argmax(&logit) as i32;
        let keep_going = on_token(next);
        emitted += 1;
        if model.cfg.stop_ids.contains(&next) {
            break;
        }
        if !keep_going || emitted >= n_new {
            break;
        }

        // --- draft ---------------------------------------------------------
        // Auto-off: drafts that are never accepted are pure overhead (on this
        // engine, extra expert streaming). The C disables them below 10%
        // acceptance after 24 proposals; an int4 MTP head lands there.
        if budget > 0 && proposed >= 24 && accepted * 10 < proposed {
            eprintln!(
                "[MTP] {:.0}% acceptance after {proposed} proposals: drafts disabled",
                100.0 * accepted as f64 / proposed as f64
            );
            budget = 0;
        }
        let drafts = if budget > 0 {
            let dr = crate::mtp::draft(model, kv, provider, next, pos, budget, &hlast)?;
            proposed += dr.len() as u64;
            dr
        } else {
            Vec::new()
        };
        // Clamp to what we still owe the caller and to the cache.
        let mut g = drafts.len().min(n_new - emitted);
        if pos + g + 2 > kv.max_t {
            g = (kv.max_t.saturating_sub(pos + 2)).min(g);
        }

        // --- verify --------------------------------------------------------
        // One forward over [next, drafts...]: position i's logits reveal the
        // TRUE token at i+1, which is what each draft is checked against.
        let mut batch = Vec::with_capacity(1 + g);
        batch.push(next);
        batch.extend_from_slice(&drafts[..g]);
        let sb = batch.len();
        let mut h_all = vec![0f32; sb * d];
        let t = std::time::Instant::now();
        forward(model, kv, provider, &batch, pos, &mut h_all)?;
        forwards += 1;
        let ms = t.elapsed().as_secs_f64() * 1e3;
        if timing {
            eprintln!("[timing] decode tok {}: {ms:.1} ms ({:.2} tok/s)", pos - s, 1e3 / ms);
        }
        decode_ms.push(ms);

        let tl = std::time::Instant::now();
        let los: Vec<Vec<f32>> =
            (0..sb).map(|i| logits(model, &h_all[i * d..(i + 1) * d])).collect();
        logits_us += tl.elapsed().as_micros() as u64;

        // Accept the longest prefix that matches what the model itself would
        // have produced — this is why speculation cannot change the output.
        let mut k = 0usize;
        let mut done = false;
        while k < g && emitted < n_new {
            if argmax(&los[k]) as i32 != drafts[k] {
                break; // rejected: everything after it is stale too
            }
            let keep = on_token(drafts[k]);
            emitted += 1;
            k += 1;
            if model.cfg.stop_ids.contains(&drafts[k - 1]) || !keep {
                done = true;
                break;
            }
        }
        accepted += k as u64;

        // Keep the head's KV in sync with the VERIFIED tokens only.
        if k >= 1 {
            crate::mtp::absorb(model, kv, provider, &drafts[..k], &h_all, pos)?;
        }
        // `hlast` must be the last ACCEPTED position, not the end of the batch:
        // the KV past `pos + k` is stale and will simply be overwritten.
        hlast.copy_from_slice(&h_all[k * d..(k + 1) * d]);
        logit = los[k].clone();
        pos += 1 + k;
        if done {
            break;
        }
    }
    if budget > 0 && proposed > 0 {
        eprintln!(
            "[MTP] {accepted}/{proposed} drafts accepted ({:.0}%), {:.2} tok/forward",
            100.0 * accepted as f64 / proposed as f64,
            if forwards > 0 { emitted as f64 / forwards as f64 } else { 0.0 }
        );
    }
    if timing && !decode_ms.is_empty() {
        // Steady state: drop the first half (cold expert-cache misses) and
        // average the rest.
        let warm = &decode_ms[decode_ms.len() / 2..];
        let mean = warm.iter().sum::<f64>() / warm.len() as f64;
        let min = warm.iter().cloned().fold(f64::INFINITY, f64::min);
        eprintln!(
            "[timing] decode steady-state (last {} of {} tok): mean {mean:.1} ms ({:.2} tok/s), best {min:.1} ms ({:.2} tok/s)",
            warm.len(),
            decode_ms.len(),
            1e3 / mean,
            1e3 / min,
        );
    }
    if profile_on() {
        // Totals across prefill + all decode steps (microseconds -> ms).
        let ms = |a: &AtomicU64| a.load(Ordering::Relaxed) as f64 / 1e3;
        eprintln!(
            "[profile] totals: attn {:.0} ms | mamba {:.0} ms | moe {:.0} ms (of which expert-load {:.0} ms) | dense {:.0} ms | embed {:.0} ms | logits {:.0} ms",
            ms(&ATTN_US),
            ms(&MAMBA_US),
            ms(&MOE_US),
            ms(&LOAD_US),
            ms(&DENSE_US),
            ms(&EMBED_US),
            logits_us as f64 / 1e3,
        );
        eprintln!(
            "[profile] moe-compute breakdown: router {:.0} ms | gather {:.0} ms | gpu-ffn(+sync) {:.0} ms | scatter {:.0} ms | shared {:.0} ms",
            ms(&ROUTER_US),
            ms(&GATHER_US),
            ms(&GPUFFN_US),
            ms(&SCATTER_US),
            ms(&SHARED_US),
        );
        eprintln!(
            "[profile] attn breakdown: proj {:.0} ms | rope+cache {:.0} ms | dsa-indexer {:.0} ms | core {:.0} ms | o-proj {:.0} ms",
            ms(&ATTN_PROJ_US),
            ms(&ATTN_ROPE_US),
            ms(&ATTN_INDEX_US),
            ms(&ATTN_CORE_US),
            ms(&ATTN_OPROJ_US),
        );
        eprintln!(
            "[profile] mamba breakdown: scan {:.0} ms | in/out-proj {:.0} ms | conv {:.0} ms | gated-norm {:.0} ms | other {:.0} ms",
            ms(&MAMBA_SCAN_US),
            ms(&MAMBA_PROJ_US),
            ms(&MAMBA_CONV_US),
            ms(&MAMBA_NORM_US),
            ms(&MAMBA_US)
                - ms(&MAMBA_SCAN_US)
                - ms(&MAMBA_PROJ_US)
                - ms(&MAMBA_CONV_US)
                - ms(&MAMBA_NORM_US),
        );
    }
    Ok(DecodeStats {
        emitted,
        forwards,
        drafts_proposed: proposed,
        drafts_accepted: accepted,
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::quantize::qtensor_from_f32;

    // A tiny but real Nemotron-H config (single Mamba layer) with hand-picked small dims:
    // hidden 4, 2 heads × head_dim 2 (d_inner 4), 1 group × d_state 2, conv_kernel 2
    // (conv_dim = 4 + 2·1·2 = 8, in_proj out = 4 + 8 + 2 = 14).
    fn mamba_cfg() -> Config {
        let json = colibri_json::Json::parse(
            r#"{
            "model_type":"nemotron_h",
            "hidden_size":4, "num_hidden_layers":1,
            "num_attention_heads":2, "num_key_value_heads":1, "head_dim":2,
            "vocab_size":8, "hybrid_override_pattern":"M",
            "n_routed_experts":4, "num_experts_per_tok":2, "moe_intermediate_size":4,
            "moe_latent_size":2, "moe_shared_expert_intermediate_size":4,
            "ssm_state_size":2, "conv_kernel":2, "mamba_num_heads":2, "mamba_head_dim":2,
            "n_groups":1, "chunk_size":2, "mlp_hidden_act":"relu2", "time_step_min":0.001,
            "layer_norm_epsilon":1e-5
        }"#,
        )
        .unwrap();
        Config::from_json(&json).unwrap()
    }

    // Deterministic pseudo-random-ish weight for reproducible tests.
    fn wv(n: usize, seed: usize) -> Vec<f32> {
        (0..n).map(|i| (((i + seed) as f32 * 0.41).sin() * 0.5) + 0.05).collect()
    }

    // A Mamba layer over `mamba_cfg`, with f32 (exact) projection weights so the test
    // exercises the mixer wiring, not quantization error.
    fn mamba_layer(cfg: &Config) -> Layer {
        let d = cfg.hidden as usize; // 4
        let d_inner = cfg.mamba_inter as usize; // 4
        let nh = cfg.mamba_n_heads as usize; // 2
        let conv_dim =
            d_inner + 2 * cfg.mamba_n_groups as usize * cfg.mamba_d_state as usize; // 8
        let proj_out = d_inner + conv_dim + nh; // 14
        let k = cfg.mamba_d_conv as usize; // 2
        let mut l = Layer::default();
        l.mamba_in_proj = Some(qtensor_from_f32(&wv(proj_out * d, 1), proj_out, d, 16));
        l.mamba_out_proj = Some(qtensor_from_f32(&wv(d * d_inner, 2), d, d_inner, 16));
        l.mamba_conv_w = wv(conv_dim * k, 3);
        l.mamba_conv_b = wv(conv_dim, 4);
        l.mamba_a_log = wv(nh, 5);
        l.mamba_d = wv(nh, 6);
        l.mamba_dt_bias = wv(nh, 7);
        l.mamba_norm = wv(d_inner, 8);
        l
    }

    fn mamba_kv(cfg: &Config) -> KvCache {
        let mut kv = KvCache::new(cfg.n_layers as usize, 0, 0, 16);
        kv.enable_mamba2(cfg);
        kv
    }

    /// The mixer's whole point: a single-shot prefill over `S` tokens must equal a run of
    /// `S` one-token decode steps carrying the conv + ssm state (mirrors the scan-level
    /// test in `mamba2.rs`, but through in_proj → conv(state) → scan(state) → gated norm →
    /// out_proj). This validates BOTH recurrent caches and the split arithmetic.
    #[test]
    fn mamba2_mixer_prefill_equals_stepwise_decode() {
        let cfg = mamba_cfg();
        let l = mamba_layer(&cfg);
        let d = cfg.hidden as usize;
        let s = 3usize;
        let x = wv(s * d, 100);

        // Prefill: all S tokens in one call, fresh state.
        let mut kv_full = mamba_kv(&cfg);
        let mut out_full = vec![0f32; s * d];
        mamba2_mixer(&cfg, &l, &mut kv_full, 0, &x, s, &mut out_full);

        // Decode: one token per call, carrying state across calls.
        let mut kv_step = mamba_kv(&cfg);
        let mut out_step = vec![0f32; s * d];
        for t in 0..s {
            let mut o = vec![0f32; d];
            mamba2_mixer(&cfg, &l, &mut kv_step, 0, &x[t * d..(t + 1) * d], 1, &mut o);
            out_step[t * d..(t + 1) * d].copy_from_slice(&o);
        }

        for i in 0..s * d {
            assert!(
                (out_full[i] - out_step[i]).abs() < 1e-5,
                "mismatch at {i}: prefill {} vs stepwise {}",
                out_full[i],
                out_step[i]
            );
        }
        // Sanity: the mixer actually did something (not all zeros).
        assert!(out_full.iter().any(|v| v.abs() > 1e-6), "mixer produced all-zero output");
    }

    /// A fresh mixer with zero state must reproduce the stateless primitives directly:
    /// conv with left zero-pad ([`causal_conv1d_silu`]) then the scan, gated norm, out_proj.
    /// Guards the split offsets (gate|hBC|dt and h|B|C) against a silent transposition.
    #[test]
    fn mamba2_mixer_matches_stateless_primitives_at_sequence_start() {
        let cfg = mamba_cfg();
        let l = mamba_layer(&cfg);
        let d = cfg.hidden as usize;
        let d_inner = cfg.mamba_inter as usize;
        let ng = cfg.mamba_n_groups as usize;
        let ds = cfg.mamba_d_state as usize;
        let nh = cfg.mamba_n_heads as usize;
        let conv_dim = d_inner + 2 * ng * ds;
        let proj_out = d_inner + conv_dim + nh;
        let k = cfg.mamba_d_conv as usize;
        let gn = ng * ds;
        let s = 2usize;
        let x = wv(s * d, 200);

        // Reference: replay the mixer math with the standalone primitives (zero state).
        let mut proj = vec![0f32; s * proj_out];
        matmul_qt(&mut proj, &x, l.mamba_in_proj.as_ref().unwrap(), s);
        let (mut gate, mut hbc, mut dt) =
            (vec![0f32; s * d_inner], vec![0f32; s * conv_dim], vec![0f32; s * nh]);
        for t in 0..s {
            let b = t * proj_out;
            gate[t * d_inner..(t + 1) * d_inner].copy_from_slice(&proj[b..b + d_inner]);
            hbc[t * conv_dim..(t + 1) * conv_dim]
                .copy_from_slice(&proj[b + d_inner..b + d_inner + conv_dim]);
            dt[t * nh..(t + 1) * nh].copy_from_slice(&proj[b + d_inner + conv_dim..b + proj_out]);
        }
        let conv = causal_conv1d_silu(&hbc, &l.mamba_conv_w, &l.mamba_conv_b, s, conv_dim, k);
        let (mut h, mut bb, mut cc) =
            (vec![0f32; s * d_inner], vec![0f32; s * gn], vec![0f32; s * gn]);
        for t in 0..s {
            let b = t * conv_dim;
            h[t * d_inner..(t + 1) * d_inner].copy_from_slice(&conv[b..b + d_inner]);
            bb[t * gn..(t + 1) * gn].copy_from_slice(&conv[b + d_inner..b + d_inner + gn]);
            cc[t * gn..(t + 1) * gn].copy_from_slice(&conv[b + d_inner + gn..b + conv_dim]);
        }
        let dims = MambaDims {
            n_heads: nh,
            head_dim: cfg.mamba_head_dim as usize,
            d_state: ds,
            n_groups: ng,
            dt_min: cfg.mamba_dt_min,
        };
        let mut st = crate::mamba2::SsmState::zeros(nh, cfg.mamba_head_dim as usize, ds);
        let y = selective_scan(
            dims, &mut st, &h, &bb, &cc, &dt, &l.mamba_a_log, &l.mamba_d, &l.mamba_dt_bias, s,
        );
        let yn = gated_rmsnorm(&y, &gate, &l.mamba_norm, s, d_inner, ng, cfg.eps);
        let mut expect = vec![0f32; s * d];
        matmul_qt(&mut expect, &yn, l.mamba_out_proj.as_ref().unwrap(), s);

        let mut kv = mamba_kv(&cfg);
        let mut out = vec![0f32; s * d];
        mamba2_mixer(&cfg, &l, &mut kv, 0, &x, s, &mut out);
        for i in 0..s * d {
            assert!((out[i] - expect[i]).abs() < 1e-5, "at {i}: {} vs {}", out[i], expect[i]);
        }
    }
}
