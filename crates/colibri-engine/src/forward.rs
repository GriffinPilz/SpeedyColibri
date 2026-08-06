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
/// Kimi-K3 KDA (delta-rule linear attention) mixer wall time — the analog of `ATTN_US`
/// for the 69 KDA layers, kept separate so a K3 profile does not blend two mixers with
/// completely different cost models into one number.
static KDA_US: AtomicU64 = AtomicU64::new(0);
/// Kimi-K3 attention-residual wall time: every `apply_attn_res` call, which on K3 is
/// twice per layer plus once at the model level (187 per forward pass). Its own line
/// because it is a cost centre no other arch has.
static ATTNRES_US: AtomicU64 = AtomicU64::new(0);
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
/// Further sub-totals of `MOE_US`, added because the printed breakdown accounted for
/// only ~74% of `MOE_US` on Nemotron-H decode and the missing 26% was the single
/// largest unexplained block in the whole step. A phase timer whose parts do not sum
/// to it is not a breakdown, so the printout now shows the residual explicitly.
///
/// - `MOE_FC_US`  — the two latent projections `fc1`/`fc2` in `nemotron_moe`.
/// - `MOE_SEL_US` — top-K selection and the union/weight matrix build (pure CPU).
/// - `MOE_GRP_US` — the grouped expert call: descriptor wrapping, the fused kernel,
///   and the weighted scatter. On Nemotron this replaces the per-expert loop that
///   `GATHER_US`/`GPUFFN_US`/`SCATTER_US` measure, which is why those read ~0 here.
pub(crate) static MOE_FC_US: AtomicU64 = AtomicU64::new(0);
pub(crate) static MOE_SEL_US: AtomicU64 = AtomicU64::new(0);
pub(crate) static MOE_GRP_US: AtomicU64 = AtomicU64::new(0);
/// Third round of sub-totals, for the same reason as the second: the residual came
/// back. On a MiniMax-M3 512-token prefill `other` was **6743 ms of a 21595 ms**
/// moe-compute phase (20% of it, and 6.2 s more than the July-2026 baseline), which
/// is the whole of a measured 1.358× prefill regression — invisible because nothing
/// timed it. These three cover everything `compute_experts_partial` does outside the
/// gather/ffn/scatter it already measured:
///
/// - `MOE_ALLOC_US`  — the per-call CPU scratch: `out`, and `xg`/`hh` allocated
///   **once per expert per layer** (~7680 visits × 2 × ~256 KB on M3). This repo has
///   already been bitten once by per-call scratch faulting under a full expert cache
///   (pooling it was worth 1.32× prefill), so it gets its own timer rather than
///   sitting in a residual again.
/// - `MOE_EXPGET_US` — `provider.expert()` inside the per-expert loop. Was annotated
///   "cache hit (prefetched); not timed here" — an assumption, now measured.
/// - `MOE_PREP_US`   — building the per-expert row/weight lists, which is
///   O(experts × tokens) over a stride-`ne` read of the router weights.
pub(crate) static MOE_ALLOC_US: AtomicU64 = AtomicU64::new(0);
pub(crate) static MOE_EXPGET_US: AtomicU64 = AtomicU64::new(0);
pub(crate) static MOE_PREP_US: AtomicU64 = AtomicU64::new(0);
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
        std::env::var("DRAFT")
            .ok()
            .and_then(|v| v.parse::<usize>().ok())
            .unwrap_or(0)
            .min(63)
    })
}
/// Largest sequence length that runs the Mamba2 seq scan in BIT-exact (strict
/// nn-order) reduction mode. The MTP verify forward (S = 1 + draft_budget, capped at
/// 64) must match the S==1 decode path to the bit; above this, prefill keeps the faster
/// tree-sum. Default 64 covers any draft budget; `COLI_MAMBA_EXACT_SEQ` overrides.
pub(crate) fn mamba_exact_seq_max() -> usize {
    static N: OnceLock<usize> = OnceLock::new();
    *N.get_or_init(|| {
        std::env::var("COLI_MAMBA_EXACT_SEQ")
            .ok()
            .and_then(|v| v.parse::<usize>().ok())
            .unwrap_or(64)
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
    layer_forward_kind(
        model, kv, provider, l, li, None, x, s, pos_base, nrm, tmp, dsa_sel,
    )
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
    // Kimi-K3 cannot be run one layer at a time. Its layers consume and return
    // (prefix_sum, block_residual) — stack-level state this signature has nowhere to
    // carry — so there is no `x` to update in place. Refuse rather than fall through to
    // the two-sublayer driver below, which would compute a plausible but wrong result
    // (ordinary residual adds, no attention-residual mixing) with no visible symptom.
    // The real driver is [`kimi_forward`], which `forward` dispatches to.
    if cfg.arch == Arch::KimiK3 {
        return Err(io::Error::new(
            io::ErrorKind::Unsupported,
            "Kimi-K3 has no per-layer forward: its layers thread (prefix_sum, \
             block_residual) through the whole stack. Use `forward`, which dispatches \
             to `kimi_forward`.",
        ));
    }
    let d = cfg.hidden as usize;
    // in_ln -> attention -> residual
    for si in 0..s {
        rmsnorm(
            &mut nrm[si * d..(si + 1) * d],
            &x[si * d..(si + 1) * d],
            &l.in_ln,
            cfg.eps,
        );
    }
    if cfg.arch.is_gqa() {
        // MiniMax-M3: grouped-query attention (no MLA latent, no DSA indexer).
        timed(&ATTN_US, || {
            attention_gqa(cfg, l, li, kv, nrm, s, pos_base, tmp)
        });
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
                            cfg,
                            l,
                            li,
                            kv,
                            nrm,
                            s,
                            pos_base,
                            tmp,
                            reused,
                            &cc.sharding,
                            &*cc.transport,
                        );
                    }
                }
            }
            Ok(attention_with(
                cfg,
                l,
                li,
                kv,
                nrm,
                s,
                pos_base,
                tmp,
                AttnCore::Reconstruct,
                reused,
            ))
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
        rmsnorm(
            &mut nrm[si * d..(si + 1) * d],
            &x[si * d..(si + 1) * d],
            &l.post_ln,
            cfg.eps,
        );
    }
    if l.sparse {
        // with_shared only when the model actually has a shared expert (GLM/M3 do;
        // MiniMax-M2 has none — n_shared 0, shared_intermediate_size 0).
        timed(&MOE_US, || {
            moe(cfg, l, li, nrm, s, tmp, cfg.n_shared > 0, provider)
        })?;
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
        rmsnorm(
            &mut nrm[si * d..(si + 1) * d],
            &x[si * d..(si + 1) * d],
            &l.in_ln,
            cfg.eps,
        );
    }
    match kind {
        LayerKind::Mamba => timed(&MAMBA_US, || mamba2_mixer(cfg, l, kv, li, nrm, s, tmp)),
        // NoPE GQA attention (no rotary, no QK-norm — see `attention_gqa`).
        LayerKind::Attn => timed(&ATTN_US, || {
            attention_gqa(cfg, l, li, kv, nrm, s, pos_base, tmp)
        }),
        LayerKind::Moe => timed(&MOE_US, || {
            crate::moe::nemotron_moe(cfg, l, li, nrm, s, tmp, provider)
        })?,
        // Reaching here means a Kimi-K3 model was routed into the Nemotron-H forward
        // path. KDA is a different mixer (linear/delta-rule, not a selective scan), so
        // there is nothing to approximate with — fail loudly rather than run the wrong one.
        LayerKind::Kda => {
            return Err(io::Error::new(
                io::ErrorKind::Unsupported,
                "KDA layer dispatched into the Nemotron-H forward path; Kimi-K3 has no \
                 forward path yet",
            ))
        }
    }
    for j in 0..s * d {
        x[j] += tmp[j];
    }
    trace_state(li, s, pos_base, x);
    Ok(())
}

/// Run the DeepSeek-V4 Compressor (`COLI_DSV4_COMPRESS=1`, default ON).
///
/// The Compressor is how V4 has context past `sliding_window`; without it the model is
/// exact to 128 tokens and wrong beyond. The knob exists to A/B its cost, not because
/// running it is optional for correctness.
fn dsv4_compress_enabled() -> bool {
    static ON: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *ON.get_or_init(|| std::env::var("COLI_DSV4_COMPRESS").ok().as_deref() != Some("0"))
}

/// Run the DeepSeek-V4 Indexer (`COLI_DSV4_INDEXER=0` to disable, default ON).
///
/// Selects which compressed rows a query attends to on the 21 `compress_ratio == 4`
/// layers. Disabling it falls back to attending to ALL closed compressed rows, which is
/// what the ratio-128 layers do anyway — so the two arms are IDENTICAL below 2048 tokens
/// of context and diverge only past it. That makes the knob a real A/B rather than a
/// correctness switch, and it is the cheap way to attribute a long-context change.
fn dsv4_indexer_enabled() -> bool {
    static ON: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *ON.get_or_init(|| std::env::var("COLI_DSV4_INDEXER").ok().as_deref() != Some("0"))
}

/// Tokens per DeepSeek-V4 prefill chunk. `None` = derive it from the KV budget below;
/// `Some(0)` = never chunk; `Some(n)` = a fixed `n`. Set with `COLI_DSV4_CHUNK`.
fn dsv4_chunk_override() -> Option<usize> {
    static N: std::sync::OnceLock<Option<usize>> = std::sync::OnceLock::new();
    *N.get_or_init(|| {
        std::env::var("COLI_DSV4_CHUNK").ok().and_then(|v| v.parse::<usize>().ok())
    })
}

/// The chunk size for a prefill of `s` tokens: **as coarse as the KV budget allows**.
///
/// Chunking is what keeps V4's raw KV constant in context — the ring holds the widest span
/// a SINGLE call reads back, so an unchunked prefill sizes it to the whole prompt. But it
/// is NOT free, and the cost is not the one a byte count would suggest.
///
/// **Measured on the box, 2048-token prompt:** the ring drops 2048 -> 639 rows
/// (193.5 -> 60.4 MB) with identical tokens, and prefill goes **119.6 s -> 168.7 s, 1.41x
/// slower**. Smaller `S` means fewer rows per routed expert, so the expert streaming that
/// dominates V4 prefill is amortised over less work — the same effect already recorded as
/// the MoE-pipelining negative, where chunking destroyed queue depth faster than the
/// overlap recovered it.
///
/// So a fixed token chunk is the wrong rule: at 2048 tokens it buys 133 MB of a 107 GB
/// process — nothing — and charges 41% of prefill for it. The budget makes the trade
/// track what is actually at stake: **do not chunk at all until the retained KV would
/// exceed `COLI_DSV4_KV_BUDGET_MB`, then chunk exactly as coarsely as that allows.** A
/// 2048-token prompt runs in one call as before; a 1M-token prompt, which would otherwise
/// retain ~95 GiB of raw rows and simply not fit, chunks at ~10.8k and retains 1 GiB.
///
/// `COLI_DSV4_CHUNK` overrides with a fixed size (`0` = never chunk), which is what makes
/// the 1.41x above an A/B rather than an assertion.
fn dsv4_prefill_chunk(cfg: &colibri_core::Config, s: usize) -> usize {
    match dsv4_chunk_override() {
        Some(0) => return usize::MAX, // never chunk — the whole prompt is one call
        Some(n) => return n.max(1),
        None => {}
    }
    // The budget lives in `KvCache` beside the reservation that charges for it —
    // `ring_rows` is what `fixed_bytes` bills, so the policy and the accounting cannot drift.
    let row = crate::KvCache::raw_row_bytes(cfg).max(1);
    let budget_rows =
        (crate::KvCache::ring_budget_bytes(cfg) / row).max(cfg.window.max(1) as usize);
    // Under budget: one call, exactly as before chunking existed. This is the common case
    // and it must stay free.
    if s <= budget_rows {
        return usize::MAX;
    }
    budget_rows
}

/// How much the Indexer actually pruned: queries that reached the SCORING path, candidate
/// rows they considered, and rows they kept.
///
/// This exists because the obvious end-to-end A/B cannot tell success from a no-op. The
/// Indexer is designed to drop the LEAST relevant rows, so "tokens unchanged" is the
/// expected result when it works — and also exactly what a silently-skipped Indexer
/// produces. Counting the pruning distinguishes them; nothing else observable does.
static IDX_SCORED: std::sync::atomic::AtomicU64 = std::sync::atomic::AtomicU64::new(0);
static IDX_SEEN: std::sync::atomic::AtomicU64 = std::sync::atomic::AtomicU64::new(0);
static IDX_KEPT: std::sync::atomic::AtomicU64 = std::sync::atomic::AtomicU64::new(0);
static IDX_SKIPPED: std::sync::atomic::AtomicU64 = std::sync::atomic::AtomicU64::new(0);
static IDX_SKIP_MAX: std::sync::atomic::AtomicU64 = std::sync::atomic::AtomicU64::new(0);

/// `(queries scored, candidate rows seen, rows kept)`. All zero means the scoring path
/// never ran — either the context is too short to prune, or it is not wired up.
pub fn dsv4_indexer_stats() -> (u64, u64, u64) {
    let rel = std::sync::atomic::Ordering::Relaxed;
    (IDX_SCORED.load(rel), IDX_SEEN.load(rel), IDX_KEPT.load(rel))
}

/// `(queries that took the keep-everything shortcut, the largest candidate count among
/// them)`. If that maximum ever exceeds `index_topk`, the shortcut fired when it should
/// have scored — which is the difference between "nothing to prune" and "did not prune".
pub fn dsv4_indexer_skips() -> (u64, u64) {
    let rel = std::sync::atomic::Ordering::Relaxed;
    (IDX_SKIPPED.load(rel), IDX_SKIP_MAX.load(rel))
}

/// How many compressed rows the Indexer keeps per query. `index_topk` (512) unless
/// `COLI_DSV4_INDEX_TOPK` overrides it — a TEST knob: at 512 the selection cannot bite
/// until 2048 tokens of context, so verifying it end-to-end costs a ten-minute prefill.
/// Lowering it makes the same code path reachable in seconds.
fn dsv4_index_topk(cfg: &colibri_core::Config) -> usize {
    static OVERRIDE: std::sync::OnceLock<Option<usize>> = std::sync::OnceLock::new();
    let ov = *OVERRIDE.get_or_init(|| {
        std::env::var("COLI_DSV4_INDEX_TOPK").ok().and_then(|v| v.parse::<usize>().ok())
    });
    ov.unwrap_or_else(|| cfg.index_topk.max(0) as usize)
}

/// Skip convolving the conv1d history rows whose outputs the caller discards (default on).
/// `COLI_CONV_HIST=1` computes them anyway, which is the pre-2026-08-03 behaviour.
fn conv_skip_hist() -> bool {
    static ON: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *ON.get_or_init(|| std::env::var("COLI_CONV_HIST").ok().as_deref() != Some("1"))
}

/// Per-layer state hash (`COLI_TRACE_STATE=1`), for finding where two runs of the *same*
/// input first differ.
///
/// Token identity is a poor detector of nondeterminism: it only fires when a bit
/// difference happens to flip an argmax, so a real race shows up as a rare, unreproducible
/// token change with no indication of where it came from. Hashing the residual stream
/// after every layer turns that into an exact (step, layer) coordinate on the first
/// occurrence.
///
/// FNV-1a over the raw f32 bits — bitwise, not approximate: two states that differ in the
/// last ULP must hash differently, or the tool would launder exactly what it is looking for.
pub(crate) fn trace_state(li: usize, s: usize, pos_base: usize, x: &[f32]) {
    static ON: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    if !*ON.get_or_init(|| std::env::var("COLI_TRACE_STATE").ok().as_deref() == Some("1")) {
        return;
    }
    let mut h: u64 = 0xcbf29ce484222325;
    for v in x {
        for b in v.to_bits().to_le_bytes() {
            h = (h ^ b as u64).wrapping_mul(0x100000001b3);
        }
    }
    eprintln!("[trace] pos={pos_base} s={s} layer={li} h={h:016x}");
}

/// Kimi-K3's "attention residual": a softmax attention over the stack's saved states
/// that produces the **input** to a sublayer. Port of the reference `_apply_attn_res`:
///
/// ```text
///   v      = [blocks... ; prefix]           # [S, n_blocks+1, hidden] candidates
///   k      = v * rsqrt(mean(v^2) + eps)     # RMSNorm, WITHOUT its weight
///   sw     = norm * proj                    # the two [hidden] vectors, multiplied
///   scores = (k * sw).sum(-1)               # [S, n_blocks+1]
///   out    = softmax(scores) @ v            # weighted average of the RAW v, not of k
/// ```
///
/// Two things here are easy to get wrong from the tensor shapes alone. `norm` and `proj`
/// are multiplied elementwise into ONE `[hidden]` score vector — not applied in sequence
/// as a norm and then a projection, which is what the `[1, hidden]` shape of
/// `*_res_proj` suggests; that is also why `k` is a weightless RMSNorm (the weight is
/// already folded into `sw`). And the output averages the **raw** `v`, not the
/// normalised `k` — the norm exists only to score.
///
/// Each token mixes only its OWN candidates, so this is `S` independent attentions of
/// width `n_blocks + 1`. Nothing crosses positions, which is why it needs no mask and
/// behaves identically at prefill and decode.
///
/// With `blocks` empty this is the identity — one candidate, `softmax` over one score is
/// 1.0, so `out == prefix`. That is exactly the case the reference's
/// `block_residual.shape[1] > 0` guard skips, so the caller needs no special case.
#[allow(clippy::too_many_arguments)]
fn apply_attn_res(
    prefix: &[f32],
    blocks: &[Vec<f32>],
    norm: &[f32],
    proj: &[f32],
    eps: f32,
    s: usize,
    d: usize,
    out: &mut [f32],
) {
    assert_eq!(norm.len(), d, "attn-res norm is [hidden]");
    assert_eq!(proj.len(), d, "attn-res proj is [1, hidden]");
    let nc = blocks.len() + 1;
    // Shared by every token and candidate, so hoist it out of the loop.
    let sw: Vec<f32> = (0..d).map(|j| norm[j] * proj[j]).collect();
    let mut scores = vec![0f32; nc];
    for i in 0..s {
        let cand = |c: usize| -> &[f32] {
            let v: &[f32] = if c < blocks.len() { &blocks[c] } else { prefix };
            &v[i * d..(i + 1) * d]
        };
        for (c, sc) in scores.iter_mut().enumerate() {
            let v = cand(c);
            let mean_sq = v.iter().map(|&x| x * x).sum::<f32>() / d as f32;
            let r = 1.0 / (mean_sq + eps).sqrt();
            // Elementwise `k * sw` then sum, matching the reference's shape of the
            // computation rather than the algebraically-equal `r * dot(v, sw)`.
            *sc = (0..d).map(|j| (v[j] * r) * sw[j]).sum::<f32>();
        }
        crate::math::softmax(&mut scores);
        let o = &mut out[i * d..(i + 1) * d];
        o.fill(0.0);
        for (c, &p) in scores.iter().enumerate() {
            let v = cand(c);
            for j in 0..d {
                o[j] += p * v[j];
            }
        }
    }
}

/// The stack-level state Kimi-K3 threads between layers, in place of a hidden state.
///
/// This is a separate type because its rules are the part of K3 that is easiest to get
/// wrong and hardest to notice: every alternative below still runs, still produces
/// finite output, and still agrees between prefill and decode. Only a direct test of the
/// state transitions distinguishes them.
///
/// * `prefix` is an accumulator of sublayer outputs, NOT a residual stream — nothing
///   norms it and adds back into it.
/// * At a block boundary it is snapshotted into `blocks` and then **reset**: the next
///   sublayer output replaces it rather than adding to it. That is the reference's
///   `prefix_sum = None`, which `have_prefix == false` represents.
/// * The snapshot happens before the layer's mixer runs, so a candidate is the state as
///   it ENTERED that layer.
struct AttnResState {
    prefix: Vec<f32>,
    blocks: Vec<Vec<f32>>,
    /// `false` means "just snapshotted, not yet restarted" — the reference's `None`.
    have_prefix: bool,
    /// `attn_res_block_size`.
    bs: usize,
}

impl AttnResState {
    fn new(embeddings: Vec<f32>, bs: usize) -> Self {
        assert!(bs > 0, "kimi: attn_res_block_size must be > 0");
        AttnResState {
            prefix: embeddings,
            blocks: Vec::new(),
            have_prefix: true,
            bs,
        }
    }

    /// Snapshot the accumulator if `li` is a block boundary. Call once per layer, after
    /// that layer's attention mix and before its mixer.
    fn maybe_snapshot(&mut self, li: usize) {
        if li % self.bs == 0 {
            self.blocks.push(self.prefix.clone());
            self.have_prefix = false;
        }
    }

    /// Fold one sublayer's output in: added to the accumulator, or — immediately after a
    /// snapshot — becoming it.
    fn accumulate(&mut self, out: &[f32]) {
        if self.have_prefix {
            for (p, &o) in self.prefix.iter_mut().zip(out.iter()) {
                *p += o;
            }
        } else {
            self.prefix.copy_from_slice(out);
            self.have_prefix = true;
        }
    }
}

/// Run the Kimi-K3 stack. K3 does not have an ordinary residual stream, so it cannot
/// reuse [`layer_forward`]; this is the whole-stack driver, a port of the reference
/// `KimiLinearModel.forward` + `KimiDecoderLayer._forward_attn_residual`.
///
/// Two pieces of state thread through every layer, which is the reason this exists:
///
/// * `prefix` — the accumulator. It sums sublayer outputs, and **resets** at each block
///   boundary (the reference sets `prefix_sum = None` there and the next sublayer output
///   becomes the new accumulator, rather than being added to the old one). `have_prefix`
///   is that `None`.
/// * `blocks` — the candidate set. Every `attn_res_block_size`-th layer snapshots the
///   accumulator **as it entered the layer** into it. On K3 that is layers 0, 12, ..., 84
///   → 8 candidates by the end.
///
/// Each sublayer's input is [`apply_attn_res`] over `(prefix, blocks)`; the sublayer's
/// output goes back into `prefix`. So a layer reads a mix of the whole stack's history
/// and contributes to a running sum — the norm→mixer→add shape of an ordinary
/// transformer is not what happens here.
///
/// Cost note: `blocks` is `n_blocks * S * hidden` f32 and lives for the whole pass —
/// **~939 MB at a 4096-token prefill** on the real geometry (8 x 4096 x 7168 x 4), on
/// top of the KV cache and the expert cache. It is 229 KB at decode (`S == 1`), so this
/// only bites prefill. No capacity estimate for K3 has counted it yet.
///
/// `hidden_out` is the RAW hidden state — the model-level attention residual is applied
/// here, but `final_norm` is not, matching [`forward`] and what [`logits`] expects.
pub fn kimi_forward<P: ExpertProvider>(
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

    let mut embeddings = vec![0f32; s * d];
    timed(&EMBED_US, || {
        for (i, &tok) in ids.iter().enumerate() {
            embed_row(
                &model.embed,
                tok as usize,
                &mut embeddings[i * d..(i + 1) * d],
            );
        }
    });
    // `from_json_kimi` range-checks `attn_res_block_size`, so it cannot be 0 here.
    let mut st = AttnResState::new(embeddings, cfg.attn_res_block_size as usize);

    // COLI_DEBUG_ACT=1: the same localisation aid `forward` has, which this path would
    // otherwise skip entirely. It reports the ACCUMULATOR, not a hidden state — K3 has no
    // hidden state — plus the candidate count, because a `prefix` norm is only meaningful
    // alongside how many blocks the next mix will average over, and a reset shows up as a
    // sudden drop that is expected at each boundary rather than a sign of divergence.
    let dbg_act = std::env::var("COLI_DEBUG_ACT").ok().as_deref() == Some("1");
    let pnorm = |tag: &str, p: &[f32], blocks: usize| {
        if s == 0 {
            return;
        }
        let n = |r: &[f32]| r.iter().map(|v| v * v).sum::<f32>().sqrt();
        eprintln!(
            "[act] {tag}: |prefix[0]|={:.4} |prefix[{}]|={:.4} blocks={blocks} p[0][..4]={:?}",
            n(&p[..d]),
            s - 1,
            n(&p[(s - 1) * d..s * d]),
            &p[..4.min(d)]
        );
    };
    if dbg_act {
        pnorm("embed", &st.prefix, st.blocks.len());
    }

    let mut h = vec![0f32; s * d]; // sublayer input, after the attn-res mix
    let mut nrm = vec![0f32; s * d];
    let mut tmp = vec![0f32; s * d];

    for (li, l) in model.layers.iter().enumerate() {
        // ---- attention sublayer -------------------------------------------
        timed(&ATTNRES_US, || {
            apply_attn_res(
                &st.prefix,
                &st.blocks,
                &l.attn_res_norm,
                &l.attn_res_proj,
                cfg.eps,
                s,
                d,
                &mut h,
            )
        });
        // After that mix and before the mixer, so a candidate is the state as it
        // ENTERED this layer — the reference's ordering exactly.
        st.maybe_snapshot(li);
        for si in 0..s {
            rmsnorm(
                &mut nrm[si * d..(si + 1) * d],
                &h[si * d..(si + 1) * d],
                &l.in_ln,
                cfg.eps,
            );
        }
        match cfg.layer_kind[li] {
            LayerKind::Kda => timed(&KDA_US, || {
                crate::kda::kda_mixer(cfg, l, kv, li, &nrm, s, &mut tmp)
            }),
            // Gated MLA. K3 ships no DSA indexer, so `attention_with` finds `ix_wk`
            // absent and runs dense — no selection to thread between layers.
            _ => {
                timed(&ATTN_US, || {
                    attention_with(
                        cfg,
                        l,
                        li,
                        kv,
                        &nrm,
                        s,
                        pos_base,
                        &mut tmp,
                        AttnCore::Reconstruct,
                        None,
                    )
                });
            }
        }
        st.accumulate(&tmp);

        // ---- FFN sublayer -------------------------------------------------
        // No `blocks.is_empty()` guard: layer 0 is always a block boundary, so by here
        // there is always at least one candidate.
        timed(&ATTNRES_US, || {
            apply_attn_res(
                &st.prefix,
                &st.blocks,
                &l.mlp_res_norm,
                &l.mlp_res_proj,
                cfg.eps,
                s,
                d,
                &mut h,
            )
        });
        for si in 0..s {
            rmsnorm(
                &mut nrm[si * d..(si + 1) * d],
                &h[si * d..(si + 1) * d],
                &l.post_ln,
                cfg.eps,
            );
        }
        if l.sparse {
            timed(&MOE_US, || {
                crate::moe::kimi_moe(cfg, l, li, &nrm, s, &mut tmp, provider)
            })?;
        } else {
            timed(&DENSE_US, || dense_mlp(l, &nrm, s, &mut tmp));
        }
        st.accumulate(&tmp);
        // Every layer, not just the first few: on a 93-layer stack with a reset every 12
        // the interesting failure is usually a drift or a blow-up partway down, and the
        // block boundaries are only legible if you can see all of them.
        if dbg_act {
            pnorm(&format!("layer{li}"), &st.prefix, st.blocks.len());
        }
    }

    // Model-level attention residual, then hand back the RAW state.
    timed(&ATTNRES_US, || {
        apply_attn_res(
            &st.prefix,
            &st.blocks,
            &model.output_attn_res_norm,
            &model.output_attn_res_proj,
            cfg.eps,
            s,
            d,
            hidden_out,
        )
    });
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

/// Whether [`mamba2_mixer`] reuses its scratch across calls (default) or allocates fresh
/// each time (`COLI_MAMBA_SCRATCH=0`, the pre-#96 behaviour). See the call site.
fn mamba_scratch_reuse() -> bool {
    static ON: OnceLock<bool> = OnceLock::new();
    *ON.get_or_init(|| std::env::var("COLI_MAMBA_SCRATCH").ok().as_deref() != Some("0"))
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

    let in_proj = l
        .mamba_in_proj
        .as_ref()
        .expect("mamba layer missing in_proj");
    let out_proj = l
        .mamba_out_proj
        .as_ref()
        .expect("mamba layer missing out_proj");

    // Reused across layers and calls — see `MambaScratch`. Moved out and back rather than
    // held borrowed, so the body below is unchanged apart from the slice types.
    //
    // `COLI_MAMBA_SCRATCH=0` bypasses the reuse and allocates fresh, i.e. the pre-#96
    // behaviour. This exists to A/B the retained buffers: #96 measured them worth ~1.4 s of
    // a warm prefill, but the same PR is where a ~4.7 s shared-expert regression appeared
    // (task #98), and the retained allocation is the only always-on change in it. Keeping
    // the old path reachable is what makes that testable on one binary.
    let mut sc = if mamba_scratch_reuse() {
        MAMBA_SCRATCH.with(|c| c.take())
    } else {
        MambaScratch::default()
    };

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
    // Only rows [hist, aug_len) are wanted; the history rows exist to feed the taps.
    // `COLI_CONV_HIST=1` restores the old "convolve everything, discard the front"
    // behaviour so the two can be A/B'd in ONE binary — comparing two builds would put
    // the arms in different processes and, in practice, different sessions.
    let from = if conv_skip_hist() { hist } else { 0 };
    let conv_aug = timed(&MAMBA_CONV_US, || {
        causal_conv1d_silu(aug, &l.mamba_conv_w, &l.mamba_conv_b, aug_len, conv_dim, kk, from)
    });
    let conv_out = &conv_aug[(hist - from) * conv_dim..]; // [s, conv_dim]
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
                    &mut st.data,
                    &mut yv,
                    h,
                    b,
                    c,
                    &dt_h,
                    &da_h,
                    &l.mamba_d,
                    nh,
                    hd,
                    ds,
                    ng,
                )
                .then_some(yv)
            } else {
                let (dt_h, da_h) =
                    crate::mamba2::seq_head_scalars(dims, dt, &l.mamba_a_log, &l.mamba_dt_bias, s);
                let st = kv.mamba_ssm_mut(layer);
                // Small S (MTP verify / tiny prefills) reduces d_state in strict nn-order so
                // the S>1 logits are BIT-identical to the S==1 decode path — otherwise a
                // near-tie argmax in verify forks the accepted token from DRAFT=0. Large-S
                // prefill keeps the fast tree-sum (the strict serial sum would bottleneck it).
                let exact = s <= mamba_exact_seq_max();
                crate::gpu::try_mamba2_scan_seq(
                    &mut st.data,
                    &mut yv,
                    h,
                    b,
                    c,
                    &dt_h,
                    &da_h,
                    &l.mamba_d,
                    nh,
                    hd,
                    ds,
                    ng,
                    s,
                    exact,
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
    // Under the bypass `sc` is a local that drops here, freeing the buffers exactly as the
    // pre-#96 code did; storing it would resurrect the retention this arm exists to remove.
    if mamba_scratch_reuse() {
        MAMBA_SCRATCH.with(|c| c.set(sc));
    }
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

    // Kimi-K3 threads (prefix_sum, block_residual) through the whole stack instead of
    // carrying one hidden state, so it owns the layer loop rather than plugging into
    // the one below. Counted as a forward step above, like every other arch.
    //
    // Returns BEFORE the exact-experts guard below on purpose: that guard exists for the
    // MTP verify/replay path, and K3 has no MTP head, so forcing the bit-exact per-row
    // gemv over its prefill would only cost speed for a case that cannot arise.
    if cfg.arch == Arch::KimiK3 {
        return kimi_forward(model, kv, provider, ids, pos_base, hidden_out);
    }

    // DeepSeek-V4 likewise owns its loop: Hyper-Connections make the residual stream
    // `[s, hc_mult, hidden]`, so there is no single hidden vector for the shared loop to
    // carry. Gated on `hc_mult` rather than on the arch tag so a V4 variant that ships
    // without HC would take the ordinary path instead of indexing copies it does not have.
    if cfg.arch == Arch::DeepseekV4 && cfg.hc_mult > 0 {
        return dsv4_forward_chunked(
            model,
            kv,
            provider,
            ids,
            pos_base,
            hidden_out,
            dsv4_prefill_chunk(cfg, s),
        );
    }

    // Small multi-token forwards (the MTP verify / replay) run the routed NVFP4 experts on
    // the bit-exact per-row gemv path, so a collided expert (>1 row) matches sequential
    // decode to the bit — pairs with the Mamba `exact` scan gate. Held for this forward's
    // scope; large-S prefill keeps the fast WSMM/WMMA path. See gpu::ExactExpertsGuard.
    #[cfg(feature = "cuda")]
    let _exact_guard = crate::gpu::ExactExpertsGuard::new(s > 1 && s <= mamba_exact_seq_max());

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
        rmsnorm(
            &mut nrm[si * d..(si + 1) * d],
            &x[si * d..(si + 1) * d],
            &l.in_ln,
            cfg.eps,
        );
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
        rmsnorm(
            &mut nrm[si * d..(si + 1) * d],
            &x[si * d..(si + 1) * d],
            &l.post_ln,
            cfg.eps,
        );
    }
    if l.sparse {
        timed(&MOE_US, || {
            moe(cfg, l, li, nrm, n, tmp, cfg.n_shared > 0, provider)
        })?;
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
        eprintln!(
            "[timing] prefill {s} tok: {ms:.1} ms ({:.1} tok/s)",
            s as f64 / (ms / 1e3)
        );
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
    // Nemotron-H carries recurrent Mamba2 state that a rejected draft would corrupt;
    // snapshot it around each verify forward and roll back partial accepts. The env
    // switch (default on) exists only to A/B the fix — COLI_MTP_ROLLBACK=0 reproduces
    // the pre-fix corruption, so a token-identity gate can prove it detects the bug.
    let track_mamba =
        kv.has_mamba() && std::env::var("COLI_MTP_ROLLBACK").map_or(true, |v| v != "0");
    let mut mamba_snap = crate::model::MambaSnapshot::default();

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
        // Speculating over Mamba layers is destructive: the verify forward advances every
        // Mamba2 layer's recurrent state by all `sb` tokens. Snapshot it first so a
        // partial accept can be rolled back to the accepted prefix (attention KV needs no
        // such save — it is position-indexed and overwritten by the next forward).
        if track_mamba && g > 0 {
            kv.snapshot_mamba_into(&mut mamba_snap);
        }
        let t = std::time::Instant::now();
        forward(model, kv, provider, &batch, pos, &mut h_all)?;
        forwards += 1;
        let ms = t.elapsed().as_secs_f64() * 1e3;
        if timing {
            eprintln!(
                "[timing] decode tok {}: {ms:.1} ms ({:.2} tok/s)",
                pos - s,
                1e3 / ms
            );
        }
        decode_ms.push(ms);

        let tl = std::time::Instant::now();
        let los: Vec<Vec<f32>> = (0..sb)
            .map(|i| logits(model, &h_all[i * d..(i + 1) * d]))
            .collect();
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

        // Roll the Mamba recurrent state back over the rejected drafts. The verify forward
        // advanced it by all `1+g` tokens; only `1+k` were accepted. Restore the pre-verify
        // snapshot and replay the accepted prefix `[next, drafts[..k]]` so the state lands
        // exactly at `pos+k`. `k == g` needs no restore — the full advance already IS the
        // accepted state; and if we're stopping (`done`) the state is never read again.
        if track_mamba && k < g && !done {
            kv.restore_mamba_from(&mamba_snap);
            let mut h_replay = vec![0f32; (1 + k) * d];
            forward(model, kv, provider, &batch[..1 + k], pos, &mut h_replay)?;
        }

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
            if forwards > 0 {
                emitted as f64 / forwards as f64
            } else {
                0.0
            }
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
            "[profile] totals: attn {:.0} ms | mamba {:.0} ms | kda {:.0} ms | attn-res {:.0} ms | moe {:.0} ms (of which expert-load {:.0} ms) | dense {:.0} ms | embed {:.0} ms | logits {:.0} ms",
            ms(&ATTN_US),
            ms(&MAMBA_US),
            ms(&KDA_US),
            ms(&ATTNRES_US),
            ms(&MOE_US),
            ms(&LOAD_US),
            ms(&DENSE_US),
            ms(&EMBED_US),
            logits_us as f64 / 1e3,
        );
        // Split `expert-load` into what the reader did and what the cache did. GLM
        // moves 117.6 GB per run at 11.6 GB/s measured on the device — 10.1 s of a
        // 14.3 s expert-load — so ~4 s of that window has no request in flight. This
        // line says which phase owns it instead of leaving it to inference.
        {
            let (setup, drain, post, bytes, njobs) = colibri_safetensors::batch_profile();
            let cache = ms(&LOAD_US) - (setup + drain + post) as f64 / 1e3;
            // `drain` accumulates from EVERY loader thread, and prefill runs a background
            // prefetch-ahead loader concurrently with the foreground while `LOAD_US`
            // brackets only the foreground call. So in prefill `drain` is summed THREAD
            // time and routinely exceeds the phase that contains it — GLM has printed
            // `drain 95754 ms` against `expert-load 41450 ms`, K3 `drain 171216 ms`
            // against a 456 s whole run. When that happens these are not a partition and
            // the derived GB/s is meaningless: K3 reported "7.57 GB/s" on a run the device
            // counters put an order of magnitude lower. Printing a negative remainder and a
            // fabricated rate has produced wrong conclusions more than once, so say which
            // case this is instead. `bytes` and `njobs` stay valid either way — threads
            // read disjoint bytes, and a count is a count.
            if cache < 0.0 {
                eprintln!(
                    "[profile] expert-load breakdown: span-setup {:.0} ms | drain {:.0} ms SUMMED \
                     ACROSS LOADER THREADS (exceeds expert-load {:.0} ms — not a partition, and no \
                     wall-clock rate is derivable from it; use /proc/diskstats) | {:.2} GB in {} jobs \
                     | post {:.0} ms",
                    setup as f64 / 1e3,
                    drain as f64 / 1e3,
                    ms(&LOAD_US),
                    bytes as f64 / 1e9,
                    njobs,
                    post as f64 / 1e3,
                );
            } else {
                eprintln!(
                    "[profile] expert-load breakdown: span-setup {:.0} ms | drain {:.0} ms ({:.2} GB in {} jobs, {:.2} GB/s) | post {:.0} ms | cache+other {:.0} ms",
                    setup as f64 / 1e3,
                    drain as f64 / 1e3,
                    bytes as f64 / 1e9,
                    njobs,
                    if drain > 0 { bytes as f64 / 1e3 / drain as f64 } else { 0.0 },
                    post as f64 / 1e3,
                    cache,
                );
            }
            // Decompose the `cache+other` residual. `build` is what `experts_batch`
            // costs on top of the reader — per-expert construction from the bytes.
            eprintln!(
                "[profile] cache breakdown: miss-filter {:.0} ms | build {:.0} ms | insert {:.0} ms | evict {:.0} ms (select {:.0} + free {:.0})",
                ms(&crate::cache::CACHE_FILTER_US),
                ms(&crate::cache::CACHE_FETCH_US) - (setup + drain + post) as f64 / 1e3,
                ms(&crate::cache::CACHE_INSERT_US),
                ms(&crate::cache::CACHE_EVICT_US),
                ms(&crate::cache::EVICT_SELECT_US),
                ms(&crate::cache::EVICT_DROP_US),
            );
            let (lu, mc, al, spans) = colibri_safetensors::span_profile();
            eprintln!(
                "[profile] span-setup breakdown: {} spans | tensor-lookup {:.0} ms | mincore {:.0} ms | alloc {:.0} ms",
                spans,
                lu as f64 / 1e3,
                mc as f64 / 1e3,
                al as f64 / 1e3,
            );
            // Only populated under COLI_RESIDENCY_PROBE=1 — it costs an exact mincore walk.
            let (psp, pby, pres, pruns, pempty) = colibri_safetensors::partial_profile();
            if psp > 0 {
                eprintln!(
                    "[profile] partial residency: {psp} missed spans ({:.1} GB) | {:.1}% already resident | {} missing runs ({:.0} per span, {:.0} KB each) | {pempty} fully absent",
                    pby as f64 / 1e9,
                    100.0 * pres as f64 / pby.max(1) as f64,
                    pruns,
                    pruns as f64 / psp as f64,
                    (pby - pres) as f64 / 1e3 / pruns.max(1) as f64,
                );
            }
            let (ph, pm, pp, pd, pdb) = colibri_core::pool_profile();
            eprintln!(
                "[profile] buf pool: {ph} hits / {pm} misses | {pp} recycled / {pd} rejected ({:.1} GB re-freed)",
                pdb as f64 / 1e9,
            );
            // Page-locking reports 0% DMA-direct on every model while the hooks are
            // installed and the budget is set, so the failure is somewhere between those
            // two facts. These four separate the candidates: `fail` is the driver refusing,
            // `capped` is the budget ledger, and all-zero means nothing ever reached
            // `pin_alloc` — a pool already populated by the time the hook arrived.
            // Charge the buffer pool to Class::ReadBuf before reading the peaks, so the
            // line below reports it. It was committed nowhere, which is why `readbuf` read
            // 0.0 GB on every model.
            crate::ram::set_usage(crate::ram::Class::ReadBuf, colibri_core::pool_live_bytes());
            // Peak per class, which is what a reserve has to cover. RUNTIME_RESERVE is a
            // flat 10 GB standing in for Scratch + ReadBuf + the CUDA context; these are the
            // measured numbers it should be derived from.
            if let Some(m) = crate::ram::manager() {
                use crate::ram::Class::*;
                eprintln!(
                    "[profile] ram peak: dense {:.1} | experts {:.1} | kv {:.1} | scratch {:.1} \
                     | readbuf {:.1} GB (ceiling {:.1})",
                    m.peak_in(Dense) as f64 / 1e9,
                    m.peak_in(Experts) as f64 / 1e9,
                    m.peak_in(Kv) as f64 / 1e9,
                    m.peak_in(Scratch) as f64 / 1e9,
                    m.peak_in(ReadBuf) as f64 / 1e9,
                    m.ceiling() as f64 / 1e9,
                );
            }
            // The CUDA context is the third term in that reserve. On GB10 "VRAM" is the same
            // LPDDR5X pool as the heap, so every byte here is real RAM. It used to be
            // invisible — `Class::Scratch` measured 0.0 on all five models because its only
            // charge site is a prediction on the grouped NVFP4 path, which does not fire on
            // a plain `gen`. The monitor tick now charges this number every 100 ms, so
            // `scratch` above should equal this line.
            //
            // Kept as a separate print precisely so the two can be COMPARED: a mismatch
            // means gpu.rs's prediction won the last tick, which is the only other writer.
            #[cfg(feature = "cuda")]
            eprintln!(
                "[profile] cuda scratch: {:.2} GB — real LPDDR5X, charged to Class::Scratch \
                 (should match `scratch` above; if not, gpu.rs's prediction raced this tick)",
                colibri_backend::cuda::scratch_bytes() as f64 / 1e9,
            );
            let (pok, pfail, pbytes, pcap) = colibri_core::quant::pin_profile();
            eprintln!(
                "[profile] page-lock: {pok} ok / {pfail} failed / {pcap} capped | {:.1} GB locked",
                pbytes as f64 / 1e9,
            );
            // Splits `expert-get` in the moe breakdown below. Same-count misses got ~9×
            // more expensive between two builds, so the question is which third: waiting
            // on the cache lock, the inner provider's load, or insert+evict.
            let (flock, fload, fins) = crate::cache::fetch_profile();
            eprintln!(
                "[profile] expert fetch: lock-wait {:.0} ms | inner-load {:.0} ms | insert+evict {:.0} ms",
                flock as f64 / 1e3,
                fload as f64 / 1e3,
                fins as f64 / 1e3,
            );
            let (calls, threads, spawn) = colibri_safetensors::batch_pool_profile();
            eprintln!(
                "[profile] drain pool: {} batches, {} OS threads created ({:.0}/batch) | spawn-issue {:.0} ms",
                calls,
                threads,
                if calls > 0 { threads as f64 / calls as f64 } else { 0.0 },
                spawn as f64 / 1e3,
            );
        }
        // Every sub-total of MOE_US, plus the residual. `other` is what no timer
        // claims — keep it printed even when it is small, because the only reason
        // this line exists is that a 26% residual sat here unnoticed.
        let moe_parts = ms(&ROUTER_US)
            + ms(&GATHER_US)
            + ms(&GPUFFN_US)
            + ms(&SCATTER_US)
            + ms(&SHARED_US)
            + ms(&LOAD_US)
            + ms(&MOE_FC_US)
            + ms(&MOE_SEL_US)
            + ms(&MOE_GRP_US)
            + ms(&MOE_ALLOC_US)
            + ms(&MOE_EXPGET_US)
            + ms(&MOE_PREP_US);
        eprintln!(
            "[profile] moe-compute breakdown: router {:.0} ms | select {:.0} ms | fc1+fc2 {:.0} ms | group {:.0} ms | prep {:.0} ms | expert-get {:.0} ms | alloc {:.0} ms | gather {:.0} ms | gpu-ffn(+sync) {:.0} ms | scatter {:.0} ms | shared {:.0} ms | other {:.0} ms",
            ms(&ROUTER_US),
            ms(&MOE_SEL_US),
            ms(&MOE_FC_US),
            ms(&MOE_GRP_US),
            ms(&MOE_PREP_US),
            ms(&MOE_EXPGET_US),
            ms(&MOE_ALLOC_US),
            ms(&GATHER_US),
            ms(&GPUFFN_US),
            ms(&SCATTER_US),
            ms(&SHARED_US),
            (ms(&MOE_US) - moe_parts).max(0.0),
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
        (0..n)
            .map(|i| (((i + seed) as f32 * 0.41).sin() * 0.5) + 0.05)
            .collect()
    }

    // A Mamba layer over `mamba_cfg`, with f32 (exact) projection weights so the test
    // exercises the mixer wiring, not quantization error.
    fn mamba_layer(cfg: &Config) -> Layer {
        let d = cfg.hidden as usize; // 4
        let d_inner = cfg.mamba_inter as usize; // 4
        let nh = cfg.mamba_n_heads as usize; // 2
        let conv_dim = d_inner + 2 * cfg.mamba_n_groups as usize * cfg.mamba_d_state as usize; // 8
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
        assert!(
            out_full.iter().any(|v| v.abs() > 1e-6),
            "mixer produced all-zero output"
        );
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
        let (mut gate, mut hbc, mut dt) = (
            vec![0f32; s * d_inner],
            vec![0f32; s * conv_dim],
            vec![0f32; s * nh],
        );
        for t in 0..s {
            let b = t * proj_out;
            gate[t * d_inner..(t + 1) * d_inner].copy_from_slice(&proj[b..b + d_inner]);
            hbc[t * conv_dim..(t + 1) * conv_dim]
                .copy_from_slice(&proj[b + d_inner..b + d_inner + conv_dim]);
            dt[t * nh..(t + 1) * nh].copy_from_slice(&proj[b + d_inner + conv_dim..b + proj_out]);
        }
        // Reference path: no history rows prepended, so every output row is wanted.
        let conv = causal_conv1d_silu(&hbc, &l.mamba_conv_w, &l.mamba_conv_b, s, conv_dim, k, 0);
        let (mut h, mut bb, mut cc) = (
            vec![0f32; s * d_inner],
            vec![0f32; s * gn],
            vec![0f32; s * gn],
        );
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
            dims,
            &mut st,
            &h,
            &bb,
            &cc,
            &dt,
            &l.mamba_a_log,
            &l.mamba_d,
            &l.mamba_dt_bias,
            s,
        );
        let yn = gated_rmsnorm(&y, &gate, &l.mamba_norm, s, d_inner, ng, cfg.eps);
        let mut expect = vec![0f32; s * d];
        matmul_qt(&mut expect, &yn, l.mamba_out_proj.as_ref().unwrap(), s);

        let mut kv = mamba_kv(&cfg);
        let mut out = vec![0f32; s * d];
        mamba2_mixer(&cfg, &l, &mut kv, 0, &x, s, &mut out);
        for i in 0..s * d {
            assert!(
                (out[i] - expect[i]).abs() < 1e-5,
                "at {i}: {} vs {}",
                out[i],
                expect[i]
            );
        }
    }

    // ---- Kimi-K3 attention residuals ------------------------------------------------

    /// The fixture the reference transcription below was evaluated on.
    #[allow(clippy::type_complexity)]
    fn attn_res_fixture() -> (Vec<f32>, Vec<Vec<f32>>, Vec<f32>, Vec<f32>) {
        let prefix = vec![0.1, -0.2, 0.3, 0.5, 1.0, 0.5, -0.5, 0.25];
        let blocks = vec![
            vec![0.2, 0.1, -0.1, 0.4, -0.3, 0.2, 0.6, 0.1],
            vec![-0.5, 0.3, 0.2, 0.1, 0.4, -0.4, 0.2, 0.3],
        ];
        let norm = vec![1.5, 0.5, 2.0, 1.0];
        let proj = vec![0.3, -0.7, 1.1, 0.2];
        (prefix, blocks, norm, proj)
    }

    /// [`apply_attn_res`] must reproduce the reference `_apply_attn_res` numerically.
    ///
    /// The expected values are NOT a second Rust implementation — re-implementing it
    /// here would just re-encode whatever I understood the formula to be. They were
    /// produced by evaluating a line-by-line transcription of the reference Python on
    /// this fixture, so the test fails if my *reading* of the reference is wrong, not
    /// only if the Rust is.
    #[test]
    fn attn_res_matches_the_reference_numerically() {
        let (prefix, blocks, norm, proj) = attn_res_fixture();
        let (s, d) = (2usize, 4usize);
        let mut out = vec![0f32; s * d];
        apply_attn_res(&prefix, &blocks, &norm, &proj, 1e-5, s, d, &mut out);

        let expect = [
            0.055048401,
            -0.14826655,
            0.27699308,
            0.46382627, //
            -0.069_257_09,
            0.013403297,
            0.46532293,
            0.16417568,
        ];
        for i in 0..s * d {
            assert!(
                (out[i] - expect[i]).abs() < 1e-6,
                "at {i}: got {} want {}",
                out[i],
                expect[i]
            );
        }
    }

    /// With no saved blocks there is a single candidate, so the softmax is 1.0 and the
    /// result is the accumulator untouched. This is what lets the driver skip the
    /// reference's `block_residual.shape[1] > 0` guard — if it ever stopped holding,
    /// layer 0 would silently mix something else into its attention input.
    #[test]
    fn attn_res_with_no_blocks_is_the_identity() {
        let (prefix, _, norm, proj) = attn_res_fixture();
        let (s, d) = (2usize, 4usize);
        let mut out = vec![0f32; s * d];
        apply_attn_res(&prefix, &[], &norm, &proj, 1e-5, s, d, &mut out);
        assert_eq!(out, prefix, "one candidate must pass through unchanged");
    }

    /// `norm` and `proj` are MULTIPLIED into one score vector, so swapping them cannot
    /// change the result. A norm-then-project reading — which is what the `[1, hidden]`
    /// shape of `*_res_proj` suggests, and what these fields were documented as for
    /// four commits — is not symmetric, so this pins the distinction that shapes alone
    /// cannot.
    #[test]
    fn attn_res_score_weight_is_symmetric_in_norm_and_proj() {
        let (prefix, blocks, norm, proj) = attn_res_fixture();
        let (s, d) = (2usize, 4usize);
        let mut a = vec![0f32; s * d];
        let mut b = vec![0f32; s * d];
        apply_attn_res(&prefix, &blocks, &norm, &proj, 1e-5, s, d, &mut a);
        apply_attn_res(&prefix, &blocks, &proj, &norm, 1e-5, s, d, &mut b);
        assert_eq!(
            a, b,
            "score weight is norm*proj — swapping them must be a no-op"
        );
    }

    /// Every token mixes only its OWN candidates. Perturbing token 1's state must leave
    /// token 0's output bit-identical; if the mix ever crossed positions it would need a
    /// causal mask, and decode (which sees one token at a time) could not agree with
    /// prefill.
    #[test]
    fn attn_res_does_not_mix_across_positions() {
        let (prefix, blocks, norm, proj) = attn_res_fixture();
        let (s, d) = (2usize, 4usize);
        let mut base = vec![0f32; s * d];
        apply_attn_res(&prefix, &blocks, &norm, &proj, 1e-5, s, d, &mut base);

        let mut perturbed = prefix.clone();
        for v in perturbed[d..].iter_mut() {
            *v += 3.0;
        }
        let mut got = vec![0f32; s * d];
        apply_attn_res(&perturbed, &blocks, &norm, &proj, 1e-5, s, d, &mut got);

        assert_eq!(got[..d], base[..d], "token 0 must not see token 1");
        assert_ne!(got[d..], base[d..], "token 1 must actually have changed");
    }

    /// Drive [`AttnResState`] over a stack the way `kimi_forward` does, with `outs` as
    /// the sublayer outputs in order (two per layer). Returns the final accumulator and
    /// the saved candidates. `d == 1`, so every value is checkable by hand.
    fn run_state(
        bs: usize,
        embed: f32,
        n_layers: usize,
        outs: &[f32],
    ) -> (Vec<f32>, Vec<Vec<f32>>) {
        let mut st = AttnResState::new(vec![embed], bs);
        for li in 0..n_layers {
            st.maybe_snapshot(li);
            st.accumulate(&[outs[2 * li]]);
            st.accumulate(&[outs[2 * li + 1]]);
        }
        (st.prefix, st.blocks)
    }

    /// The accumulator RESETS at a block boundary: the sublayer output that follows a
    /// snapshot replaces it rather than adding to it (the reference's `prefix_sum =
    /// None`). Without the reset everything still runs and prefill still matches decode
    /// — both would just be wrong together — so this is the only thing that catches it.
    #[test]
    fn attn_res_state_resets_the_accumulator_at_a_block_boundary() {
        // bs = 2, so layers 0 and 2 are boundaries.
        // L0: snapshot [10] -> reset -> 1 -> 1+2=3
        // L1: no snapshot   -> 3+4=7 -> 7+5=12
        // L2: snapshot [12] -> reset -> 6 -> 6+7=13
        let (prefix, blocks) = run_state(2, 10.0, 3, &[1.0, 2.0, 4.0, 5.0, 6.0, 7.0]);
        assert_eq!(
            prefix,
            vec![13.0],
            "post-boundary output must REPLACE the accumulator"
        );
        assert_eq!(blocks, vec![vec![10.0], vec![12.0]]);
    }

    /// A snapshot captures the accumulator as it ENTERED the layer — before that layer's
    /// own sublayers contribute. Snapshotting after the mixer instead would save
    /// `10 + 1` here.
    #[test]
    fn attn_res_state_snapshots_the_state_entering_the_layer() {
        let (_, blocks) = run_state(2, 10.0, 1, &[1.0, 2.0]);
        assert_eq!(
            blocks,
            vec![vec![10.0]],
            "candidate must predate this layer's mixer"
        );
    }

    /// Layer 0 is always a boundary, so there is always ≥1 candidate by the time the FFN
    /// sublayer mixes. That is what lets the driver skip the reference's
    /// `block_residual.shape[1] > 0` guard on the second mix.
    #[test]
    fn attn_res_state_always_snapshots_at_layer_zero() {
        for bs in 1..=5 {
            let mut st = AttnResState::new(vec![1.0], bs);
            st.maybe_snapshot(0);
            assert_eq!(st.blocks.len(), 1, "layer 0 must be a boundary at bs={bs}");
        }
    }

    /// The real geometry: 93 layers at `attn_res_block_size` 12 saves 8 candidates
    /// (layers 0, 12, ..., 84). That count sets the transient `block_residual` footprint
    /// — 8 x S x hidden f32, ~939 MB at a 4096-token prefill.
    #[test]
    fn attn_res_state_saves_eight_candidates_on_the_real_geometry() {
        let mut st = AttnResState::new(vec![0.0], 12);
        for li in 0..93 {
            st.maybe_snapshot(li);
            st.accumulate(&[1.0]);
            st.accumulate(&[1.0]);
        }
        assert_eq!(
            st.blocks.len(),
            8,
            "93 layers / block 12 -> boundaries at 0,12,..,84"
        );
    }

    // ---- DeepSeek-V4 raw-KV ring (task #69) -------------------------------

    /// Replay of what `dsv4_attention` does to the raw latent tier, driven through the
    /// same [`dsv4_ring_for`] the production path calls, with one distinguishable value
    /// per (position, lane) standing in for the projected row: size the ring, write this
    /// call's rows, read the span back. Returns the key blocks the attention core would
    /// see — the only thing the ring can change.
    ///
    /// `ring: false` skips the sizing (passing `window` through unchanged, so the SPAN is
    /// identical) and leaves the cache linear. That arm is the reference.
    fn dsv4_replay(window: i32, ring: bool, prompt: usize, gen: usize) -> Vec<Vec<f32>> {
        const W: usize = 6; // stands in for cfg.qk_head
        let mut kv = KvCache::new(1, W, 2, prompt + gen);
        let val = |p: usize, j: usize| (p * 100 + j) as f32;
        let mut seen = Vec::new();
        let mut pos_base = 0usize;
        for s in std::iter::once(prompt).chain(std::iter::repeat(1).take(gen)) {
            let (raw_lo, total, _) = if ring {
                dsv4_ring_for(&mut kv, pos_base, s, window)
            } else {
                dsv4_raw_span(pos_base, s, window)
            };
            for i in 0..s {
                let p = pos_base + i;
                for (j, v) in kv.latent_row_mut(0, p).iter_mut().enumerate() {
                    *v = val(p, j);
                }
            }
            let mut cache = Vec::new();
            kv.extend_latent_rows(0, raw_lo, total, &mut cache);
            seen.push(cache);
            pos_base = total;
        }
        seen
    }

    /// The ring is a storage change, not a model change: every key block the attention
    /// core is handed must be byte-identical with and without it.
    ///
    /// Run across prompt lengths on both sides of the window, because the two regimes
    /// size the ring differently — a from-scratch prefill needs all `S` rows, a decode
    /// step only `window` — and it is the handoff between them that a modulo bug lands in.
    #[test]
    fn the_ring_does_not_change_the_keys_dsv4_attention_sees() {
        const WINDOW: i32 = 8;
        for prompt in [1usize, 3, 7, 8, 9, 20] {
            let linear = dsv4_replay(WINDOW, false, prompt, 40);
            let ringed = dsv4_replay(WINDOW, true, prompt, 40);
            assert_eq!(linear.len(), ringed.len());
            for (step, (a, b)) in linear.iter().zip(&ringed).enumerate() {
                assert_eq!(a, b, "prompt {prompt}, step {step}: ring changed the key block");
            }
        }
    }

    /// ...and it holds a constant number of rows while doing so. Without this the test
    /// above would still pass on a ring sized to the whole sequence, which is the change
    /// not being made.
    #[test]
    fn the_ring_stays_window_sized_across_a_long_generation() {
        const WINDOW: i32 = 8;
        let mut kv = KvCache::new(1, 6, 2, 10_000);
        let mut pos_base = 0usize;
        for s in std::iter::once(4usize).chain(std::iter::repeat(1).take(5_000)) {
            let (_, total, _) = dsv4_ring_for(&mut kv, pos_base, s, WINDOW);
            pos_base = total;
        }
        // 5004 positions generated; the prefill of 4 is under the window, so the widest
        // span any call ever asked for is the window itself.
        assert_eq!(kv.ring(), WINDOW as usize);
    }

    /// A prefill wider than the window sets the ring, because that one call reads all `S`
    /// of its own rows. This is the case where the ring's saving comes from generation
    /// rather than from the prompt — and the reason a long PROMPT still costs full price
    /// until prefill is chunked.
    #[test]
    fn a_prefill_longer_than_the_window_sizes_the_ring_to_the_prompt() {
        let mut kv = KvCache::new(1, 6, 2, 10_000);
        dsv4_ring_for(&mut kv, 0, 500, 8);
        assert_eq!(kv.ring(), 500, "a from-scratch prefill reads every row it wrote");
        // Decode then runs forever inside that same 500 rows — it never grows again.
        let mut pos_base = 500usize;
        for _ in 0..2_000 {
            let (_, t, _) = dsv4_ring_for(&mut kv, pos_base, 1, 8);
            pos_base = t;
        }
        assert_eq!(kv.ring(), 500);
    }

    /// A V4-shaped config for the chunking policy: only `window`, `kv_lora`/`qk_rope` and
    /// the layer count matter to it.
    fn chunk_policy_cfg() -> Config {
        // Every layer full-attention (1-indexed), so all 43 hold KV exactly as V4's do.
        let all: String =
            (1..=43).map(|i| i.to_string()).collect::<Vec<_>>().join(",");
        let json = format!(
            r#"{{"model_type":"kimi_k3","architectures":["KimiK3ForConditionalGeneration"],
                "text_config":{{"hidden_size":64,"num_hidden_layers":43,
                "num_attention_heads":8,"num_key_value_heads":8,"num_experts":8,
                "num_experts_per_token":2,"num_shared_experts":1,"moe_intermediate_size":8,
                "intermediate_size":16,"routed_expert_hidden_size":16,"first_k_dense_replace":0,
                "q_lora_rank":8,"kv_lora_rank":512,"qk_nope_head_dim":448,
                "qk_rope_head_dim":64,"v_head_dim":512,"vocab_size":32,
                "max_position_embeddings":1048576,"rms_norm_eps":1e-5,"rope_theta":10000.0,
                "attn_res_block_size":12,
                "linear_attn_config":{{"head_dim":8,"num_heads":8,"short_conv_kernel_size":2,
                "full_attn_layers":[{all}]}}}}}}"#
        );
        let mut cfg = Config::from_json(&colibri_json::Json::parse(&json).unwrap()).unwrap();
        cfg.window = 128;
        // The real V4 geometry, which is what makes the row size — and so the budget's
        // token count — the one the box actually sees: 43 KV layers at 512 + 64 floats.
        assert_eq!(KvCache::kv_layers(&cfg), 43, "all 43 layers must hold KV, as V4's do");
        cfg
    }

    /// **The policy that decides whether chunking's cost is ever paid.**
    ///
    /// Chunking is NOT free: measured on the box, a 2048-token prompt chunked at 512 kept
    /// 639 raw rows instead of 2048 (193.5 -> 60.4 MB) and took **1.41x longer to prefill**
    /// — smaller `S` amortises the routed-expert streaming over less work. At that size the
    /// memory saved is 133 MB of a 107 GB process, i.e. nothing, so a fixed token chunk
    /// would charge 41% of prefill for no benefit.
    ///
    /// Hence: do not chunk until the retained KV would exceed the budget, then chunk
    /// exactly as coarsely as the budget allows. These assertions are the rule, not the
    /// implementation restated — each names a prompt size and what should happen to it.
    #[test]
    fn v4_prefill_chunks_only_when_the_kv_budget_says_it_must() {
        let cfg = chunk_policy_cfg();
        let row = KvCache::raw_row_bytes(&cfg);
        assert!(row > 0);
        let budget_rows = KvCache::ring_budget_bytes(&cfg) / row;
        assert!(budget_rows > 2048, "1 GiB should hold well past a short prompt");

        // Short and medium prompts: ONE call, exactly as before chunking existed. This is
        // the assertion that keeps the measured 1.41x off the common path.
        for s in [1usize, 512, 2048, budget_rows] {
            assert_eq!(
                dsv4_prefill_chunk(&cfg, s),
                usize::MAX,
                "{s} tokens fits the KV budget and must not be chunked"
            );
        }
        // Past the budget it chunks — and chunks AT the budget, the coarsest size that
        // still bounds the KV, because every token below that costs prefill throughput.
        for s in [budget_rows + 1, 100_000, 1_000_000] {
            assert_eq!(
                dsv4_prefill_chunk(&cfg, s),
                budget_rows,
                "{s} tokens exceeds the budget and must chunk at exactly the budget"
            );
        }
        // And the bound actually holds: retained rows are the chunk plus a window, whatever
        // the context. 1M tokens would otherwise retain ~95 GiB of raw rows.
        let retained = (budget_rows + cfg.window as usize) * row;
        assert!(
            retained < 2 * KvCache::ring_budget_bytes(&cfg),
            "retained {retained} should stay near the {} budget",
            KvCache::ring_budget_bytes(&cfg)
        );
        assert!(
            1_000_000 * row > 40 * retained,
            "unchunked 1M would be {} — the saving must be large or the policy is pointless",
            1_000_000 * row
        );
    }
}

// ===================== DeepSeek-V4-Flash =====================
//
// V4 owns its layer loop for a stronger reason than Kimi-K3 does: its residual stream is
// `[s, hc_mult, hidden]`, not `[s, hidden]`. Hyper-Connections keep `hc_mult` copies and
// mix them with learned weights at every sublayer boundary, so there is no single hidden
// vector to hand to the shared loop.
//
// Block order, transcribed from `Block.forward` in the reference `inference/model.py`:
//
//     residual = x
//     x, post, comb = hc_pre(x, hc_attn_*)   # [hc,d] -> [d]
//     x = attn_norm(x)                       # in_ln
//     x = attn(x)
//     x = hc_post(x, residual, post, comb)   # [d] -> [hc,d]
//     residual = x                           # RE-TAKEN: the POST-ATTENTION stream
//     x, post, comb = hc_pre(x, hc_ffn_*)
//     x = ffn_norm(x)                        # post_ln
//     x = ffn(x)
//     x = hc_post(x, residual, post, comb)
//
// The second `residual = x` is load-bearing. Reusing the block input would typecheck and
// still produce plausible activations.

/// `cos`/`sin` rows for one absolute position.
#[inline]
fn v4_rope_rows<'a>(cos: &'a [f32], sin: &'a [f32], pos: usize, half: usize) -> (&'a [f32], &'a [f32]) {
    (&cos[pos * half..(pos + 1) * half], &sin[pos * half..(pos + 1) * half])
}

/// One DeepSeek-V4 attention sublayer over `xn[s, hidden]` (already `hc_pre`'d and normed)
/// into `out[s, hidden]`.
#[allow(clippy::too_many_arguments)]
/// Which compressed rows each query attends to.
///
/// Two regimes, and the reference picks between them per layer:
/// - `compress_ratio == 128` (and any layer without an Indexer): **all** rows whose window
///   has closed, `t < (p+1)/ratio` — `get_compress_topk_idxs`.
/// - `compress_ratio == 4`: the **Indexer** scores every closed row and keeps the best
///   `index_topk` (512).
///
/// Below 2048 tokens of context the two agree by construction: `end_pos/4 <= 512` means
/// top-k keeps everything, and attention is a set operation so the reordering is
/// invisible. The Indexer only starts DROPPING rows past that, which is exactly the
/// regime it exists for — and a useful gate, since turning it on must not perturb a short
/// generation at all.
#[allow(clippy::too_many_arguments)]
fn dsv4_compress_select(
    cfg: &colibri_core::Config,
    l: &Layer,
    li: usize,
    kv: &mut KvCache,
    xn: &[f32],
    q_lat: &[f32],
    s: usize,
    pos_base: usize,
    ccos: &[f32],
    csin: &[f32],
    comp_avail: &dyn Fn(usize) -> usize,
) -> Vec<Vec<usize>> {
    let all = |kv: &KvCache| -> Vec<Vec<usize>> {
        let _ = kv;
        (0..s).map(|i| (0..comp_avail(pos_base + i)).collect()).collect()
    };
    // Say ONCE, out loud, why the Indexer is not selecting. Its success case is "tokens
    // unchanged" — it drops the rows that scored lowest — so a structural skip is
    // invisible in the output and indistinguishable from it working. That is not
    // hypothetical: an end-to-end A/B at 2400 tokens came back byte-identical because the
    // scoring path never ran at all.
    let ratio = l.comp_ratio as usize;
    // An Indexer exists ONLY where `compress_ratio == 4`. A ratio-128 layer attending to
    // every closed row is `get_compress_topk_idxs`, i.e. correct — so it must not be
    // reported as a fault. Reporting it was worse than saying nothing: it fired first,
    // and a one-shot warning then hid the real reason on the layers that do have one.
    let expected = ratio == 4;
    let inactive = |why: &str| {
        if !expected {
            return;
        }
        static ONCE: std::sync::Once = std::sync::Once::new();
        ONCE.call_once(|| eprintln!("[dsv4] indexer INACTIVE on a ratio-4 layer: {why}"));
    };
    if !dsv4_indexer_enabled() || ratio == 0 {
        return all(kv);
    }
    let (Some(wq_b), Some(wproj)) = (l.idx_wq_b.as_ref(), l.idx_wproj.as_ref()) else {
        inactive("no indexer.wq_b / weights_proj weights on the layer");
        return all(kv);
    };
    let (Some(cwkv), Some(cwgate)) = (l.idx_comp_wkv.as_ref(), l.idx_comp_wgate.as_ref()) else {
        inactive("indexer weights present but its compressor wkv/wgate are missing");
        return all(kv);
    };
    if !kv.icomp_ready() {
        inactive("icomp_init never ran — the indexer has nowhere to put its compressed KV");
        return all(kv);
    }

    let nh = cfg.index_nh as usize;
    let ihd = cfg.index_hd as usize;
    let rd = cfg.qk_rope as usize;
    let half = rd / 2;

    // ---- the Indexer's OWN Compressor (rotate = true) ---------------------
    // Same pooling as the main one, but at `index_head_dim` and finished with a Hadamard
    // rotation + FP4 simulation instead of the FP8 sim on the non-rope dims. Scoring the
    // query against the MAIN compressed rows instead would be silent and wrong: right
    // shapes, different space.
    let cw = cwkv.o as usize;
    let mut ckv = vec![0f32; s * cw];
    let mut csc = vec![0f32; s * cw];
    matmul_qt(&mut ckv, xn, cwkv, s);
    matmul_qt(&mut csc, xn, cwgate, s);
    let mut rows: Vec<Vec<f32>> = Vec::new();
    if let Some(st) = kv.icomp_state_mut(li) {
        if pos_base == 0 {
            rows = crate::dsv4::compress_prefill(&ckv, &csc, &l.idx_comp_ape, s, ratio, ihd, st)
                .chunks_exact(ihd)
                .map(|r| r.to_vec())
                .collect();
        } else {
            for i in 0..s {
                if let Some(r) = crate::dsv4::compress_decode(
                    &ckv[i * cw..(i + 1) * cw],
                    &csc[i * cw..(i + 1) * cw],
                    &l.idx_comp_ape,
                    pos_base + i,
                    st,
                ) {
                    rows.push(r);
                }
            }
        }
    }
    // Snapshot before pushing — see the identical note on the main Compressor above.
    let ibase = kv.icomp_rows(li).len() / ihd;
    for (bi, r) in rows.into_iter().enumerate() {
        let mut nr = vec![0f32; ihd];
        rmsnorm(&mut nr, &r, &l.idx_comp_norm, cfg.eps);
        let b = ibase + bi;
        let p = b * ratio;
        if (p + 1) * half <= ccos.len() {
            let (c, sn) = v4_rope_rows(ccos, csin, p, half);
            crate::dsv4::rope_interleaved(&mut nr, c, sn, rd, false);
        }
        crate::dsv4::hadamard_rotate(&mut nr, ihd);
        crate::dsv4::fp4_act_quant_sim(&mut nr, 32);
        kv.icomp_push(li, &nr);
    }

    let n_ic = kv.icomp_rows(li).len() / ihd;
    if n_ic == 0 {
        inactive("the indexer's compressor has emitted no rows yet");
        return all(kv);
    }

    // ---- query: wq_b on the SHARED q-LoRA bottleneck ----------------------
    // `qr` is `q_norm(wq_a(x))`, the same tensor the main q path consumes — the Indexer
    // re-projects it rather than owning a second LoRA. Note there is NO per-head RMS here:
    // that belongs to the main query only.
    let mut q = vec![0f32; s * nh * ihd];
    matmul_qt(&mut q, q_lat, wq_b, s);
    for i in 0..s {
        let (c, sn) = v4_rope_rows(ccos, csin, pos_base + i, half);
        for hh in 0..nh {
            let b = (i * nh + hh) * ihd;
            crate::dsv4::rope_interleaved(&mut q[b..b + ihd], c, sn, rd, false);
            crate::dsv4::hadamard_rotate(&mut q[b..b + ihd], ihd);
            crate::dsv4::fp4_act_quant_sim(&mut q[b..b + ihd], 32);
        }
    }

    // ---- per-head weights, then score -------------------------------------
    let mut w = vec![0f32; s * nh];
    matmul_qt(&mut w, xn, wproj, s);
    let wscale = (ihd as f32).powf(-0.5) * (nh as f32).powf(-0.5);
    for v in w.iter_mut() {
        *v *= wscale;
    }

    let ikv = kv.icomp_rows(li);
    // `COLI_DSV4_INDEX_TOPK` lowers the cap so selection engages at a context short enough
    // to iterate on. At the real 512 it takes >2048 tokens — a ~10 minute prefill — to
    // exercise the scoring path at all, which is far too slow a loop to debug against.
    let topk = dsv4_index_topk(cfg);
    let mut out: Vec<Vec<usize>> = Vec::with_capacity(s);
    let mut score: Vec<f32> = Vec::new();
    for i in 0..s {
        let avail = comp_avail(pos_base + i).min(n_ic);
        if avail == 0 {
            out.push(Vec::new());
            continue;
        }
        // Everything fits: skip the scoring entirely. This is the common case (context
        // under 2048) and it keeps the Indexer from costing anything where it cannot
        // change the answer.
        if topk == 0 || avail <= topk {
            let rel = std::sync::atomic::Ordering::Relaxed;
            IDX_SKIPPED.fetch_add(1, rel);
            IDX_SKIP_MAX.fetch_max(avail as u64, rel);
            out.push((0..avail).collect());
            continue;
        }
        score.clear();
        for t in 0..avail {
            let kr = &ikv[t * ihd..(t + 1) * ihd];
            let mut acc = 0f32;
            for hh in 0..nh {
                let qv = &q[(i * nh + hh) * ihd..(i * nh + hh) * ihd + ihd];
                let mut d = 0f32;
                for (a, b) in qv.iter().zip(kr) {
                    d += a * b;
                }
                // relu FIRST, then the per-head weight, then sum over heads.
                acc += d.max(0.0) * w[i * nh + hh];
            }
            score.push(acc);
        }
        let mut order: Vec<usize> = (0..avail).collect();
        order.sort_unstable_by(|&a, &b| {
            score[b].partial_cmp(&score[a]).unwrap_or(std::cmp::Ordering::Equal).then(a.cmp(&b))
        });
        order.truncate(topk);
        let rel = std::sync::atomic::Ordering::Relaxed;
        IDX_SCORED.fetch_add(1, rel);
        IDX_SEEN.fetch_add(avail as u64, rel);
        IDX_KEPT.fetch_add(order.len() as u64, rel);
        // Attention is order-invariant, but sorting keeps the index list deterministic
        // and makes an A/B against the all-rows arm diffable.
        order.sort_unstable();
        out.push(order);
    }
    out
}

/// The raw-KV span a DeepSeek-V4 attention call at `pos_base` with `s` tokens touches,
/// as `(raw_lo, total, win)`. `window` is `cfg.window`; `0` means no sliding window, and
/// `win` is then the whole sequence.
///
/// `raw_lo` is the earliest raw position any query in THIS call can reach: query 0 sits
/// at `pos_base` and its window opens `win-1` earlier. For decode that is exactly `win`
/// rows; for a from-scratch prefill it is the whole span, which is what the reference
/// passes (`kv` = all `seqlen` rows at `start_pos == 0`, the ring buffer only after).
///
/// A single function because three things must agree on it — the ring's width, the rows
/// written, and the rows read back — and it has already been wrong once: slicing from
/// `total - win` on every call left a long prefill's early rows out of the cache and
/// underflowed the causal offset `pos_base - raw_from` on `usize`. That was the panic on
/// any prompt over 128 tokens.
pub(crate) fn dsv4_raw_span(pos_base: usize, s: usize, window: i32) -> (usize, usize, usize) {
    let total = pos_base + s;
    let win = if window > 0 { window as usize } else { total };
    ((pos_base + 1) - win.min(pos_base + 1), total, win)
}

/// [`dsv4_raw_span`], with the raw-KV ring widened to hold it.
///
/// The pairing is the whole safety argument: the ring maps position `p` to slot
/// `p % ring`, so a ring narrower than the span being read returns a row belonging to a
/// different position — in bounds, plausible, and wrong. Sizing from the span in the same
/// breath as computing it makes that combination unrepresentable, which is why this is one
/// function rather than two calls a caller has to remember to keep together.
///
/// `window == 0` (every arch but V4) leaves the cache linear: `ring_ensure` is skipped and
/// rows stay indexed by absolute position.
fn dsv4_ring_for(
    kv: &mut KvCache,
    pos_base: usize,
    s: usize,
    window: i32,
) -> (usize, usize, usize) {
    let (raw_lo, total, win) = dsv4_raw_span(pos_base, s, window);
    if window > 0 {
        // Idempotent once wide enough, so decode pays a comparison and returns.
        kv.ring_ensure(total - raw_lo);
    }
    (raw_lo, total, win)
}

#[allow(clippy::too_many_arguments)]
fn dsv4_attention(
    cfg: &colibri_core::Config,
    l: &Layer,
    li: usize,
    kv: &mut KvCache,
    xn: &[f32],
    s: usize,
    pos_base: usize,
    cos: &[f32],
    sin: &[f32],
    ccos: &[f32],
    csin: &[f32],
    out: &mut [f32],
) {
    let h = cfg.n_heads as usize;
    let hd = cfg.qk_head as usize; // 512 — V4's head_dim; K and V are the same latent
    let rd = cfg.qk_rope as usize; // 64
    let ql = cfg.q_lora as usize;
    let half = rd / 2;
    let prof = crate::forward::profile_on();
    let rel = std::sync::atomic::Ordering::Relaxed;
    let t_proj = std::time::Instant::now();

    // ---- q: wq_a -> q_norm -> wq_b -> per-head RMS -> rope -----------------
    let mut q_lat = vec![0f32; s * ql];
    matmul_qt(&mut q_lat, xn, &l.q_a, s);
    let mut row = vec![0f32; ql.max(hd)];
    for i in 0..s {
        rmsnorm(&mut row[..ql], &q_lat[i * ql..(i + 1) * ql], &l.q_a_ln, cfg.eps);
        q_lat[i * ql..(i + 1) * ql].copy_from_slice(&row[..ql]);
    }
    let mut q = vec![0f32; s * h * hd];
    matmul_qt(&mut q, &q_lat, &l.q_b, s);
    // Parameter-free, and NOT the same normalisation as `q_a_ln` above.
    crate::dsv4::per_head_rms(&mut q, hd, cfg.eps);
    for i in 0..s {
        let (c, sn) = v4_rope_rows(cos, sin, pos_base + i, half);
        for hh in 0..h {
            let b = (i * h + hh) * hd;
            crate::dsv4::rope_interleaved(&mut q[b..b + hd], c, sn, rd, false);
        }
    }

    // ---- the raw span this call touches, and the ring that has to hold it --
    //
    // Derived ONCE, up here, because the write just below and the read further down must
    // agree about it — see `dsv4_ring_for`.
    let (raw_lo, total, win) = dsv4_ring_for(kv, pos_base, s, cfg.window);
    let n_raw = total - raw_lo;

    // ---- kv: wkv -> kv_norm -> rope, then into the latent cache ------------
    // ONE `head_dim`-wide latent is both K and V. There is no `kv_b` and no separate value
    // projection, which is exactly why the MLA path cannot be reused here (it asserts a
    // `kv_lora + qk_rope` width that V4 does not have).
    let mut kvt = vec![0f32; s * hd];
    matmul_qt(&mut kvt, xn, &l.kv_a, s);
    for i in 0..s {
        rmsnorm(&mut row[..hd], &kvt[i * hd..(i + 1) * hd], &l.kv_a_ln, cfg.eps);
        let (c, sn) = v4_rope_rows(cos, sin, pos_base + i, half);
        crate::dsv4::rope_interleaved(&mut row[..hd], c, sn, rd, false);
        // QAT match: the reference FP8-simulates the NON-rope dims and deliberately leaves
        // the rope dims alone — `act_quant(kv[..., :-rd], 64, ..., True)` — so the two
        // halves carry different precision by design. Quantising the whole row (or none of
        // it) is the natural simplification and loses that distinction; the rope dims are
        // left in full precision because position is where it matters most.
        crate::dsv4::act_quant_sim(&mut row[..hd - rd], 64);
        kv.latent_row_mut(li, pos_base + i).copy_from_slice(&row[..hd]);
    }

    if prof {
        ATTN_PROJ_US.fetch_add(t_proj.elapsed().as_micros() as u64, rel);
    }

    // ---- Compressor: pool this step's tokens into compressed KV blocks ----
    //
    // These blocks carry ALL context older than the sliding window. Emitted once every
    // `comp_ratio` tokens, normed, and roped with the Compressor's OWN base
    // (`compress_theta` 160000) at the position where each window ENDS.
    if dsv4_compress_enabled() && l.comp_ratio > 0 {
        if let (Some(wkv), Some(wgate)) = (l.comp_wkv.as_ref(), l.comp_wgate.as_ref()) {
            let cw = wkv.o as usize;
            let ratio = l.comp_ratio as usize;
            let mut ckv = vec![0f32; s * cw];
            let mut csc = vec![0f32; s * cw];
            matmul_qt(&mut ckv, xn, wkv, s);
            matmul_qt(&mut csc, xn, wgate, s);
            let mut rows: Vec<Vec<f32>> = Vec::new();
            if let Some(st) = kv.comp_state_mut(li) {
                if pos_base == 0 {
                    rows = crate::dsv4::compress_prefill(&ckv, &csc, &l.comp_ape, s, ratio, hd, st)
                        .chunks_exact(hd)
                        .map(|r| r.to_vec())
                        .collect();
                } else {
                    for i in 0..s {
                        if let Some(r) = crate::dsv4::compress_decode(
                            &ckv[i * cw..(i + 1) * cw],
                            &csc[i * cw..(i + 1) * cw],
                            &l.comp_ape,
                            pos_base + i,
                            st,
                        ) {
                            rows.push(r);
                        }
                    }
                }
            }
            // Snapshot the block count BEFORE pushing: `comp_rows` grows by one each
            // iteration, so reading it inside the loop gave `base + 2*bi` and roped every
            // block after the first at twice its spacing. Only reachable when one call
            // emits several blocks — i.e. prefill, which used to panic before it got here.
            let base = kv.comp_rows(li).len() / hd;
            for (bi, row) in rows.into_iter().enumerate() {
                let mut nr = vec![0f32; hd];
                rmsnorm(&mut nr, &row, &l.comp_norm, cfg.eps);
                // Block `b` covers tokens [b*ratio, (b+1)*ratio); the reference ropes it at
                // the window's END, i.e. `start_pos + 1 - ratio` for the decode emission.
                let b = base + bi;
                let p = b * ratio;
                if (p + 1) * half <= ccos.len() {
                    let (c, sn) = v4_rope_rows(ccos, csin, p, half);
                    crate::dsv4::rope_interleaved(&mut nr, c, sn, rd, false);
                }
                kv.comp_push(li, &nr);
            }
        }
    }

    // ---- attention over the causal span, with the per-head sink -----------
    //
    // The keys are the raw sliding window PLUS the compressed blocks. Compressed rows are
    // prepended: they stand for older positions, and `attention_dsv4` treats the leading
    // rows as the earliest context. Without them attention sees only what fits in the
    // window, which is why context was capped at 128.
    let t_core = std::time::Instant::now();
    let ratio = if dsv4_compress_enabled() { l.comp_ratio as usize } else { 0 };
    let n_comp = if ratio > 0 { kv.comp_rows(li).len() / hd } else { 0 };

    // Key space: raw rows first, then compressed — the reference's layout, where the
    // compressed indices are offset by the raw count. `extend_latent_rows` is the
    // ring-aware read: `raw_lo..total` is one run until the ring wraps and two after,
    // which is why this is an append rather than a borrowed slice.
    let mut cache = Vec::with_capacity((n_raw + n_comp) * hd);
    kv.extend_latent_rows(li, raw_lo, total, &mut cache);
    cache.extend_from_slice(&kv.comp_rows(li)[..n_comp * hd]);

    // A compressed row is visible to position `p` once its window has CLOSED:
    // `t < (p+1)/ratio`. These deliberately OVERLAP the raw window — the reference lets a
    // recent token be attended both raw and compressed rather than excluding either, and
    // the old code excluded them ("would double-count"), which is not what V4 does.
    let comp_avail = |p: usize| if ratio > 0 { ((p + 1) / ratio).min(n_comp) } else { 0 };
    let sel = dsv4_compress_select(cfg, l, li, kv, xn, &q_lat, s, pos_base, ccos, csin, &comp_avail);
    let (idxs, topk) = crate::dsv4::key_indices(win, total, pos_base, s, raw_lo, n_raw, &sel);

    let mut o = vec![0f32; s * h * hd];
    // Reference: `softmax_scale = head_dim ** -0.5`. No YaRN mscale, despite YaRN rope —
    // checked, because DeepSeek's other models DO apply one.
    let scale = (hd as f32).powf(-0.5);
    // GPU first. This core measured 48% of V4 decode as a scalar CPU loop, and `coli gen`
    // reported `0 attention cores` — it had never run on the GPU. The CPU path stays as
    // the fallback and as the exact-arithmetic reference (`COLI_DSV4_GPU_ATTN=0`).
    #[cfg(feature = "cuda")]
    let on_gpu =
        crate::gpu::try_dsv4_sparse_attn(&mut o, &q, &cache, &l.attn_sink, &idxs, s, h, hd, topk);
    #[cfg(not(feature = "cuda"))]
    let on_gpu = false;
    if !on_gpu {
        crate::dsv4::attention_dsv4_sparse(&q, &cache, &l.attn_sink, s, h, hd, &idxs, topk, scale, &mut o);
    }

    if prof {
        ATTN_CORE_US.fetch_add(t_core.elapsed().as_micros() as u64, rel);
    }

    // ---- INVERSE rope on the output ---------------------------------------
    // V is the same latent as K and already carries the forward rotation, so the context
    // inherits it and must be de-rotated. Omitting this leaves every output rotated by its
    // own position: right magnitudes, wrong model.
    for i in 0..s {
        let (c, sn) = v4_rope_rows(cos, sin, pos_base + i, half);
        for hh in 0..h {
            let b = (i * h + hh) * hd;
            crate::dsv4::rope_interleaved(&mut o[b..b + hd], c, sn, rd, true);
        }
    }

    // ---- grouped O-LoRA: block-diagonal `o_a`, then a dense `o_b` ----------
    let t_o = std::time::Instant::now();
    let g = cfg.o_groups.max(1) as usize;
    let rank = cfg.o_lora as usize;
    let dg = h * hd / g;
    let mut mid = vec![0f32; s * g * rank];
    for (gi, wg) in l.o_a_groups.iter().enumerate() {
        // Gather this group's slice of every row, so the group's matmul sees `[s, dg]`.
        let mut xg = vec![0f32; s * dg];
        for i in 0..s {
            xg[i * dg..(i + 1) * dg].copy_from_slice(&o[i * h * hd + gi * dg..i * h * hd + (gi + 1) * dg]);
        }
        let mut yg = vec![0f32; s * rank];
        matmul_qt(&mut yg, &xg, wg, s);
        for i in 0..s {
            mid[i * g * rank + gi * rank..i * g * rank + (gi + 1) * rank]
                .copy_from_slice(&yg[i * rank..(i + 1) * rank]);
        }
    }
    matmul_qt(out, &mid, l.o_b.as_ref().expect("V4 layer missing o_b"), s);
    if prof {
        ATTN_OPROJ_US.fetch_add(t_o.elapsed().as_micros() as u64, rel);
    }
}

/// One DeepSeek-V4 block: two Hyper-Connection sublayers (attention, then MoE).
#[allow(clippy::too_many_arguments)]
fn dsv4_layer_forward<P: ExpertProvider>(
    model: &Model,
    kv: &mut KvCache,
    provider: &P,
    l: &Layer,
    li: usize,
    x: &mut [f32], // [s, hc, d] — the Hyper-Connection stream, in place
    s: usize,
    // Token ids for these rows — needed ONLY by the `n_hash_layers` MoE layers, which
    // select experts from `tid2eid[token_id]` rather than from the router scores.
    ids: &[i32],
    pos_base: usize,
    cos: &[f32],
    sin: &[f32],
    ccos: &[f32],
    csin: &[f32],
    sc: &mut Dsv4Scratch,
) -> io::Result<()> {
    let cfg = &model.cfg;
    let d = cfg.hidden as usize;
    let hc = cfg.hc_mult as usize;
    let iters = cfg.hc_sinkhorn_iters as usize;

    // Two passes with identical shape; only the weights and the sublayer differ.
    for pass in 0..2 {
        let (hc_fn, hc_scale, hc_base, norm) = if pass == 0 {
            (&l.hc_attn_fn, &l.hc_attn_scale, &l.hc_attn_base, &l.in_ln)
        } else {
            (&l.hc_ffn_fn, &l.hc_ffn_scale, &l.hc_ffn_base, &l.post_ln)
        };
        // `residual` is re-read from `x` on BOTH passes — the FFN's residual is the
        // post-attention stream, not the block input.
        sc.residual[..s * hc * d].copy_from_slice(&x[..s * hc * d]);

        for i in 0..s {
            crate::hc::hc_pre(
                &sc.residual[i * hc * d..(i + 1) * hc * d],
                hc_fn,
                hc_scale,
                hc_base,
                hc,
                d,
                cfg.eps,     // norm_eps: the rsqrt inside hc_pre
                cfg.hc_eps,  // hc_eps: floors the Sinkhorn weights — a DIFFERENT epsilon
                iters,
                &mut sc.mixed[i * d..(i + 1) * d],
                &mut sc.post[i],
                &mut sc.comb[i],
            );
        }
        for i in 0..s {
            rmsnorm(
                &mut sc.normed[i * d..(i + 1) * d],
                &sc.mixed[i * d..(i + 1) * d],
                norm,
                cfg.eps,
            );
        }

        if pass == 0 {
            timed(&ATTN_US, || {
                dsv4_attention(cfg, l, li, kv, &sc.normed[..s * d], s, pos_base, cos, sin, ccos, csin, &mut sc.sub[..s * d]);
            });
        } else {
            timed(&MOE_US, || {
                crate::moe::dsv4_moe(cfg, l, li, &sc.normed[..s * d], s, ids, &mut sc.sub[..s * d], provider)
            })?;
        }

        for i in 0..s {
            crate::hc::hc_post(
                &sc.sub[i * d..(i + 1) * d],
                &sc.residual[i * hc * d..(i + 1) * hc * d],
                &sc.post[i],
                &sc.comb[i],
                hc,
                d,
                &mut x[i * hc * d..(i + 1) * hc * d],
            );
        }
    }
    Ok(())
}

/// Buffers reused across V4 layers. Allocating these per layer costs more than the
/// arithmetic at `s = 1`, and the Hyper-Connection stream makes them `hc_mult` times the
/// usual size, so the churn is 4x what it would be on a plain residual.
struct Dsv4Scratch {
    residual: Vec<f32>,
    mixed: Vec<f32>,
    normed: Vec<f32>,
    sub: Vec<f32>,
    post: Vec<[f32; crate::hc::MAX_HC]>,
    comb: Vec<[[f32; crate::hc::MAX_HC]; crate::hc::MAX_HC]>,
}

impl Dsv4Scratch {
    fn new(s: usize, hc: usize, d: usize) -> Self {
        Dsv4Scratch {
            residual: vec![0f32; s * hc * d],
            mixed: vec![0f32; s * d],
            normed: vec![0f32; s * d],
            sub: vec![0f32; s * d],
            post: vec![[0f32; crate::hc::MAX_HC]; s],
            comb: vec![[[0f32; crate::hc::MAX_HC]; crate::hc::MAX_HC]; s],
        }
    }
}

/// DeepSeek-V4 forward. Owns its layer loop because the residual stream is
/// `[s, hc_mult, hidden]`.
///
/// **Context limit.** V4's raw KV cache is a ring buffer of `sliding_window` (128) rows;
/// everything older is reachable only through the Compressor, which is not implemented.
/// This path keeps the full history and attends densely over it, which is EXACTLY what the
/// reference computes while the span fits in the window, and diverges past it. That is a
/// hard edge, so it is reported once rather than left to look like ordinary drift.
/// [`forward`]'s DeepSeek-V4 arm with an explicit chunk size: run `ids` through
/// [`dsv4_forward`] in slices of at most `chunk` tokens.
///
/// **What chunking buys.** `dsv4_forward` is already position-incremental — it is the same
/// entry decode uses at `s == 1` — so this is a loop over it, not a second code path. The
/// win is the raw-KV ring's *width*: [`dsv4_ring_for`] sizes the ring from the widest span
/// a SINGLE call reads back, which for a whole-prompt prefill is the prompt itself. That is
/// why the ring made generation free and left prompts at full price. Chunked, the ring is
/// `window + chunk - 1` rows however long the context — the difference between a 1M-token
/// prompt retaining 1M raw rows and retaining 639.
///
/// It also bounds the transient activations, which matter more here than elsewhere:
/// Hyper-Connections make the residual `[s, hc_mult, hidden]`, so an unchunked long prefill
/// materialises `hc_mult` (4 on the real model) times the usual `s x hidden`.
///
/// **Equivalence is tested, not assumed.** `dsv4_chunked_prefill_matches_one_shot` runs a
/// prompt whole and at six chunk sizes and compares every hidden lane. It has teeth: it
/// fails if the Compressor's per-layer block index restarts on a continuation call — the
/// bug this file's history already records, whose symptom was every block after the first
/// roped at twice its spacing — and if the batch-pooling path is taken past the first call.
///
/// Exposed (rather than reading the env knob inline) because [`dsv4_prefill_chunk`] is a
/// `OnceLock` and so cannot be varied in-process — the same reason
/// [`generate_stream_drafting`] exists alongside [`generate_stream`].
#[allow(clippy::too_many_arguments)]
pub fn dsv4_forward_chunked<P: ExpertProvider>(
    model: &Model,
    kv: &mut KvCache,
    provider: &P,
    ids: &[i32],
    pos_base: usize,
    hidden_out: &mut [f32],
    chunk: usize,
) -> io::Result<()> {
    let d = model.cfg.hidden as usize;
    let s = ids.len();
    let chunk = chunk.max(1);
    if s <= chunk {
        return dsv4_forward(model, kv, provider, ids, pos_base, hidden_out);
    }
    for off in (0..s).step_by(chunk) {
        let n = chunk.min(s - off);
        dsv4_forward(
            model,
            kv,
            provider,
            &ids[off..off + n],
            pos_base + off,
            &mut hidden_out[off * d..(off + n) * d],
        )?;
    }
    Ok(())
}

fn dsv4_forward<P: ExpertProvider>(
    model: &Model,
    kv: &mut KvCache,
    provider: &P,
    ids: &[i32],
    pos_base: usize,
    hidden_out: &mut [f32],
) -> io::Result<()> {
    let cfg = &model.cfg;
    let d = cfg.hidden as usize;
    let hc = cfg.hc_mult as usize;
    let s = ids.len();
    let total = pos_base + s;

    // Only a concern when the Compressor is OFF: with it on, context past the window is
    // carried by the compressed blocks, which is the whole point of the mechanism.
    if cfg.window > 0 && total > cfg.window as usize && !dsv4_compress_enabled() {
        static WARNED: std::sync::Once = std::sync::Once::new();
        WARNED.call_once(|| {
            eprintln!(
                "[dsv4] context {total} exceeds sliding_window {} with the Compressor \
                 DISABLED — everything past {} tokens is missing from attention. \
                 Unset COLI_DSV4_COMPRESS=0 to carry it in compressed blocks.",
                cfg.window, cfg.window
            );
        });
    }

    // TWO rope tables, and which one a layer uses depends on whether it has a Compressor.
    //
    // From `Attention.__init__`: a layer with `compress_ratio != 0` builds its `freqs_cis`
    // with `compress_rope_theta` (160000) AND YaRN enabled; a layer without one uses
    // `rope_theta` (10000) with YaRN **disabled** (`original_seq_len = 0`). That single
    // buffer then ropes the layer's q, its kv, its Compressor blocks and its Indexer alike.
    //
    // This was previously one YaRN table at `theta` for all main attention, with the
    // compress table used only for compressed blocks — wrong on BOTH classes, and wrong on
    // 41 of 43 layers in the direction that matters, since the compress layers are the
    // ones carrying long-range context.
    let (ccos, csin) = crate::dsv4::yarn_rope_tables(
        cfg.qk_rope as usize, total.max(1), cfg.compress_theta, 16.0, 65536, 32.0, 1.0,
    );
    // YaRN off: `original_seq_len = 0` short-circuits the interpolation entirely.
    let (ncos, nsin) = crate::dsv4::yarn_rope_tables(
        cfg.qk_rope as usize, total.max(1), cfg.theta, 16.0, 0, 32.0, 1.0,
    );
    if dsv4_compress_enabled() && kv.comp_rows_len() == 0 {
        let min_ratio = cfg
            .compress_ratios
            .iter()
            .copied()
            .filter(|&r| r > 0)
            .min()
            .unwrap_or(4)
            .max(1) as usize;
        kv.comp_init(
            model.layers.len(),
            &cfg.compress_ratios,
            cfg.qk_head as usize,
            cfg.max_ctx as usize / min_ratio + 2,
        );
        // The Indexer's own compressed rows, on the ratio-4 layers only — that is the
        // exact condition under which the reference constructs an `Indexer`.
        if dsv4_indexer_enabled() && cfg.index_hd > 0 {
            kv.icomp_init(
                model.layers.len(),
                &cfg.compress_ratios,
                4,
                cfg.index_hd as usize,
                cfg.max_ctx as usize / 4 + 2,
            );
        }
    }

    // Embed, then REPEAT into all `hc_mult` copies — the reference's
    // `h.unsqueeze(2).repeat(1, 1, hc_mult, 1)`. Seeding only copy 0 would leave the other
    // three at zero and the Sinkhorn mixing would quietly propagate that.
    let mut x = vec![0f32; s * hc * d];
    timed(&EMBED_US, || {
        let mut e = vec![0f32; d];
        for (i, &tok) in ids.iter().enumerate() {
            embed_row(&model.embed, tok as usize, &mut e);
            for k in 0..hc {
                x[(i * hc + k) * d..(i * hc + k) * d + d].copy_from_slice(&e);
            }
        }
    });

    // COLI_DEBUG_ACT=1: the residual stream's L2 norm at the last position, per layer. The
    // other two drivers have had this; V4 was the only one without, which is precisely why
    // the first chunked-vs-unchunked disagreement could only be argued about rather than
    // measured. Reports copy 0 of `hc_mult` and the whole `[hc, d]` row, because a
    // Hyper-Connection failure typically shows as the copies diverging from one another
    // rather than as any single one blowing up.
    let dbg_act = std::env::var("COLI_DEBUG_ACT").ok().as_deref() == Some("1");
    let pnorm = |tag: &str, x: &[f32]| {
        if !dbg_act || s == 0 {
            return;
        }
        let n = |r: &[f32]| r.iter().map(|v| v * v).sum::<f32>().sqrt();
        let last = (s - 1) * hc * d;
        eprintln!(
            "[act] {tag}: pos={} |copy0|={:.6e} |row|={:.6e}",
            pos_base + s - 1,
            n(&x[last..last + d]),
            n(&x[last..last + hc * d]),
        );
    };
    pnorm("embed", &x);

    let mut sc = Dsv4Scratch::new(s, hc, d);
    for (li, l) in model.layers.iter().enumerate() {
        // A layer WITH a Compressor ropes everything — q, kv, its blocks, its Indexer —
        // with the compress-theta YaRN table; a layer without one uses the plain base
        // table and no YaRN. Read from the config, not from `l.comp_ratio`, so the choice
        // does not silently follow the `COLI_DSV4_COMPRESS` knob: the weights were trained
        // with one table per layer regardless of whether we run the pooling.
        let compressed = cfg.compress_ratios.get(li).copied().unwrap_or(0) > 0;
        let (mcos, msin) = if compressed { (&ccos, &csin) } else { (&ncos, &nsin) };
        dsv4_layer_forward(model, kv, provider, l, li, &mut x, s, ids, pos_base, mcos, msin, &ccos, &csin, &mut sc)?;
        pnorm(&format!("layer{li}"), &x);
        trace_state(li, s, pos_base, &x);
    }

    // Collapse `[hc, d] -> [d]` with the model-level head: a plain sigmoid gate, NO
    // Sinkhorn, and a scalar scale where the per-layer one takes a 3-vector.
    for i in 0..s {
        crate::hc::hc_head(
            &x[i * hc * d..(i + 1) * hc * d],
            &model.hc_head_fn,
            model.hc_head_scale,
            &model.hc_head_base,
            hc,
            d,
            cfg.eps,
            cfg.hc_eps,
            &mut hidden_out[i * d..(i + 1) * d],
        );
    }
    Ok(())
}
