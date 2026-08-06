//! End-to-end DeepSeek-V4-Flash forward pass on a tiny synthetic container.
//!
//! V4 owns its driver (`dsv4_forward`) because its residual stream is `[s, hc_mult, hidden]`
//! rather than `[s, hidden]` — Hyper-Connections keep `hc_mult` copies and mix them at every
//! sublayer boundary — and because its attention keeps a raw sliding window plus a
//! Compressor-pooled tail. This file exercises that driver through the real loader, on a
//! container carrying every V4-specific piece:
//!
//!   * Hyper-Connections at both sublayers and the model head,
//!   * a **ratio-4** layer (Compressor **and** Indexer), a **ratio-2** layer (Compressor
//!     only), and a layer with **neither** — the three classes the real 43-layer stack has,
//!   * the O-LoRA output projection with its row-group split,
//!   * the per-head attention sink,
//!   * all-MoE layers with a shared expert.
//!
//! **What this can and cannot prove.** The weights are synthetic, so nothing here says the
//! outputs match the real checkpoint. What it pins is that the stack loads and runs without
//! NaNs, and — the reason it exists — that **one prefill call agrees with the same tokens
//! fed in chunks**. That is the property chunked prefill needs and the one the raw-KV ring
//! and the Compressor's cross-call carry could each break silently: wrong block positions
//! give plausible tokens, not a crash.
//!
//! The window is deliberately tiny (4) so a 12-token prompt wraps the ring three times and
//! emits several Compressor blocks. A test whose context fits inside the window would pass
//! on a build where neither mechanism runs at all.

use colibri_engine::{forward, load_model_with, KvCache, LoadOptions, ShardsExpertProvider};
use std::fs::File;
use std::io::Write;
use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicU64, Ordering};

const D: usize = 8; // hidden
const NL: usize = 3;
const HC: usize = 2; // hc_mult — 2, not the real 4: same code, a quarter of the arithmetic
const H: usize = 2; // attention heads
const HEAD_DIM: usize = 8; // = kv_lora = qk_head = v_head; ONE latent is both K and V
const QK_ROPE: usize = 4;
const Q_LORA: usize = 4;
const O_LORA: usize = 4;
const O_GROUPS: usize = 2;
const WINDOW: usize = 4;
const IDX_NH: usize = 2;
const IDX_HD: usize = 4;
const IDX_TOPK: usize = 2;

const E: usize = 4;
const TOPK: usize = 2;
const MOE_INTER: usize = 4;
const N_SHARED: usize = 1;
const S_I: usize = MOE_INTER * N_SHARED;
const VOCAB: usize = 10;
const MAX_CTX: usize = 64;

/// Layer 0 has a Compressor **and** an Indexer (ratio 4 is what builds one), layer 1 a
/// Compressor only, layer 2 neither — the three classes of the real stack, in one model.
const RATIOS: [i32; NL] = [4, 2, 0];

/// How far a chunked prefill may differ from a one-shot one, relative — **by chunk size**.
///
/// Measured, not guessed. On this fixture every chunk of **2 or more is bit-identical** to
/// the single call, on the CPU build *and* the CUDA build. The chunking composes exactly:
/// the Compressor's cross-call carry, the ring's `pos % R` mapping and the causal span
/// arithmetic all reproduce the batched result to the bit.
///
/// `chunk == 1` is the exception, and only under CUDA (4.5e-2 observed). `S == 1` selects a
/// different family of GPU kernels — the decode fast paths — so that arm is not comparing
/// two chunkings, it is comparing prefill kernels against decode kernels. It is kept in the
/// sweep because it is the harshest possible split of the Compressor's carry, which is what
/// the test is really for.
///
/// **What this fixture cannot show:** it is 3 layers at hidden 8, too small to dispatch the
/// tiled/WSMM kernels, so every `S >= 2` here takes the same code path. On the real 43-layer
/// model, `S = 128` and `S = 512` DO select different kernels and diverge — measured with
/// `COLI_DEBUG_ACT=1`, starting at 8.7e-6 after layer 0 and plateauing near 5e-3. That is
/// kernel selection, not chunking, and this test is deliberately not the instrument for it.
fn max_chunk_rel_diff(chunk: usize) -> f32 {
    if chunk == 1 && cfg!(feature = "cuda") {
        6e-2
    } else {
        0.0 // bit-identical, and a regression should say so loudly
    }
}

fn mix_width(hc: usize) -> usize {
    (2 + hc) * hc
}

fn temp_dir() -> PathBuf {
    static N: AtomicU64 = AtomicU64::new(0);
    let base = std::env::var("TMPDIR").unwrap_or_else(|_| "/tmp".into());
    let mut p = PathBuf::from(base);
    p.push(format!(
        "colibri-v4-{}-{}",
        std::process::id(),
        N.fetch_add(1, Ordering::Relaxed)
    ));
    std::fs::create_dir_all(&p).unwrap();
    p
}

fn config_json() -> String {
    let ratios = RATIOS.map(|v| v.to_string()).join(",");
    format!(
        r#"{{"model_type":"deepseek_v4","architectures":["DeepseekV4ForCausalLM"],
        "hidden_size":{D},"num_hidden_layers":{NL},"num_attention_heads":{H},
        "num_key_value_heads":1,"head_dim":{HEAD_DIM},"qk_rope_head_dim":{QK_ROPE},
        "q_lora_rank":{Q_LORA},"o_lora_rank":{O_LORA},"o_groups":{O_GROUPS},
        "n_routed_experts":{E},"num_experts_per_tok":{TOPK},
        "moe_intermediate_size":{MOE_INTER},"n_shared_experts":{N_SHARED},
        "vocab_size":{VOCAB},"max_position_embeddings":{MAX_CTX},
        "rms_norm_eps":1e-5,"rope_theta":10000.0,"compress_rope_theta":160000.0,
        "routed_scaling_factor":1.0,"norm_topk_prob":true,"n_group":1,"topk_group":1,
        "sliding_window":{WINDOW},"compress_ratios":[{ratios}],
        "index_topk":{IDX_TOPK},"index_n_heads":{IDX_NH},"index_head_dim":{IDX_HD},
        "hc_mult":{HC},"hc_sinkhorn_iters":4,"hc_eps":1e-6,
        "num_hash_layers":0,"eos_token_id":9}}"#
    )
}

fn tensor_list() -> Vec<(String, usize)> {
    let mw = mix_width(HC);
    let n = HC * D;
    let mut t: Vec<(String, usize)> = vec![
        ("model.embed_tokens.weight".into(), VOCAB * D),
        ("lm_head.weight".into(), VOCAB * D),
        ("model.norm.weight".into(), D),
        // The model-level Hyper-Connection head: a plain sigmoid gate collapsing [hc,d]->[d].
        ("model.hc_head_fn".into(), HC * D),
        ("model.hc_head_base".into(), 1),
        ("model.hc_head_scale".into(), 1),
    ];
    for i in 0..NL {
        let p = |s: &str| format!("model.layers.{i}.{s}");
        t.push((p("input_layernorm.weight"), D));
        t.push((p("post_attention_layernorm.weight"), D));

        // Attention: q LoRA, the single kv latent, the O-LoRA pair, the per-head sink.
        t.push((p("self_attn.q_a_proj.weight"), Q_LORA * D));
        t.push((p("self_attn.q_a_layernorm.weight"), Q_LORA));
        t.push((p("self_attn.q_b_proj.weight"), H * HEAD_DIM * Q_LORA));
        t.push((p("self_attn.kv_a_proj.weight"), HEAD_DIM * D));
        t.push((p("self_attn.kv_a_layernorm.weight"), HEAD_DIM));
        t.push((p("self_attn.o_a_proj.weight"), O_GROUPS * O_LORA * (H * HEAD_DIM / O_GROUPS)));
        t.push((p("self_attn.o_b_proj.weight"), D * O_GROUPS * O_LORA));
        t.push((p("self_attn.attn_sink"), H));

        // Hyper-Connections, one set per sublayer.
        for s in ["hc_attn", "hc_ffn"] {
            t.push((p(&format!("{s}_fn")), mw * n));
            t.push((p(&format!("{s}_base")), mw));
            t.push((p(&format!("{s}_scale")), 3));
        }

        // Compressor, where `compress_ratios` says there is one. `wkv`/`wgate` are twice
        // as wide on a ratio-4 layer (the overlapping-window form), and `ape` is
        // `ratio * width` — the loader checks that product, so it is a real constraint.
        let ratio = RATIOS[i] as usize;
        if ratio > 0 {
            let coff = if ratio == 4 { 2 } else { 1 };
            let w = coff * HEAD_DIM;
            t.push((p("self_attn.compressor.wkv.weight"), w * D));
            t.push((p("self_attn.compressor.wgate.weight"), w * D));
            t.push((p("self_attn.compressor.ape"), ratio * w));
            t.push((p("self_attn.compressor.norm.weight"), HEAD_DIM));
        }
        // Indexer — only on the ratio-4 layer, which is exactly when the reference builds
        // one. Its Compressor is SEPARATE (rotate=true) and lives at `index_head_dim`.
        if ratio == 4 {
            let iw = 2 * IDX_HD;
            t.push((p("self_attn.indexer.wq_b.weight"), IDX_NH * IDX_HD * Q_LORA));
            t.push((p("self_attn.indexer.weights_proj.weight"), IDX_NH * D));
            t.push((p("self_attn.indexer.compressor.wkv.weight"), iw * D));
            t.push((p("self_attn.indexer.compressor.wgate.weight"), iw * D));
            t.push((p("self_attn.indexer.compressor.ape"), 4 * iw));
            t.push((p("self_attn.indexer.compressor.norm.weight"), IDX_HD));
        }

        // Every V4 layer is MoE — no dense layer, no `first_k_dense_replace`.
        t.push((p("mlp.gate.weight"), E * D));
        t.push((p("mlp.gate.bias"), E));
        t.push((p("mlp.shared_experts.gate_proj.weight"), S_I * D));
        t.push((p("mlp.shared_experts.up_proj.weight"), S_I * D));
        t.push((p("mlp.shared_experts.down_proj.weight"), D * S_I));
        for e in 0..E {
            let pe = |s: &str| format!("model.layers.{i}.mlp.experts.{e}.{s}.weight");
            t.push((pe("gate_proj"), MOE_INTER * D));
            t.push((pe("up_proj"), MOE_INTER * D));
            t.push((pe("down_proj"), D * MOE_INTER));
        }
    }
    t
}

fn write_model(dir: &Path) {
    let tensors = tensor_list();
    let mut header = String::from("{");
    let mut off = 0usize;
    let mut payload: Vec<u8> = Vec::new();
    for (idx, (name, numel)) in tensors.iter().enumerate() {
        if idx > 0 {
            header.push(',');
        }
        let nbytes = numel * 4;
        header.push_str(&format!(
            "\"{name}\":{{\"dtype\":\"F32\",\"shape\":[{numel}],\"data_offsets\":[{off},{}]}}",
            off + nbytes
        ));
        off += nbytes;
        let seed: usize = name.bytes().map(|b| b as usize).sum();
        for k in 0..*numel {
            let v = (((k + seed) % 7) as f32 - 3.0) * 0.08;
            payload.extend_from_slice(&v.to_le_bytes());
        }
    }
    header.push('}');
    let hb = header.as_bytes();
    let mut f = File::create(dir.join("model.safetensors")).unwrap();
    f.write_all(&(hb.len() as u64).to_le_bytes()).unwrap();
    f.write_all(hb).unwrap();
    f.write_all(&payload).unwrap();
}

fn tiny_model(dir: &Path) -> colibri_engine::Model {
    std::fs::write(dir.join("config.json"), config_json()).unwrap();
    write_model(dir);
    load_model_with(dir, LoadOptions { dbits: 8, ebits: 8 }).expect("load V4 container")
}

/// The container loads and the whole V4 stack runs: Hyper-Connections, the O-LoRA output,
/// the attention sink, a ratio-4 layer with its Indexer, a ratio-2 layer, a bare layer, and
/// all-MoE FFNs. Output must be finite — a NaN here means one of those pieces is producing
/// garbage the shapes could not catch.
#[test]
fn dsv4_stack_runs_end_to_end() {
    let dir = temp_dir();
    let model = tiny_model(&dir);
    assert_eq!(model.cfg.hc_mult as usize, HC, "Hyper-Connections must be on");
    assert_eq!(model.cfg.window as usize, WINDOW);
    assert_eq!(model.layers.len(), NL);

    let provider = ShardsExpertProvider::new(&model.shards, &model.cfg, 8);
    let prompt: Vec<i32> = (0..12).map(|i| ((i * 3 + 1) % VOCAB as i32) as i32).collect();
    assert!(prompt.len() > 2 * WINDOW, "must exceed the window, or nothing wraps");

    let mut kv = KvCache::for_model(&model, 32);
    let mut hidden = vec![0f32; prompt.len() * D];
    forward(&model, &mut kv, &provider, &prompt, 0, &mut hidden).expect("V4 forward");
    assert!(hidden.iter().all(|v| v.is_finite()), "V4 stack produced a non-finite value");
    assert!(hidden.iter().any(|v| *v != 0.0), "V4 stack produced all zeros");

    // The mechanisms actually RAN. Without these the fixture would pass just as happily on
    // a build where the Compressor never emitted a block and the ring never engaged — and
    // "it loads and the numbers are finite" would be the only thing proved. Each of the
    // three V4 pieces has a success case that looks identical to being skipped.
    assert_eq!(kv.ring(), prompt.len(), "the raw KV ring must be sized by prefill");
    for li in 0..NL {
        let blocks = kv.comp_rows(li).len() / HEAD_DIM;
        if RATIOS[li] > 0 {
            let want = prompt.len() / RATIOS[li] as usize;
            assert_eq!(blocks, want, "layer {li} (ratio {}) must emit blocks", RATIOS[li]);
        } else {
            assert_eq!(blocks, 0, "layer {li} has no Compressor and must emit nothing");
        }
    }
    let (scored, seen, kept) = colibri_engine::forward::dsv4_indexer_stats();
    assert!(scored > 0, "the Indexer never scored — the ratio-4 layer is not exercised");
    assert!(kept <= seen && seen > 0, "indexer stats are degenerate: {kept}/{seen}");
}

/// One prefill call through the production entry point, at an explicit chunk size.
///
/// `dsv4_forward_chunked` rather than a loop of `forward()` calls in the test: the loop is
/// the thing under test, so driving it by hand here would leave the shipped one unexercised
/// — and `COLI_DSV4_CHUNK` is a `OnceLock`, so it cannot be varied in-process.
fn run_chunked(
    model: &colibri_engine::Model,
    provider: &ShardsExpertProvider,
    ids: &[i32],
    chunk: usize,
) -> (Vec<f32>, KvCache) {
    let mut kv = KvCache::for_model(model, 64);
    let mut out = vec![0f32; ids.len() * D];
    colibri_engine::forward::dsv4_forward_chunked(model, &mut kv, provider, ids, 0, &mut out, chunk)
        .expect("V4 chunked forward");
    (out, kv)
}

/// **The property chunked prefill needs.** Feeding a prompt in chunks must produce the same
/// hidden states as feeding it whole.
///
/// Everything V4 carries across calls has to line up for this: the raw-KV ring's `pos % R`
/// mapping, the per-layer Compressor's block accounting (`compress_prefill` on the first
/// call, `compress_decode` afterwards — two different code paths that must agree), the
/// Indexer's separate compressor, and the causal span arithmetic. A chunk boundary that
/// mis-advances a block index does not crash; it returns plausible, wrong numbers.
///
/// Several chunk sizes on purpose: one that divides the prompt, one that does not, one
/// below the window and one above it. A single size can land on a boundary that happens to
/// agree.
#[test]
fn dsv4_chunked_prefill_matches_one_shot() {
    let dir = temp_dir();
    let model = tiny_model(&dir);
    let provider = ShardsExpertProvider::new(&model.shards, &model.cfg, 8);
    let prompt: Vec<i32> = (0..12).map(|i| ((i * 3 + 1) % VOCAB as i32) as i32).collect();

    let (whole, kv_whole) = run_chunked(&model, &provider, &prompt, prompt.len());
    assert!(whole.iter().all(|v| v.is_finite()));

    assert_eq!(kv_whole.ring(), prompt.len(), "unchunked: the ring holds the whole prompt");

    for chunk in [1usize, 2, 3, 5, 6, 8] {
        let (got, kv) = run_chunked(&model, &provider, &prompt, chunk);
        assert_eq!(got.len(), whole.len(), "chunk {chunk}: wrong output length");
        // THE POINT OF THE EXERCISE: the retained raw rows are bounded by the CHUNK, not by
        // the prompt. Stated as a bound rather than an exact width on purpose — the exact
        // value depends on how the last partial chunk falls, and a formula written here
        // would be a second copy of `dsv4_raw_span` free to drift from the real one. What
        // matters is that it cannot grow with context; `dsv4_ring_is_constant_in_context`
        // below pins that directly.
        assert!(
            kv.ring() <= WINDOW + chunk,
            "chunk {chunk}: ring {} exceeds window+chunk",
            kv.ring()
        );
        assert!(
            kv.ring() < prompt.len(),
            "chunk {chunk}: ring {} did not shrink below the prompt",
            kv.ring()
        );
        // The compressed tier must also match — it is what carries context past the
        // window, so equal hidden states with a different block count would mean the two
        // agreed by luck on this prompt and would diverge on a longer one.
        for li in 0..NL {
            assert_eq!(
                kv.comp_rows(li).len(),
                kv_whole.comp_rows(li).len(),
                "chunk {chunk}, layer {li}: different number of compressed blocks"
            );
        }
        let (mut worst, mut at) = (0f32, 0usize);
        for (i, (a, b)) in whole.iter().zip(&got).enumerate() {
            // Relative, with a floor: this container's weights are int8-quantised at
            // hidden 8, so lanes land near zero and an absolute bound would be reporting
            // quantisation grain rather than agreement.
            let r = (a - b).abs() / a.abs().max(b.abs()).max(1e-3);
            if r > worst {
                worst = r;
                at = i;
            }
        }
        eprintln!("[chunk-equiv] chunk={chunk} worst_rel={worst:.3e} at lane {at}");
        assert!(
            worst <= max_chunk_rel_diff(chunk),
            "chunk {chunk}: worst relative difference {worst:.3e} at hidden[{at}] \
             (token {}, lane {}) — {} vs {}",
            at / D,
            at % D,
            whole[at],
            got[at]
        );
    }
}

/// **The 1M claim, in miniature.** At a fixed chunk size the retained raw rows must not
/// grow with the prompt.
///
/// This is the property the whole exercise is for, and it is the one an exact-width
/// assertion cannot express: a formula can be satisfied by a ring that still scales, as
/// long as the formula scales too. Three prompt lengths, one chunk, one ring width.
///
/// Also checks the unchunked arm still grows — otherwise the test would pass on a build
/// where chunking does nothing because the ring was never the problem.
#[test]
fn dsv4_ring_is_constant_in_context() {
    let dir = temp_dir();
    let model = tiny_model(&dir);
    let provider = ShardsExpertProvider::new(&model.shards, &model.cfg, 8);
    const CHUNK: usize = 4;

    let mut widths = Vec::new();
    let mut unchunked = Vec::new();
    for len in [16usize, 32, 48] {
        let ids: Vec<i32> = (0..len).map(|i| ((i * 3 + 1) % VOCAB) as i32).collect();
        let mut kv = KvCache::for_model(&model, 64);
        let mut out = vec![0f32; len * D];
        colibri_engine::forward::dsv4_forward_chunked(
            &model, &mut kv, &provider, &ids, 0, &mut out, CHUNK,
        )
        .unwrap();
        assert!(out.iter().all(|v| v.is_finite()), "len {len}: non-finite output");
        widths.push(kv.ring());

        let (_, kv_whole) = run_chunked(&model, &provider, &ids, len);
        unchunked.push(kv_whole.ring());
    }

    assert_eq!(
        widths[0], widths[1],
        "ring grew from a 16- to a 32-token prompt: {widths:?}"
    );
    assert_eq!(widths[1], widths[2], "ring grew again at 48 tokens: {widths:?}");
    assert!(widths[0] <= WINDOW + CHUNK, "ring {} exceeds window+chunk", widths[0]);
    // The control: without chunking it tracks the prompt exactly, which is what the ring
    // alone left unsolved.
    assert_eq!(unchunked, vec![16, 32, 48], "unchunked ring should track the prompt");
}
