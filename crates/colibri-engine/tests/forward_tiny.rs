//! End-to-end forward pass + greedy decode on a tiny synthetic GLM snapshot
//! (dense weights + routed expert tensors). We can't check exact token values
//! without a reference model, but we assert the whole pipeline wires up: it runs
//! without panicking, produces in-range tokens and finite logits, and is
//! deterministic.
//!
//! # KNOWN FLAKE — run these serially under `--features cuda`
//!
//! ```text
//! cargo test --release --features cuda -- --test-threads=1
//! ```
//!
//! With cargo's default parallelism these tests fail nondeterministically on a CUDA
//! build — measured on GB10 at 0, 3, 4 and 5 failures across repeated runs of the
//! *same* binary. Serial passes 3/3; CPU-only passes in parallel. So it is GPU state
//! shared across concurrently-running tests, not a temp-dir collision and not the
//! `-arch` target (sm_121 and sm_121a flake identically).
//!
//! The root cause is NOT diagnosed. Candidates, unverified: the global `BUF_POOL`
//! recycling buffer addresses while `gpu.rs` keys its device-tensor cache by pointer
//! (`try_matmul_qt`), or CUDA context init racing. Do not read a green parallel run
//! as evidence of a fix — one arch was observed passing 6/6 once and failing 5/6 two
//! runs later.

use colibri_engine::{
    forward, generate_greedy, load_model_with, logits, preload_parallel, repack, ExpertProvider,
    KvCache, LoadOptions, PreloadStore, ShardsExpertProvider,
};
use std::fs::File;
use std::io::Write;
use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicU64, Ordering};

const D: usize = 8;
const NL: usize = 2;
const H: usize = 2;
const E: usize = 4;
const MOE_INTER: usize = 4;
const DENSE_INTER: usize = 8;
const FIRST_DENSE: usize = 1;
const Q_LORA: usize = 4;
const KV_LORA: usize = 4;
const QK_NOPE: usize = 4;
const QK_ROPE: usize = 2;
const V_HEAD: usize = 4;
const N_SHARED: usize = 1;
const VOCAB: usize = 10;
const QK_HEAD: usize = QK_NOPE + QK_ROPE;
const S_I: usize = MOE_INTER * N_SHARED;

fn temp_dir() -> PathBuf {
    static N: AtomicU64 = AtomicU64::new(0);
    let base = std::env::var("TMPDIR").unwrap_or_else(|_| "/tmp".into());
    let mut p = PathBuf::from(base);
    p.push(format!(
        "colibri-fwd-{}-{}",
        std::process::id(),
        N.fetch_add(1, Ordering::Relaxed)
    ));
    std::fs::create_dir_all(&p).unwrap();
    p
}

fn config_json() -> String {
    format!(
        r#"{{"hidden_size":{D},"num_hidden_layers":{NL},"num_attention_heads":{H},
        "n_routed_experts":{E},"num_experts_per_tok":2,"moe_intermediate_size":{MOE_INTER},
        "intermediate_size":{DENSE_INTER},"first_k_dense_replace":{FIRST_DENSE},
        "q_lora_rank":{Q_LORA},"kv_lora_rank":{KV_LORA},"qk_nope_head_dim":{QK_NOPE},
        "qk_rope_head_dim":{QK_ROPE},"v_head_dim":{V_HEAD},"n_shared_experts":{N_SHARED},
        "vocab_size":{VOCAB},"n_group":1,"topk_group":1,"norm_topk_prob":false,
        "rms_norm_eps":1e-5,"routed_scaling_factor":1.0,
        "rope_parameters":{{"rope_theta":10000.0}},"eos_token_id":[999],
        "index_topk":0,"index_n_heads":0,"index_head_dim":0}}"#
    )
}

/// Append a complete MTP head at layer index `NL`: its own always-sparse block
/// (attention + router + shared + routed experts) plus the four fusion tensors.
fn push_mtp_head(t: &mut Vec<(String, usize)>) {
    let i = NL;
    let p = |s: &str| format!("model.layers.{i}.{s}");
    t.push((p("input_layernorm.weight"), D));
    t.push((p("post_attention_layernorm.weight"), D));
    t.push((p("self_attn.q_a_proj.weight"), Q_LORA * D));
    t.push((p("self_attn.q_a_layernorm.weight"), Q_LORA));
    t.push((p("self_attn.q_b_proj.weight"), H * QK_HEAD * Q_LORA));
    t.push((p("self_attn.kv_a_proj_with_mqa.weight"), (KV_LORA + QK_ROPE) * D));
    t.push((p("self_attn.kv_a_layernorm.weight"), KV_LORA));
    t.push((p("self_attn.kv_b_proj.weight"), H * (QK_NOPE + V_HEAD) * KV_LORA));
    t.push((p("self_attn.o_proj.weight"), D * H * V_HEAD));
    t.push((p("mlp.gate.weight"), E * D));
    t.push((p("mlp.gate.e_score_correction_bias"), E));
    t.push((p("mlp.shared_experts.gate_proj.weight"), S_I * D));
    t.push((p("mlp.shared_experts.up_proj.weight"), S_I * D));
    t.push((p("mlp.shared_experts.down_proj.weight"), D * S_I));
    for e in 0..E {
        let pe = |s: &str| format!("model.layers.{i}.mlp.experts.{e}.{s}.weight");
        t.push((pe("gate_proj"), MOE_INTER * D));
        t.push((pe("up_proj"), MOE_INTER * D));
        t.push((pe("down_proj"), D * MOE_INTER));
    }
    t.push((p("eh_proj.weight"), D * 2 * D)); // [D, 2D]
    t.push((p("enorm.weight"), D));
    t.push((p("hnorm.weight"), D));
    t.push((p("shared_head.norm.weight"), D));
}

fn tensor_list() -> Vec<(String, usize)> {
    let mut t: Vec<(String, usize)> = vec![
        ("model.embed_tokens.weight".into(), VOCAB * D),
        ("lm_head.weight".into(), VOCAB * D),
        ("model.norm.weight".into(), D),
    ];
    for i in 0..NL {
        let p = |s: &str| format!("model.layers.{i}.{s}");
        t.push((p("input_layernorm.weight"), D));
        t.push((p("post_attention_layernorm.weight"), D));
        t.push((p("self_attn.q_a_proj.weight"), Q_LORA * D));
        t.push((p("self_attn.q_a_layernorm.weight"), Q_LORA));
        t.push((p("self_attn.q_b_proj.weight"), H * QK_HEAD * Q_LORA));
        t.push((p("self_attn.kv_a_proj_with_mqa.weight"), (KV_LORA + QK_ROPE) * D));
        t.push((p("self_attn.kv_a_layernorm.weight"), KV_LORA));
        t.push((p("self_attn.kv_b_proj.weight"), H * (QK_NOPE + V_HEAD) * KV_LORA));
        t.push((p("self_attn.o_proj.weight"), D * H * V_HEAD));
        if i < FIRST_DENSE {
            t.push((p("mlp.gate_proj.weight"), DENSE_INTER * D));
            t.push((p("mlp.up_proj.weight"), DENSE_INTER * D));
            t.push((p("mlp.down_proj.weight"), D * DENSE_INTER));
        } else {
            t.push((p("mlp.gate.weight"), E * D));
            t.push((p("mlp.gate.e_score_correction_bias"), E));
            t.push((p("mlp.shared_experts.gate_proj.weight"), S_I * D));
            t.push((p("mlp.shared_experts.up_proj.weight"), S_I * D));
            t.push((p("mlp.shared_experts.down_proj.weight"), D * S_I));
            // routed experts (streamed by the provider)
            for e in 0..E {
                let pe = |s: &str| format!("model.layers.{i}.mlp.experts.{e}.{s}.weight");
                t.push((pe("gate_proj"), MOE_INTER * D));
                t.push((pe("up_proj"), MOE_INTER * D));
                t.push((pe("down_proj"), D * MOE_INTER));
            }
        }
    }
    t
}

fn write_model(dir: &Path) {
    write_model_opt(dir, false)
}

/// `with_mtp` appends the speculative head, so MTP paths are exercisable without
/// the real 350 GB container.
fn write_model_opt(dir: &Path, with_mtp: bool) {
    let mut tensors = tensor_list();
    if with_mtp {
        push_mtp_head(&mut tensors);
    }
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
        // deterministic small values that vary per tensor (hash the name)
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

/// The MTP head's forward paths wire up: `draft` chains the head to propose
/// tokens, and `absorb` runs it over verified tokens for its KV.
///
/// NOTE what this can and cannot prove. The fixture's weights are synthetic, so
/// "acceptance" here is meaningless — this asserts the *plumbing* (shapes, layer
/// index n_layers, the extra KV row, in-range tokens, no panic, determinism).
/// Whether `fuse`'s norm placement / concat order match how the head was trained
/// can only be shown by acceptance rate on the real model.
#[test]
fn mtp_head_drafts_and_absorbs() {
    let dir = temp_dir();
    std::fs::write(dir.join("config.json"), config_json()).unwrap();
    write_model_opt(&dir, true);

    let model = load_model_with(&dir, LoadOptions { dbits: 4, ebits: 4 }).expect("load");
    assert!(model.has_mtp, "fixture ships a complete head");
    let provider = ShardsExpertProvider::new(&model.shards, &model.cfg, 4);

    // prefill a prompt through the main stack
    let prompt = [1i32, 5, 2];
    let mut kv = KvCache::for_model(&model, 32);
    assert_eq!(kv.kv_start.len(), NL + 1, "head gets its own KV row");
    let mut hidden = vec![0f32; prompt.len() * D];
    forward(&model, &mut kv, &provider, &prompt, 0, &mut hidden).expect("forward");

    // the token the main model would emit next
    let lo = logits(&model, &hidden[(prompt.len() - 1) * D..]);
    let next = colibri_engine::argmax(&lo) as i32;

    // draft from it: `next` sits at index prompt.len()-1+1 == prompt.len()-1 ...
    // C convention: kv_idx is the index `next` was stored at.
    let kv_idx = prompt.len();
    let last_hidden = &hidden[(prompt.len() - 1) * D..prompt.len() * D];
    let g = 3;
    let drafts =
        colibri_engine::mtp_draft(&model, &mut kv, &provider, next, kv_idx, g, last_hidden)
            .expect("draft");

    assert_eq!(drafts.len(), g, "should propose g tokens with room to spare");
    for &t in &drafts {
        assert!((0..VOCAB as i32).contains(&t), "draft {t} out of vocab range");
    }
    // the head established its KV start at p = kv_idx - 1 (a PARTIAL cache:
    // it begins mid-sequence, unlike the main stack's 0)
    assert_eq!(kv.kv_start[NL], kv_idx - 1);
    assert!(kv.kv_start[..NL].iter().all(|&s| s == 0));

    // deterministic: same inputs -> same drafts
    let mut kv2 = KvCache::for_model(&model, 32);
    let mut h2 = vec![0f32; prompt.len() * D];
    forward(&model, &mut kv2, &provider, &prompt, 0, &mut h2).expect("forward");
    let d2 = colibri_engine::mtp_draft(
        &model,
        &mut kv2,
        &provider,
        next,
        kv_idx,
        g,
        &h2[(prompt.len() - 1) * D..prompt.len() * D],
    )
    .expect("draft");
    assert_eq!(drafts, d2, "drafting must be deterministic");

    // absorb the verified tokens: runs the head over them for its KV only
    colibri_engine::mtp_absorb(&model, &mut kv, &provider, &drafts[..1], last_hidden, kv_idx)
        .expect("absorb");

    std::fs::remove_dir_all(&dir).ok();
}

/// **Speculation's defining invariant**: accepting only drafts that match what
/// the model itself would produce means `DRAFT=n` must emit EXACTLY the tokens
/// `DRAFT=0` does. Any divergence is a bug in the verify/KV bookkeeping.
///
/// This is the strongest MTP test available without a C oracle or a trained
/// head: it catches accept-off-by-one, KV desync, and a wrong `hlast` — the
/// bugs that actually corrupt output. (It cannot catch a *wrong* `fuse`
/// orientation: that only tanks acceptance, leaving output correct.)
#[test]
fn speculation_does_not_change_output() {
    let dir = temp_dir();
    std::fs::write(dir.join("config.json"), config_json()).unwrap();
    write_model_opt(&dir, true);
    let model = load_model_with(&dir, LoadOptions { dbits: 4, ebits: 4 }).expect("load");
    assert!(model.has_mtp);
    let provider = ShardsExpertProvider::new(&model.shards, &model.cfg, 4);
    let prompt = [1i32, 5, 2];
    let n_new = 8;

    // Explicit budgets (not the DRAFT env, which is read once per process).
    let run = |budget: usize| {
        let mut kv = KvCache::for_model(&model, 64);
        let mut out: Vec<i32> = Vec::new();
        let st = colibri_engine::generate_stream_drafting(
            &model,
            &mut kv,
            &provider,
            &prompt,
            n_new,
            budget,
            |t| {
                out.push(t);
                true
            },
        )
        .expect("gen");
        (out, st)
    };

    let (baseline, st0) = run(0); // speculation off
    assert_eq!(baseline.len(), n_new, "sanity: produced the requested tokens");
    assert_eq!(st0.drafts_proposed, 0, "budget 0 must not draft");
    assert_eq!(st0.forwards, n_new as u64 - 1, "one forward per token, less the last");

    let mut any_accepted = false;
    for budget in [1usize, 2, 3, 5] {
        let (toks, st) = run(budget);
        assert_eq!(baseline, toks, "DRAFT={budget} must emit exactly the tokens DRAFT=0 does");
        assert!(st.drafts_proposed > 0, "DRAFT={budget} should propose drafts");
        assert!(st.drafts_accepted <= st.drafts_proposed);
        if st.drafts_accepted > 0 {
            any_accepted = true;
            // Every accepted draft is a forward saved — that IS the win.
            assert!(
                st.forwards < st.emitted as u64,
                "DRAFT={budget}: {} accepted but forwards ({}) not < emitted ({})",
                st.drafts_accepted,
                st.forwards,
                st.emitted
            );
        }
    }
    // Without this the identity check above is VACUOUS: if no draft is ever
    // accepted the loop degenerates to the non-speculative path and proves
    // nothing about the accept/KV bookkeeping.
    assert!(
        any_accepted,
        "no draft was ever accepted — the identity test never exercised the accept path"
    );

    std::fs::remove_dir_all(&dir).ok();
}

/// A model with no MTP head must make the head's paths no-ops, not panic.
#[test]
fn mtp_paths_are_noops_without_a_head() {
    let dir = temp_dir();
    std::fs::write(dir.join("config.json"), config_json()).unwrap();
    write_model(&dir); // no MTP head
    let model = load_model_with(&dir, LoadOptions::default()).expect("load");
    assert!(!model.has_mtp);
    let provider = ShardsExpertProvider::new(&model.shards, &model.cfg, 4);
    let mut kv = KvCache::for_model(&model, 16);
    let h = vec![0f32; D];
    assert!(colibri_engine::mtp_draft(&model, &mut kv, &provider, 1, 1, 4, &h)
        .expect("draft")
        .is_empty());
    colibri_engine::mtp_absorb(&model, &mut kv, &provider, &[1], &h, 0).expect("absorb");
    std::fs::remove_dir_all(&dir).ok();
}

#[test]
fn full_forward_and_greedy_decode() {
    let dir = temp_dir();
    std::fs::write(dir.join("config.json"), config_json()).unwrap();
    write_model(&dir);

    let model = load_model_with(&dir, LoadOptions { dbits: 4, ebits: 4 }).expect("load");
    let provider = ShardsExpertProvider::new(&model.shards, &model.cfg, 4);

    // one forward over a 3-token prompt -> finite logits, argmax in range
    let prompt = [1i32, 5, 2];
    let mut kv = KvCache::new(NL, KV_LORA, QK_ROPE, 32);
    let mut hidden = vec![0f32; prompt.len() * D];
    forward(&model, &mut kv, &provider, &prompt, 0, &mut hidden).unwrap();
    assert!(hidden.iter().all(|v| v.is_finite()), "hidden not finite");

    let last = logits(&model, &hidden[(prompt.len() - 1) * D..prompt.len() * D]);
    assert_eq!(last.len(), VOCAB);
    assert!(last.iter().all(|v| v.is_finite()), "logits not finite");

    // greedy generate 5 tokens
    let mut kv2 = KvCache::new(NL, KV_LORA, QK_ROPE, 32);
    let seq = generate_greedy(&model, &mut kv2, &provider, &prompt, 5).unwrap();
    assert_eq!(&seq[..3], &prompt); // prompt preserved
    assert!(seq.len() > 3 && seq.len() <= 8);
    assert!(seq.iter().all(|&t| (0..VOCAB as i32).contains(&t)), "token out of range: {seq:?}");

    // determinism: same prompt -> same continuation
    let mut kv3 = KvCache::new(NL, KV_LORA, QK_ROPE, 32);
    let seq2 = generate_greedy(&model, &mut kv3, &provider, &prompt, 5).unwrap();
    assert_eq!(seq, seq2, "greedy decode must be deterministic");

    std::fs::remove_dir_all(&dir).ok();
}

#[test]
fn repack_then_parallel_preload_matches_disk() {
    let dir = temp_dir();
    std::fs::write(dir.join("config.json"), config_json()).unwrap();
    write_model(&dir);

    let model = load_model_with(&dir, LoadOptions { dbits: 4, ebits: 4 }).expect("load");
    let shards = ShardsExpertProvider::new(&model.shards, &model.cfg, 4);

    // repack the E experts of the sparse layer into 3 shard files
    let out = dir.join("repacked");
    let manifest = repack(&shards, &model.cfg, &out, 3).expect("repack");
    assert_eq!(manifest.experts.len(), E); // sparse layer's experts
    assert_eq!(manifest.num_files, 3);

    // parallel load everything, then check each expert is byte-identical to disk
    let store = PreloadStore::load(&manifest, &out, u64::MAX).expect("preload");
    assert_eq!(store.len(), E);
    for eid in 0..E {
        let a = shards.expert(1, eid).unwrap(); // layer 1 is the sparse one
        let b = store.expert(1, eid).unwrap();
        assert_eq!(a.gate.q4, b.gate.q4, "gate.q4 eid {eid}");
        assert_eq!(a.gate.s, b.gate.s, "gate.s eid {eid}");
        assert_eq!(a.up.q4, b.up.q4);
        assert_eq!(a.up.s, b.up.s);
        assert_eq!(a.down.q4, b.down.q4);
        assert_eq!(a.down.s, b.down.s);
    }

    // generation with the preloaded store must equal generation from disk
    let prompt = [1i32, 5, 2];
    let mut kv1 = KvCache::new(NL, KV_LORA, QK_ROPE, 16);
    let from_disk = generate_greedy(&model, &mut kv1, &shards, &prompt, 6).unwrap();
    let mut kv2 = KvCache::new(NL, KV_LORA, QK_ROPE, 16);
    let from_preload = generate_greedy(&model, &mut kv2, &store, &prompt, 6).unwrap();
    assert_eq!(from_disk, from_preload, "preloaded output must match disk");

    std::fs::remove_dir_all(&dir).ok();
}

#[test]
fn direct_parallel_preload_matches_disk() {
    // No repack: read experts straight from the original model in parallel.
    let dir = temp_dir();
    std::fs::write(dir.join("config.json"), config_json()).unwrap();
    write_model(&dir);

    let model = load_model_with(&dir, LoadOptions { dbits: 4, ebits: 4 }).expect("load");
    let shards = ShardsExpertProvider::new(&model.shards, &model.cfg, 4);

    let store = preload_parallel(&model.shards, &model.cfg, 4, 4, u64::MAX).expect("preload");
    assert_eq!(store.len(), E);
    for eid in 0..E {
        let a = shards.expert(1, eid).unwrap();
        let b = store.expert(1, eid).unwrap();
        assert_eq!(a.gate.q4, b.gate.q4);
        assert_eq!(a.down.s, b.down.s);
    }

    let prompt = [1i32, 5, 2];
    let mut kv1 = KvCache::new(NL, KV_LORA, QK_ROPE, 16);
    let from_disk = generate_greedy(&model, &mut kv1, &shards, &prompt, 6).unwrap();
    let mut kv2 = KvCache::new(NL, KV_LORA, QK_ROPE, 16);
    let from_preload = generate_greedy(&model, &mut kv2, &store, &prompt, 6).unwrap();
    assert_eq!(from_disk, from_preload, "direct preload output must match disk");

    std::fs::remove_dir_all(&dir).ok();
}
