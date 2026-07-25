//! End-to-end test of the **Nemotron-H MTP speculative head** against a tiny synthetic
//! container: the loader's two-sublayer head, the extra KV rows, the completeness gate,
//! and the draft/absorb forward paths.
//!
//! Nemotron's head is structurally different from GLM's: `mtp_hybrid_override_pattern`
//! is `"*E"`, so it is **two** sublayers (a NoPE-GQA attention block, then a gateless
//! latent-MoE block) at layer indices `n_layers` and `n_layers + 1` — not GLM's single
//! sparse block at `n_layers`. That is the thing under test here; GLM's own head keeps
//! its coverage in `forward_tiny.rs` / `load_tiny_model.rs`.
//!
//! Own integration binary because `load_model_with` sets the **process-global**
//! activation (`set_activation`, a `OnceLock`) to gateless ReLU² — sharing a process
//! with the SwiGLU tests would make whichever ran first win.
//!
//! What this can and cannot prove: the fixture's weights are synthetic, so draft
//! *acceptance* is meaningless. It asserts the plumbing — sublayer count and kinds,
//! layer indices, KV rows, shapes, in-range tokens, determinism, no panic. Whether
//! `fuse`'s norm placement matches how the head was trained only shows up as acceptance
//! rate on the real checkpoint.

use colibri_core::LayerKind;
use colibri_engine::{
    forward, load_model_with, logits, KvCache, LoadOptions, ShardsExpertProvider, KV_UNSET,
};
use std::fs::File;
use std::io::Write;
use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicU64, Ordering};

// ---- a tiny but structurally faithful Nemotron-H ---------------------------------
const D: usize = 8; // hidden
const NL: usize = 3; // main stack: "ME*" — one of each mixer
const VOCAB: usize = 10;
// attention: 2 query heads, 1 kv head, head_dim 4  (n_heads * head_dim == hidden)
const H: usize = 2;
const KVH: usize = 1;
const HD: usize = 4;
// mamba2: 2 heads x head_dim 2 (d_inner 4), 1 group x d_state 2, conv kernel 2
const M_NH: usize = 2;
const M_HD: usize = 2;
const M_DS: usize = 2;
const M_NG: usize = 1;
const M_K: usize = 2;
const D_INNER: usize = M_NH * M_HD; // 4
const CONV_DIM: usize = D_INNER + 2 * M_NG * M_DS; // 8
const IN_PROJ: usize = D_INNER + CONV_DIM + M_NH; // 14
// latent MoE
const E: usize = 4;
const TOPK: usize = 2;
const MOE_INTER: usize = 4;
const LATENT: usize = 3;
const SHARED_INTER: usize = 6;

/// The head's own indices: sublayer 0 (attention) at NL, sublayer 1 (MoE) at NL+1.
const MTP0: usize = NL;
const MTP1: usize = NL + 1;

fn temp_dir() -> PathBuf {
    static N: AtomicU64 = AtomicU64::new(0);
    let base = std::env::var("TMPDIR").unwrap_or_else(|_| "/tmp".into());
    let mut p = PathBuf::from(base);
    p.push(format!("colibri-nemomtp-{}-{}", std::process::id(), N.fetch_add(1, Ordering::Relaxed)));
    std::fs::create_dir_all(&p).unwrap();
    p
}

/// `with_mtp` adds `mtp_hybrid_override_pattern` — the config half of "this checkpoint
/// ships a head". Without it the loader must not build one even if tensors are present.
fn config_json(with_mtp: bool) -> String {
    let mtp = if with_mtp {
        r#""mtp_hybrid_override_pattern":"*E","num_nextn_predict_layers":1,"#
    } else {
        ""
    };
    format!(
        r#"{{"model_type":"nemotron_h",
        "hidden_size":{D},"num_hidden_layers":{NL},"hybrid_override_pattern":"ME*",{mtp}
        "num_attention_heads":{H},"num_key_value_heads":{KVH},"head_dim":{HD},
        "vocab_size":{VOCAB},"max_position_embeddings":256,
        "n_routed_experts":{E},"num_experts_per_tok":{TOPK},
        "moe_intermediate_size":{MOE_INTER},"moe_latent_size":{LATENT},
        "moe_shared_expert_intermediate_size":{SHARED_INTER},
        "norm_topk_prob":false,"routed_scaling_factor":1.0,"mlp_hidden_act":"relu2",
        "ssm_state_size":{M_DS},"conv_kernel":{M_K},"mamba_num_heads":{M_NH},
        "mamba_head_dim":{M_HD},"n_groups":{M_NG},"chunk_size":2,"time_step_min":0.001,
        "layer_norm_epsilon":1e-5,"eos_token_id":[999]}}"#
    )
}

/// One Mamba2 layer's tensors (container names, element counts).
fn mamba_layer(i: usize, t: &mut Vec<(String, usize)>) {
    let p = |s: &str| format!("model.layers.{i}.{s}");
    t.push((p("input_layernorm.weight"), D));
    t.push((p("mixer.in_proj.weight"), IN_PROJ * D));
    t.push((p("mixer.out_proj.weight"), D * D_INNER));
    t.push((p("mixer.conv1d.weight"), CONV_DIM * M_K));
    t.push((p("mixer.conv1d.bias"), CONV_DIM));
    t.push((p("mixer.A_log"), M_NH));
    t.push((p("mixer.D"), M_NH));
    t.push((p("mixer.dt_bias"), M_NH));
    t.push((p("mixer.norm.weight"), D_INNER));
}

/// One GQA attention layer's tensors (NoPE, no QK-norm — so no q_norm/k_norm).
fn attn_layer(i: usize, t: &mut Vec<(String, usize)>) {
    let p = |s: &str| format!("model.layers.{i}.{s}");
    t.push((p("input_layernorm.weight"), D));
    t.push((p("mixer.q_proj.weight"), H * HD * D));
    t.push((p("mixer.k_proj.weight"), KVH * HD * D));
    t.push((p("mixer.v_proj.weight"), KVH * HD * D));
    t.push((p("mixer.o_proj.weight"), D * H * HD));
}

/// One latent-MoE layer's tensors, including its streamed routed experts (gateless
/// `up`/`down` in latent space, under `.mixer.experts.`).
fn moe_layer(i: usize, t: &mut Vec<(String, usize)>) {
    let p = |s: &str| format!("model.layers.{i}.{s}");
    t.push((p("input_layernorm.weight"), D));
    t.push((p("mixer.gate.weight"), E * D));
    t.push((p("mixer.gate.e_score_correction_bias"), E));
    t.push((p("mixer.fc1_latent_proj.weight"), LATENT * D));
    t.push((p("mixer.fc2_latent_proj.weight"), D * LATENT));
    t.push((p("mixer.shared_experts.up_proj.weight"), SHARED_INTER * D));
    t.push((p("mixer.shared_experts.down_proj.weight"), D * SHARED_INTER));
    for e in 0..E {
        t.push((p(&format!("mixer.experts.{e}.up_proj.weight")), MOE_INTER * LATENT));
        t.push((p(&format!("mixer.experts.{e}.down_proj.weight")), LATENT * MOE_INTER));
    }
}

/// The MTP head as the converter emits it: sublayer 0 = attention at `MTP0`, sublayer 1
/// = latent-MoE at `MTP1`, plus the fusion tensors and the head's final norm — all four
/// of those at the head's BASE index under GLM's names (see `nemotron_container_name`).
fn mtp_head(t: &mut Vec<(String, usize)>) {
    attn_layer(MTP0, t);
    moe_layer(MTP1, t);
    let p = |s: &str| format!("model.layers.{MTP0}.{s}");
    t.push((p("eh_proj.weight"), D * 2 * D));
    t.push((p("enorm.weight"), D));
    t.push((p("hnorm.weight"), D));
    t.push((p("shared_head.norm.weight"), D)); // from mtp.layers.1.final_layernorm
}

fn tensor_list(with_mtp: bool) -> Vec<(String, usize)> {
    let mut t: Vec<(String, usize)> = vec![
        ("model.embed_tokens.weight".into(), VOCAB * D),
        ("lm_head.weight".into(), VOCAB * D),
        ("model.norm.weight".into(), D),
    ];
    mamba_layer(0, &mut t); // 'M'
    moe_layer(1, &mut t); // 'E'
    attn_layer(2, &mut t); // '*'
    if with_mtp {
        mtp_head(&mut t);
    }
    t
}

/// Write a single-shard safetensors file of small deterministic F32 values.
fn write_tensors(dir: &Path, tensors: &[(String, usize)]) {
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
        // Vary per tensor (hash the name) and keep values small + nonzero so every
        // per-row int8 amax is nonzero.
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

fn snapshot(with_mtp: bool) -> PathBuf {
    let dir = temp_dir();
    std::fs::write(dir.join("config.json"), config_json(with_mtp)).unwrap();
    write_tensors(&dir, &tensor_list(with_mtp));
    dir
}

/// The loader builds a **two-sublayer** head with the kinds the config's
/// `mtp_hybrid_override_pattern` names, and both sublayers load like ordinary Nemotron
/// layers (`mixer.*` names, canonical `input_layernorm`).
#[test]
fn nemotron_mtp_head_loads_two_sublayers() {
    let dir = snapshot(true);
    let m = load_model_with(&dir, LoadOptions { dbits: 8, ebits: 8 }).expect("load");

    assert!(m.has_mtp, "complete two-sublayer head must enable MTP");
    let mtp = m.mtp.as_ref().unwrap();
    assert_eq!(mtp.blocks.len(), 2, "'*E' is two sublayers, not GLM's one");
    assert_eq!(mtp.blocks[0].kind, Some(LayerKind::Attn));
    assert_eq!(mtp.blocks[1].kind, Some(LayerKind::Moe));

    // sublayer 0: fused q/k/v (one matmul, as on the main stack) + o_proj.
    let a = &mtp.blocks[0].layer;
    let qkv = a.qkv_proj.as_ref().expect("head attention must fuse q/k/v");
    assert_eq!((qkv.o as usize, qkv.i as usize), ((H + 2 * KVH) * HD, D));
    assert_eq!((a.o.o as usize, a.o.i as usize), (D, H * HD));
    assert!(a.q_norm.is_empty() && a.k_norm.is_empty(), "Nemotron attention is no-qk-norm");
    assert_eq!(a.in_ln.len(), D);

    // sublayer 1: latent projections + router + gateless shared expert.
    let e = &mtp.blocks[1].layer;
    assert!(e.sparse);
    assert_eq!(e.router.len(), E * D);
    assert_eq!(e.router_bias.len(), E);
    let (fc1, fc2) = (e.fc1_latent.as_ref().unwrap(), e.fc2_latent.as_ref().unwrap());
    assert_eq!((fc1.o as usize, fc1.i as usize), (LATENT, D));
    assert_eq!((fc2.o as usize, fc2.i as usize), (D, LATENT));
    assert_eq!((e.up_proj.o as usize, e.up_proj.i as usize), (SHARED_INTER, D));

    // The head's own tensors: eh_proj consumes [e ; h] so it is 2D wide.
    assert_eq!((mtp.eh_proj.o as usize, mtp.eh_proj.i as usize), (D, 2 * D));
    assert_eq!(mtp.enorm.len(), D);
    assert_eq!(mtp.hnorm.len(), D);
    assert_eq!(mtp.mtp_norm.len(), D);
    // `layer()` (the single-block convenience) still points at sublayer 0.
    assert_eq!(mtp.layer().o.o as usize, D);
    // The main stack is untouched by the two extra indices.
    assert_eq!(m.layers.len(), NL);

    // Resident head weights must be GPU-eligible: they are NOT in `model.layers`, so the
    // loop in `load_model_with` never sees them (the `gpu_eligible` trap).
    assert!(qkv.gpu_eligible && a.o.gpu_eligible, "head attention projections");
    assert!(fc1.gpu_eligible && fc2.gpu_eligible, "head latent projections");
    assert!(e.up_proj.gpu_eligible && e.down_proj.gpu_eligible, "head shared expert");

    std::fs::remove_dir_all(&dir).ok();
}

/// A two-sublayer head needs **two** extra KV rows, both starting [`KV_UNSET`] — the
/// head's cache begins at the first decode position, not at the prompt.
#[test]
fn nemotron_mtp_head_gets_two_kv_rows() {
    let plain = snapshot(false);
    let m = load_model_with(&plain, LoadOptions::default()).unwrap();
    assert!(!m.has_mtp, "no mtp pattern in config -> no head");
    let kv = KvCache::for_model(&m, 16);
    assert_eq!(kv.kv_start.len(), NL, "no head -> no extra rows");
    assert!(kv.kv_start.iter().all(|&s| s == 0));
    std::fs::remove_dir_all(&plain).ok();

    let dir = snapshot(true);
    let m = load_model_with(&dir, LoadOptions::default()).unwrap();
    let kv = KvCache::for_model(&m, 16);
    assert_eq!(kv.kv_start.len(), NL + 2, "one KV row per head sublayer");
    assert!(kv.kv_start[..NL].iter().all(|&s| s == 0), "main stack starts at 0");
    assert_eq!(kv.kv_start[MTP0], KV_UNSET);
    assert_eq!(kv.kv_start[MTP1], KV_UNSET);
    std::fs::remove_dir_all(&dir).ok();
}

/// The head's tensors span shards, so a truncated conversion leaves a subset behind. Like
/// GLM's loader, Nemotron's must refuse a partial head outright — a half-loaded head
/// drafts garbage. Dropping a tensor from EITHER sublayer must disable it.
#[test]
fn incomplete_nemotron_mtp_head_is_ignored() {
    for dropped in [
        // sublayer 0 (attention)
        format!("model.layers.{MTP0}.mixer.v_proj.weight"),
        // sublayer 1 (MoE) — the last expert is exactly what a truncated shard loses
        format!("model.layers.{MTP1}.mixer.experts.{}.down_proj.weight", E - 1),
        // the head's own fusion input
        format!("model.layers.{MTP0}.eh_proj.weight"),
        // the head's final norm (source: mtp.layers.1.final_layernorm)
        format!("model.layers.{MTP0}.shared_head.norm.weight"),
    ] {
        let dir = temp_dir();
        std::fs::write(dir.join("config.json"), config_json(true)).unwrap();
        let tensors: Vec<(String, usize)> =
            tensor_list(true).into_iter().filter(|(n, _)| *n != dropped).collect();
        write_tensors(&dir, &tensors);

        let m = load_model_with(&dir, LoadOptions::default()).expect("load");
        assert!(!m.has_mtp, "missing {dropped} must disable the head");
        assert!(m.mtp.is_none());
        std::fs::remove_dir_all(&dir).ok();
    }
}

/// The draft path runs both sublayers and reaches `lm_head`: it proposes in-range tokens,
/// establishes a PARTIAL KV on **both** head rows at the same start position, is
/// deterministic, and `absorb` replays it over verified tokens without panicking.
#[test]
fn nemotron_mtp_head_drafts_and_absorbs() {
    let dir = snapshot(true);
    let model = load_model_with(&dir, LoadOptions { dbits: 8, ebits: 8 }).expect("load");
    assert!(model.has_mtp);
    let provider = ShardsExpertProvider::new(&model.shards, &model.cfg, 8);

    let prompt = [1i32, 5, 2];
    let mut kv = KvCache::for_model(&model, 32);
    let mut hidden = vec![0f32; prompt.len() * D];
    forward(&model, &mut kv, &provider, &prompt, 0, &mut hidden).expect("prefill");

    let lo = logits(&model, &hidden[(prompt.len() - 1) * D..]);
    assert!(lo.iter().all(|v| v.is_finite()), "prefill logits must be finite");
    let next = colibri_engine::argmax(&lo) as i32;

    let kv_idx = prompt.len();
    let last_hidden = &hidden[(prompt.len() - 1) * D..prompt.len() * D];
    let g = 3;
    let drafts =
        colibri_engine::mtp_draft(&model, &mut kv, &provider, next, kv_idx, g, last_hidden)
            .expect("draft");

    assert_eq!(drafts.len(), g, "room for g drafts in a 32-token cache");
    for &t in &drafts {
        assert!((0..VOCAB as i32).contains(&t), "draft {t} out of vocab range");
    }
    // Both sublayers ran, at their own indices, from the same partial start.
    assert_eq!(kv.kv_start[MTP0], kv_idx - 1);
    assert_eq!(kv.kv_start[MTP1], kv_idx - 1);
    assert!(kv.kv_start[..NL].iter().all(|&s| s == 0), "main stack unaffected");

    // Deterministic: same inputs -> same drafts.
    let mut kv2 = KvCache::for_model(&model, 32);
    let mut h2 = vec![0f32; prompt.len() * D];
    forward(&model, &mut kv2, &provider, &prompt, 0, &mut h2).expect("prefill");
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

    // Absorb the verified prefix: runs the head for its KV side effect only.
    colibri_engine::mtp_absorb(&model, &mut kv, &provider, &drafts[..1], last_hidden, kv_idx)
        .expect("absorb");

    std::fs::remove_dir_all(&dir).ok();
}

/// **Speculation's defining invariant**, on the hybrid arch: a draft is only accepted
/// when it matches what the main model would itself have produced, so `DRAFT=n` must
/// emit exactly the tokens `DRAFT=0` does. This is what catches accept-off-by-one, a KV
/// desync across the head's two rows, and a stale `hlast`.
#[test]
fn nemotron_speculation_does_not_change_output() {
    let dir = snapshot(true);
    let model = load_model_with(&dir, LoadOptions { dbits: 8, ebits: 8 }).expect("load");
    let provider = ShardsExpertProvider::new(&model.shards, &model.cfg, 8);
    let prompt = [1i32, 5, 2];
    let n_new = 6;

    let run = |budget: usize| -> Vec<i32> {
        let mut kv = KvCache::for_model(&model, 64);
        let mut out = Vec::new();
        colibri_engine::generate_stream_drafting(
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
        .expect("generate");
        out
    };

    let plain = run(0);
    assert_eq!(plain.len(), n_new);
    for budget in [1usize, 3] {
        assert_eq!(run(budget), plain, "DRAFT={budget} changed the output");
    }
    std::fs::remove_dir_all(&dir).ok();
}
