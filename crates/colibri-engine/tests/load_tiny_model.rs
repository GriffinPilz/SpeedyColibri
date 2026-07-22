//! End-to-end test of the weight-loader driver (`load_model`) against a tiny,
//! synthetic GLM-5.2-shaped snapshot: a real `config.json` plus a
//! `model.safetensors` carrying every dense tensor the loader reads, by GLM name.
//!
//! Dimensions are minimal but structurally faithful (first layer dense, second
//! sparse). Routed experts are streamed, so they are intentionally absent — the
//! loader must not require them.

use colibri_engine::{load_model_with, KvCache, LoadOptions, KV_UNSET};
use std::fs::File;
use std::io::Write;
use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::Mutex;

/// `MTP` is a process-global env var but cargo runs tests in parallel threads —
/// without this, `mtp_env_zero_disables_head` setting `MTP=0` leaks into any
/// concurrently-running test that expects the head to load. Every test whose
/// outcome depends on `MTP` takes this lock.
static MTP_ENV: Mutex<()> = Mutex::new(());

/// Lock `MTP_ENV`, ignoring poisoning (a panic in one test must not cascade).
fn mtp_env_guard() -> std::sync::MutexGuard<'static, ()> {
    MTP_ENV.lock().unwrap_or_else(|e| e.into_inner())
}

// tiny config
const D: usize = 8; // hidden
const NL: usize = 2; // layers
const H: usize = 2; // heads
const E: usize = 4; // experts
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

const QK_HEAD: usize = QK_NOPE + QK_ROPE; // 6
const S_I: usize = MOE_INTER * N_SHARED; // shared intermediate

fn temp_dir() -> PathBuf {
    static N: AtomicU64 = AtomicU64::new(0);
    let base = std::env::var("TMPDIR").unwrap_or_else(|_| "/tmp".into());
    let mut p = PathBuf::from(base);
    p.push(format!(
        "colibri-tinymodel-{}-{}",
        std::process::id(),
        N.fetch_add(1, Ordering::Relaxed)
    ));
    std::fs::create_dir_all(&p).unwrap();
    p
}

fn config_json() -> String {
    format!(
        r#"{{
        "hidden_size": {D}, "num_hidden_layers": {NL}, "num_attention_heads": {H},
        "n_routed_experts": {E}, "num_experts_per_tok": 2, "moe_intermediate_size": {MOE_INTER},
        "intermediate_size": {DENSE_INTER}, "first_k_dense_replace": {FIRST_DENSE},
        "q_lora_rank": {Q_LORA}, "kv_lora_rank": {KV_LORA}, "qk_nope_head_dim": {QK_NOPE},
        "qk_rope_head_dim": {QK_ROPE}, "v_head_dim": {V_HEAD}, "n_shared_experts": {N_SHARED},
        "vocab_size": {VOCAB}, "n_group": 1, "topk_group": 1, "norm_topk_prob": false,
        "rms_norm_eps": 1e-5, "routed_scaling_factor": 1.0,
        "rope_parameters": {{"rope_theta": 10000.0}}, "eos_token_id": [9],
        "index_topk": 0, "index_n_heads": 0, "index_head_dim": 0
    }}"#
    )
}

/// Every tensor of one transformer layer at index `i`, with element counts.
/// `sparse` picks MoE (router + shared expert) over the dense MLP.
fn layer_tensors(i: usize, sparse: bool, t: &mut Vec<(String, usize)>) {
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
    if !sparse {
        t.push((p("mlp.gate_proj.weight"), DENSE_INTER * D));
        t.push((p("mlp.up_proj.weight"), DENSE_INTER * D));
        t.push((p("mlp.down_proj.weight"), D * DENSE_INTER));
    } else {
        t.push((p("mlp.gate.weight"), E * D));
        t.push((p("mlp.gate.e_score_correction_bias"), E));
        t.push((p("mlp.shared_experts.gate_proj.weight"), S_I * D));
        t.push((p("mlp.shared_experts.up_proj.weight"), S_I * D));
        t.push((p("mlp.shared_experts.down_proj.weight"), D * S_I));
    }
}

/// The full list of dense tensors the loader reads, with element counts.
/// `with_mtp` appends a complete MTP head at the extra layer index `NL`: the
/// head's own (always-sparse) block, its `eh_proj`/`enorm`/`hnorm`/
/// `shared_head.norm`, and its routed experts — the loader's completeness gate
/// probes `experts.0` and `experts.{E-1}`, and the draft path streams them.
fn tensor_list(with_mtp: bool) -> Vec<(String, usize)> {
    let mut t: Vec<(String, usize)> = vec![
        ("model.embed_tokens.weight".into(), VOCAB * D),
        ("lm_head.weight".into(), VOCAB * D),
        ("model.norm.weight".into(), D),
    ];
    for i in 0..NL {
        layer_tensors(i, i >= FIRST_DENSE, &mut t);
    }
    if with_mtp {
        let i = NL; // the MTP head lives at layer index n_layers
        layer_tensors(i, true, &mut t); // C: mtpL is always sparse
        let p = |s: &str| format!("model.layers.{i}.{s}");
        t.push((p("eh_proj.weight"), D * 2 * D)); // [D, 2D]
        t.push((p("enorm.weight"), D));
        t.push((p("hnorm.weight"), D));
        t.push((p("shared_head.norm.weight"), D));
        // routed experts of the MTP block (streamed; the gate probes 0 and E-1)
        for e in 0..E {
            t.push((p(&format!("mlp.experts.{e}.gate_proj.weight")), MOE_INTER * D));
            t.push((p(&format!("mlp.experts.{e}.up_proj.weight")), MOE_INTER * D));
            t.push((p(&format!("mlp.experts.{e}.down_proj.weight")), D * MOE_INTER));
        }
    }
    t
}

/// Write a single-shard safetensors file with all tensors as small f32 values.
fn write_model(dir: &Path) {
    write_model_with(dir, false)
}

fn write_model_with(dir: &Path, with_mtp: bool) {
    write_tensors(dir, &tensor_list(with_mtp))
}

/// Write an explicit tensor list — lets a test drop a tensor to simulate a
/// shard-truncated conversion.
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
        for k in 0..*numel {
            // small nonzero values so quantization has a nonzero amax
            let v = ((k % 5) as f32 - 2.0) * 0.1;
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

#[test]
fn tiny_model_loads_end_to_end() {
    let dir = temp_dir();
    std::fs::write(dir.join("config.json"), config_json()).unwrap();
    write_model(&dir);

    let m = load_model_with(&dir, LoadOptions { dbits: 8, ebits: 8 }).expect("load_model");

    // structure
    assert_eq!(m.cfg.n_layers, NL as i32);
    assert_eq!(m.layers.len(), NL);
    assert!(!m.layers[0].sparse, "layer 0 should be dense");
    assert!(m.layers[1].sparse, "layer 1 should be sparse (MoE)");
    assert_eq!(m.dbits, 8);
    assert_eq!(m.ebits, 8);
    assert!(!m.has_mtp);
    assert!(!m.has_dsa);

    // I/O boundary: embed/lm_head loaded at io_bits=16 -> f32 (fmt 0), shape [vocab, D]
    assert_eq!((m.embed.o, m.embed.i), (VOCAB as i32, D as i32));
    assert_eq!(m.embed.fmt_code, 0);
    assert_eq!((m.lm_head.o, m.lm_head.i), (VOCAB as i32, D as i32));
    assert_eq!(m.final_norm.len(), D);

    // attention projection shapes on layer 0
    let l0 = &m.layers[0];
    assert_eq!((l0.q_a.o, l0.q_a.i), (Q_LORA as i32, D as i32));
    assert_eq!((l0.q_b.o, l0.q_b.i), ((H * QK_HEAD) as i32, Q_LORA as i32));
    assert_eq!((l0.kv_a.o, l0.kv_a.i), ((KV_LORA + QK_ROPE) as i32, D as i32));
    assert_eq!((l0.kv_b.o, l0.kv_b.i), ((H * (QK_NOPE + V_HEAD)) as i32, KV_LORA as i32));
    assert_eq!((l0.o.o, l0.o.i), (D as i32, (H * V_HEAD) as i32));
    // dense MLP on layer 0
    assert_eq!((l0.gate_proj.o, l0.gate_proj.i), (DENSE_INTER as i32, D as i32));
    assert_eq!((l0.down_proj.o, l0.down_proj.i), (D as i32, DENSE_INTER as i32));

    // MoE bits on layer 1
    let l1 = &m.layers[1];
    assert_eq!(l1.router.len(), E * D);
    assert_eq!(l1.router_bias.len(), E);
    assert_eq!((l1.sh_gate.o, l1.sh_gate.i), (S_I as i32, D as i32));
    assert_eq!((l1.sh_down.o, l1.sh_down.i), (D as i32, S_I as i32));

    // the loaded embedding is usable: dequant a row without panicking
    let mut x = vec![0f32; D];
    colibri_engine::embed_row(&m.embed, 3, &mut x);
    assert_eq!(x.len(), D);

    std::fs::remove_dir_all(&dir).ok();
}

/// A container converted WITH `--mtp` loads the speculative head: the block
/// itself plus eh_proj `[D, 2D]` and the three norms.
#[test]
fn tiny_model_with_mtp_head_loads() {
    let _env = mtp_env_guard();
    let dir = temp_dir();
    std::fs::write(dir.join("config.json"), config_json()).unwrap();
    write_model_with(&dir, true);

    let m = load_model_with(&dir, LoadOptions { dbits: 8, ebits: 8 }).expect("load_model");

    assert!(m.has_mtp, "complete MTP tensor set must enable the head");
    let mtp = m.mtp.as_ref().expect("mtp head loaded");
    // the head's own block is always sparse (C: mtpL->sparse = 1)
    assert!(mtp.layer.sparse, "MTP block must be sparse (MoE)");
    // eh_proj consumes the concatenated [e ; h] -> 2D wide, D out
    assert_eq!(mtp.eh_proj.o as usize, D);
    assert_eq!(mtp.eh_proj.i as usize, 2 * D);
    assert_eq!(mtp.enorm.len(), D);
    assert_eq!(mtp.hnorm.len(), D);
    assert_eq!(mtp.mtp_norm.len(), D);
    // the head's attention loaded like any layer's
    assert_eq!(mtp.layer.o.o as usize, D);
    // the main stack is untouched by the extra layer
    assert_eq!(m.layers.len(), NL);

    std::fs::remove_dir_all(&dir).ok();
}

/// The head's tensors span several shards, so a partial `--mtp` conversion can
/// leave a subset behind. A subset must DISABLE the head rather than half-load it
/// (a half-loaded head would draft garbage).
#[test]
fn incomplete_mtp_head_is_ignored() {
    let dir = temp_dir();
    std::fs::write(dir.join("config.json"), config_json()).unwrap();
    // Write the full MTP set, then drop one required tensor (the last expert,
    // which is exactly what a shard-truncated conversion loses).
    let dropped = format!("model.layers.{NL}.mlp.experts.{}.down_proj.weight", E - 1);
    let tensors: Vec<(String, usize)> =
        tensor_list(true).into_iter().filter(|(n, _)| *n != dropped).collect();
    write_tensors(&dir, &tensors);

    let m = load_model_with(&dir, LoadOptions::default()).expect("load_model");
    assert!(!m.has_mtp, "an incomplete MTP set must not enable the head");
    assert!(m.mtp.is_none());
    std::fs::remove_dir_all(&dir).ok();
}

/// `MTP=0` disables a present, complete head.
#[test]
fn mtp_env_zero_disables_head() {
    let _env = mtp_env_guard();
    let dir = temp_dir();
    std::fs::write(dir.join("config.json"), config_json()).unwrap();
    write_model_with(&dir, true);

    // SAFETY: single-threaded scope for this env toggle; restored below.
    std::env::set_var("MTP", "0");
    let m = load_model_with(&dir, LoadOptions::default()).expect("load_model");
    std::env::remove_var("MTP");

    assert!(!m.has_mtp, "MTP=0 must disable the head");
    assert!(m.mtp.is_none());
    std::fs::remove_dir_all(&dir).ok();
}

/// The MTP head is a real layer at index `n_layers`, so the KV needs an extra
/// row — and that row starts UNSET, because the head's cache begins at the first
/// decode position rather than at the start of the prompt.
#[test]
fn kv_cache_sizes_for_mtp_head() {
    let _env = mtp_env_guard();
    // no MTP: exactly n_layers rows, all starting at 0
    let dir = temp_dir();
    std::fs::write(dir.join("config.json"), config_json()).unwrap();
    write_model(&dir);
    let plain = load_model_with(&dir, LoadOptions::default()).unwrap();
    let kv = KvCache::for_model(&plain, 16);
    assert_eq!(kv.kv_start.len(), NL, "no MTP head -> no extra KV row");
    assert!(kv.kv_start.iter().all(|&s| s == 0));
    std::fs::remove_dir_all(&dir).ok();

    // with MTP: one extra row, and it starts UNSET
    let dir = temp_dir();
    std::fs::write(dir.join("config.json"), config_json()).unwrap();
    write_model_with(&dir, true);
    let m = load_model_with(&dir, LoadOptions::default()).unwrap();
    let mut kv = KvCache::for_model(&m, 16);
    assert_eq!(kv.kv_start.len(), NL + 1, "MTP head needs its own KV row");
    assert!(kv.kv_start[..NL].iter().all(|&s| s == 0), "main stack starts at 0");
    assert_eq!(kv.kv_start[NL], KV_UNSET, "MTP row starts unset");

    // start_at: the sentinel is > any real position, so the first call wins...
    kv.start_at(NL, 7);
    assert_eq!(kv.kv_start[NL], 7);
    // ...a later position must NOT push the start forward...
    kv.start_at(NL, 9);
    assert_eq!(kv.kv_start[NL], 7);
    // ...but an earlier one does (C: `kv_start[li] > p`).
    kv.start_at(NL, 3);
    assert_eq!(kv.kv_start[NL], 3);
    std::fs::remove_dir_all(&dir).ok();
}

#[test]
fn missing_dense_tensor_is_a_clean_error() {
    // A snapshot missing lm_head must error, not panic.
    let dir = temp_dir();
    std::fs::write(dir.join("config.json"), config_json()).unwrap();
    // write only the embed tensor
    let mut header = String::from("{");
    let numel = VOCAB * D;
    header.push_str(&format!(
        "\"model.embed_tokens.weight\":{{\"dtype\":\"F32\",\"shape\":[{numel}],\"data_offsets\":[0,{}]}}",
        numel * 4
    ));
    header.push('}');
    let hb = header.as_bytes();
    let mut f = File::create(dir.join("model.safetensors")).unwrap();
    f.write_all(&(hb.len() as u64).to_le_bytes()).unwrap();
    f.write_all(hb).unwrap();
    f.write_all(&vec![0u8; numel * 4]).unwrap();

    assert!(load_model_with(&dir, LoadOptions::default()).is_err());
    std::fs::remove_dir_all(&dir).ok();
}
