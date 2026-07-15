//! End-to-end test of the weight-loader driver (`load_model`) against a tiny,
//! synthetic GLM-5.2-shaped snapshot: a real `config.json` plus a
//! `model.safetensors` carrying every dense tensor the loader reads, by GLM name.
//!
//! Dimensions are minimal but structurally faithful (first layer dense, second
//! sparse). Routed experts are streamed, so they are intentionally absent — the
//! loader must not require them.

use colibri_engine::{load_model_with, LoadOptions};
use std::fs::File;
use std::io::Write;
use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicU64, Ordering};

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

/// The full list of dense tensors the loader reads, with element counts.
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
        }
    }
    t
}

/// Write a single-shard safetensors file with all tensors as small f32 values.
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

    let m = load_model_with(&dir, LoadOptions { dbits: 4, ebits: 4 }).expect("load_model");

    // structure
    assert_eq!(m.cfg.n_layers, NL as i32);
    assert_eq!(m.layers.len(), NL);
    assert!(!m.layers[0].sparse, "layer 0 should be dense");
    assert!(m.layers[1].sparse, "layer 1 should be sparse (MoE)");
    assert_eq!(m.dbits, 4);
    assert_eq!(m.ebits, 4);
    assert!(!m.has_mtp);
    assert!(!m.has_dsa);

    // I/O boundary: embed/lm_head loaded at io_bits=4 -> int4 (fmt 2), shape [vocab, D]
    assert_eq!((m.embed.o, m.embed.i), (VOCAB as i32, D as i32));
    assert_eq!(m.embed.fmt_code, 2);
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
