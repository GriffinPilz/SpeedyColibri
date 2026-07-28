//! End-to-end Kimi-K3 forward pass on a tiny synthetic container.
//!
//! K3 is the only arch whose layers do not carry a hidden state: they thread
//! `(prefix_sum, block_residual)` through the whole stack, so it has its own driver
//! (`kimi_forward`) rather than a per-layer function inside the shared loop. That driver
//! is what this file exercises — through the real loader, on a container with every K3
//! feature present (KDA and gated-MLA layers, a dense layer, latent MoE with fused
//! shared experts, situ, and the attention residuals at both sublayers and the model
//! level).
//!
//! What this can and cannot prove. The weights are synthetic, so nothing here says the
//! outputs are *right* for the real checkpoint. What it pins is that the whole stack
//! loads and runs, that nothing produces NaNs, and that prefill agrees with incremental
//! decode — i.e. the per-sequence state (KDA conv + delta matrix, MLA KV) threads
//! correctly across calls.
//!
//! It does NOT pin the attention-residual formula or the accumulator's block-boundary
//! rules. Measured, not assumed: deleting the accumulator reset, and disabling the
//! snapshot entirely, both leave every test in this file passing — prefill and decode
//! simply agree on the same wrong answer. Those semantics are covered by the
//! `AttnResState` and `apply_attn_res` unit tests in `forward.rs`, which do fail on
//! exactly those mutations. Keep both halves.

use colibri_engine::{forward, load_model_with, KvCache, LoadOptions, ShardsExpertProvider};
use std::fs::File;
use std::io::Write;
use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicU64, Ordering};

const D: usize = 8;
const NL: usize = 5;
/// Block boundaries land on layers 0, 2 and 4 → 3 saved candidates by the end. Small
/// enough to reason about, but >1 so the softmax mixes something real.
const BLOCK: usize = 2;
/// 1-indexed, as in the real config: checkpoint layers 1 and 4 are gated MLA, and
/// 0, 2, 3 are KDA. Deliberately NOT layer 0, so the first layer (which is also the
/// dense-FFN layer and the first block boundary) is a KDA layer.
const FULL_ATTN_1IDX: [usize; 2] = [2, 5];

const H: usize = 2; // MLA heads
const QK_NOPE: usize = 4;
const QK_ROPE: usize = 2;
const QK_HEAD: usize = QK_NOPE + QK_ROPE;
const V_HEAD: usize = 4;
const Q_LORA: usize = 4;
const KV_LORA: usize = 4;
const C_MLA: usize = H * V_HEAD;

const KDA_H: usize = 2;
const KDA_HD: usize = 4;
const C_KDA: usize = KDA_H * KDA_HD;
const D_CONV: usize = 2;
/// `f_a_proj`'s rank. The loader reads it off the tensor's own shape rather than any
/// config field, so make it differ from `KDA_HD` to prove that.
const F_RANK: usize = 3;

const E: usize = 4;
const TOPK: usize = 2;
const MOE_INTER: usize = 2;
/// `routed_expert_hidden_size` — routed experts run in THIS space, not at `hidden`.
const DL: usize = 4;
const N_SHARED: usize = 1;
const S_I: usize = MOE_INTER * N_SHARED;
const DENSE_INTER: usize = 8;
const FIRST_DENSE: usize = 1;
const VOCAB: usize = 10;

fn temp_dir() -> PathBuf {
    static N: AtomicU64 = AtomicU64::new(0);
    let base = std::env::var("TMPDIR").unwrap_or_else(|_| "/tmp".into());
    let mut p = PathBuf::from(base);
    p.push(format!("colibri-k3-{}-{}", std::process::id(), N.fetch_add(1, Ordering::Relaxed)));
    std::fs::create_dir_all(&p).unwrap();
    p
}

fn is_kda(i: usize) -> bool {
    !FULL_ATTN_1IDX.contains(&(i + 1))
}

fn config_json() -> String {
    let attn = FULL_ATTN_1IDX.map(|v| v.to_string()).join(",");
    format!(
        r#"{{"model_type":"kimi_k3","architectures":["KimiK3ForConditionalGeneration"],
        "text_config":{{
          "hidden_size":{D},"num_hidden_layers":{NL},"num_attention_heads":{H},
          "num_key_value_heads":{H},"num_experts":{E},"num_experts_per_token":{TOPK},
          "num_shared_experts":{N_SHARED},"moe_intermediate_size":{MOE_INTER},
          "intermediate_size":{DENSE_INTER},"routed_expert_hidden_size":{DL},
          "first_k_dense_replace":{FIRST_DENSE},"q_lora_rank":{Q_LORA},
          "kv_lora_rank":{KV_LORA},"qk_nope_head_dim":{QK_NOPE},
          "qk_rope_head_dim":{QK_ROPE},"v_head_dim":{V_HEAD},"vocab_size":{VOCAB},
          "max_position_embeddings":64,"rms_norm_eps":1e-5,"rope_theta":10000.0,
          "moe_renormalize":true,"moe_router_activation_func":"sigmoid",
          "num_expert_group":1,"topk_group":1,"routed_scaling_factor":1.0,
          "eos_token_id":9,"mla_use_nope":true,"mla_use_output_gate":true,
          "hidden_act":"situ","activation_situ_beta":4.0,
          "activation_situ_linear_beta":25.0,"attn_res_block_size":{BLOCK},
          "linear_attn_config":{{"head_dim":{KDA_HD},"num_heads":{KDA_H},
            "short_conv_kernel_size":{D_CONV},"full_attn_layers":[{attn}]}}}}}}"#
    )
}

fn tensor_list() -> Vec<(String, usize)> {
    let mut t: Vec<(String, usize)> = vec![
        ("model.embed_tokens.weight".into(), VOCAB * D),
        ("lm_head.weight".into(), VOCAB * D),
        ("model.norm.weight".into(), D),
        // The model-level attention residual — K3 only.
        ("model.output_attn_res_norm.weight".into(), D),
        ("model.output_attn_res_proj.weight".into(), D),
    ];
    for i in 0..NL {
        let p = |s: &str| format!("model.layers.{i}.{s}");
        t.push((p("input_layernorm.weight"), D));
        t.push((p("post_attention_layernorm.weight"), D));
        // Attention-residual score vectors, one pair per sublayer.
        t.push((p("self_attention_res_norm.weight"), D));
        t.push((p("self_attention_res_proj.weight"), D));
        t.push((p("mlp_res_norm.weight"), D));
        t.push((p("mlp_res_proj.weight"), D));

        let c = if is_kda(i) { C_KDA } else { C_MLA };
        t.push((p("self_attn.o_proj.weight"), D * c));
        // Output gate: present on BOTH mixers on K3.
        t.push((p("self_attn.g_proj.weight"), c * D));

        if is_kda(i) {
            t.push((p("self_attn.q_proj.weight"), C_KDA * D));
            t.push((p("self_attn.k_proj.weight"), C_KDA * D));
            t.push((p("self_attn.v_proj.weight"), C_KDA * D));
            t.push((p("self_attn.b_proj.weight"), KDA_H * D));
            t.push((p("self_attn.f_a_proj.weight"), F_RANK * D));
            t.push((p("self_attn.f_b_proj.weight"), C_KDA * F_RANK));
            t.push((p("self_attn.q_conv1d.weight"), C_KDA * D_CONV));
            t.push((p("self_attn.k_conv1d.weight"), C_KDA * D_CONV));
            t.push((p("self_attn.v_conv1d.weight"), C_KDA * D_CONV));
            // per KEY-DIM, not per head — K3 differs from Kimi-Linear here.
            t.push((p("self_attn.A_log"), KDA_HD));
            t.push((p("self_attn.dt_bias"), C_KDA));
            t.push((p("self_attn.o_norm.weight"), KDA_HD));
        } else {
            t.push((p("self_attn.q_a_proj.weight"), Q_LORA * D));
            t.push((p("self_attn.q_a_layernorm.weight"), Q_LORA));
            t.push((p("self_attn.q_b_proj.weight"), H * QK_HEAD * Q_LORA));
            t.push((p("self_attn.kv_a_proj_with_mqa.weight"), (KV_LORA + QK_ROPE) * D));
            t.push((p("self_attn.kv_a_layernorm.weight"), KV_LORA));
            t.push((p("self_attn.kv_b_proj.weight"), H * (QK_NOPE + V_HEAD) * KV_LORA));
        }

        if i < FIRST_DENSE {
            t.push((p("mlp.gate_proj.weight"), DENSE_INTER * D));
            t.push((p("mlp.up_proj.weight"), DENSE_INTER * D));
            t.push((p("mlp.down_proj.weight"), D * DENSE_INTER));
        } else {
            t.push((p("mlp.gate.weight"), E * D));
            t.push((p("mlp.gate.e_score_correction_bias"), E));
            t.push((p("mlp.fc1_latent_proj.weight"), DL * D));
            t.push((p("mlp.fc2_latent_proj.weight"), D * DL));
            t.push((p("mlp.routed_expert_norm.weight"), DL));
            // ONE fused shared MLP, `n_shared * moe_inter` wide.
            t.push((p("mlp.shared_experts.gate_proj.weight"), S_I * D));
            t.push((p("mlp.shared_experts.up_proj.weight"), S_I * D));
            t.push((p("mlp.shared_experts.down_proj.weight"), D * S_I));
            // Routed experts live in LATENT space: [moe_inter, DL] / [DL, moe_inter].
            for e in 0..E {
                let pe = |s: &str| format!("model.layers.{i}.mlp.experts.{e}.{s}.weight");
                t.push((pe("gate_proj"), MOE_INTER * DL));
                t.push((pe("up_proj"), MOE_INTER * DL));
                t.push((pe("down_proj"), DL * MOE_INTER));
            }
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
    load_model_with(dir, LoadOptions { dbits: 8, ebits: 8 }).expect("load K3 container")
}

/// The container loads and a full stack runs: KDA layers, gated-MLA layers, the dense
/// layer, latent MoE, and the attention residuals at both sublayers plus the model
/// level. Output must be finite — a NaN here means one of the seven K3-specific pieces
/// is producing garbage that the shapes could not catch.
#[test]
fn kimi_stack_runs_end_to_end() {
    let dir = temp_dir();
    let model = tiny_model(&dir);
    assert_eq!(model.cfg.attn_res_block_size as usize, BLOCK);
    assert_eq!(model.output_attn_res_norm.len(), D, "model-level attn-res must load");
    assert_eq!(model.output_attn_res_proj.len(), D);

    let provider = ShardsExpertProvider::new(&model.shards, &model.cfg, 8);
    let prompt = [1i32, 5, 2, 7];
    let mut kv = KvCache::for_model(&model, 32);
    let mut hidden = vec![0f32; prompt.len() * D];
    forward(&model, &mut kv, &provider, &prompt, 0, &mut hidden).expect("K3 forward");

    assert!(hidden.iter().all(|v| v.is_finite()), "K3 forward produced non-finite hidden state");
    assert!(hidden.iter().any(|&v| v != 0.0), "K3 forward produced an all-zero hidden state");

    // Deterministic.
    let mut kv2 = KvCache::for_model(&model, 32);
    let mut h2 = vec![0f32; prompt.len() * D];
    forward(&model, &mut kv2, &provider, &prompt, 0, &mut h2).expect("K3 forward");
    assert_eq!(hidden, h2, "K3 forward must be deterministic");
}

/// A single-shot prefill and a run of one-token decode steps must agree.
///
/// This is the load-bearing test for the driver. It only holds if BOTH kinds of state
/// thread correctly: the per-sequence recurrent state (KDA conv + delta matrix, MLA's
/// KV) carried in `kv` ACROSS calls, and the attention-residual state (`prefix_sum`,
/// `block_residual`) rebuilt fresh WITHIN each call. The second is the subtle one — it
/// works only because a token's attention-residual mix reads its own candidates and
/// nothing from its neighbours. Had the mix crossed positions, prefill would see a full
/// candidate set where decode sees one token's, and these would diverge.
#[test]
fn kimi_prefill_matches_incremental_decode() {
    let dir = temp_dir();
    let model = tiny_model(&dir);
    let provider = ShardsExpertProvider::new(&model.shards, &model.cfg, 8);
    let prompt = [3i32, 1, 4, 1, 5];

    let mut kv_p = KvCache::for_model(&model, 32);
    let mut prefill = vec![0f32; prompt.len() * D];
    forward(&model, &mut kv_p, &provider, &prompt, 0, &mut prefill).expect("prefill");

    // Guard against a vacuous pass: if the stack collapsed to a position-independent
    // constant, matching decode row-for-row would prove nothing. Token 0 and token 2
    // carry different ids AND different histories, so they must differ.
    assert!(
        prefill[..D].iter().zip(&prefill[2 * D..3 * D]).any(|(a, b)| (a - b).abs() > 1e-4),
        "hidden states are position-independent; the equivalence check would be vacuous"
    );

    let mut kv_d = KvCache::for_model(&model, 32);
    let mut step = vec![0f32; D];
    for (i, &tok) in prompt.iter().enumerate() {
        forward(&model, &mut kv_d, &provider, &[tok], i, &mut step).expect("decode step");
        let want = &prefill[i * D..(i + 1) * D];
        for j in 0..D {
            assert!(
                (step[j] - want[j]).abs() < 1e-5,
                "token {i} dim {j}: decode {} vs prefill {}",
                step[j],
                want[j]
            );
        }
    }
}

/// Kimi-K3 has no per-layer forward, and asking for one must fail loudly.
///
/// `layer_forward` would otherwise fall through to the shared two-sublayer driver and
/// compute a plausible-looking result with ordinary residual adds and no
/// attention-residual mixing — wrong, with no shape error and no NaN to notice.
#[test]
fn kimi_rejects_the_per_layer_driver() {
    let dir = temp_dir();
    let model = tiny_model(&dir);
    let provider = ShardsExpertProvider::new(&model.shards, &model.cfg, 8);
    let mut kv = KvCache::for_model(&model, 32);
    let (mut x, mut nrm, mut tmp) = (vec![0f32; D], vec![0f32; D], vec![0f32; D]);
    let mut sel = None;

    let err = colibri_engine::layer_forward(
        &model,
        &mut kv,
        &provider,
        &model.layers[0],
        0,
        &mut x,
        1,
        0,
        &mut nrm,
        &mut tmp,
        &mut sel,
    )
    .expect_err("K3 must refuse a per-layer forward");
    assert_eq!(err.kind(), std::io::ErrorKind::Unsupported);
    assert!(
        err.to_string().contains("kimi_forward"),
        "the error should point at the real driver, got: {err}"
    );
}

/// Routed experts are read at `routed_expert_hidden_size`, not at `hidden`.
///
/// K3's MoE block computes with experts in the latent space, so the loader has to read
/// them there too. The fixture makes the two widths differ (DL=4 vs D=8), which turns a
/// mismatch into a load failure instead of a silent misread — and this container would
/// load fine if the widths happened to match, which is why they are set apart here.
#[test]
fn kimi_routed_experts_load_in_latent_space() {
    use colibri_engine::ExpertProvider;
    let dir = temp_dir();
    let model = tiny_model(&dir);
    assert!(
        model.cfg.arch.routed_experts_are_latent(),
        "K3 must be classed as a latent-MoE arch"
    );
    let provider = ShardsExpertProvider::new(&model.shards, &model.cfg, 8);
    // Layer FIRST_DENSE is the first sparse layer.
    let e = provider.expert(FIRST_DENSE, 0).expect("expert load");
    assert_eq!(e.gate.o as usize, MOE_INTER);
    assert_eq!(e.gate.i as usize, DL, "expert input width must be moe_latent");
    assert_eq!(e.down.o as usize, DL, "expert output width must be moe_latent");
    assert_eq!(e.down.i as usize, MOE_INTER);
}

/// Every dense weight K3 actually computes with must be GPU-eligible.
///
/// This is the `gpu_eligible` trap, which has now bitten three arches: a resident weight
/// missing from the list in `mark_gpu_eligible` silently takes the single-threaded CPU
/// matmul. It cost 84% of an M3 prefill (q/k/v projections), 94% of Nemotron's mamba
/// time (in/out_proj) and 62% of its MoE phase (fc1/fc2_latent) — each found only by
/// profiling, because nothing fails and the output is identical.
///
/// K3 inherits most of the list, but `attn_gate`, `kda_b_proj`, `kda_f_a` and `kda_f_b`
/// have no analogue in any other arch. `attn_gate` alone is 8.19B params across the
/// stack — the same order as `q_b`/`kv_b`. Asserting over the whole layer rather than
/// naming the four means a future K3 tensor is covered the day it is added.
#[test]
fn kimi_dense_weights_are_all_gpu_eligible() {
    let dir = temp_dir();
    let model = tiny_model(&dir);

    let mut checked = 0usize;
    for (li, l) in model.layers.iter().enumerate() {
        // Non-Option slots: populated ones have real dimensions, defaults are 0x0.
        let fixed: [(&str, &colibri_engine::QTensor); 9] = [
            ("q_a", &l.q_a), ("q_b", &l.q_b), ("kv_a", &l.kv_a), ("kv_b", &l.kv_b),
            ("o", &l.o), ("gate_proj", &l.gate_proj), ("up_proj", &l.up_proj),
            ("down_proj", &l.down_proj), ("sh_gate", &l.sh_gate),
        ];
        for (name, t) in fixed {
            if t.o > 0 && t.i > 0 {
                assert!(t.gpu_eligible, "L{li} {name} ({}x{}) is not gpu_eligible", t.o, t.i);
                checked += 1;
            }
        }
        let opts: [(&str, &Option<colibri_engine::QTensor>); 8] = [
            ("q_proj", &l.q_proj), ("k_proj", &l.k_proj), ("v_proj", &l.v_proj),
            ("attn_gate", &l.attn_gate), ("kda_b_proj", &l.kda_b_proj),
            ("kda_f_a", &l.kda_f_a), ("kda_f_b", &l.kda_f_b),
            ("fc1_latent", &l.fc1_latent),
        ];
        for (name, t) in opts {
            if let Some(t) = t {
                assert!(t.gpu_eligible, "L{li} {name} ({}x{}) is not gpu_eligible", t.o, t.i);
                checked += 1;
            }
        }
    }
    // Guard against the assertions vacuously passing on an empty model.
    assert!(checked >= 40, "expected many weights to check, saw {checked}");
}
