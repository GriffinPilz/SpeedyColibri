//! End-to-end test of the Nemotron-H latent-space MoE mixer ([`nemotron_moe`]).
//!
//! The gateless ReLU² activation is a **process-global** ([`set_activation`], a
//! `OnceLock`), so this lives in its own integration binary: setting it here cannot
//! collide with the lib's SwiGLU unit tests (a different process). We verify that the
//! latent flow `out = fc2·Σ_k w_k·expert_k(fc1·x) + shared(x)` — with routed experts and
//! the shared expert both gateless ReLU² — matches a from-primitives reference.

use colibri_engine::{
    matmul_f32, matmul_qt, nemotron_moe, qtensor_from_f32, route, set_activation, Config, Expert,
    ExpertProvider, Layer, QTensor,
};
use std::collections::HashMap;
use std::fs::File;
use std::io::{self, Write};
use std::path::PathBuf;
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::Arc;

const HIDDEN: usize = 4;
const LATENT: usize = 3; // moe_latent_size
const MOE_INTER: usize = 5;
const SHARED_INTER: usize = 6;
const N_EXPERTS: usize = 4;
const TOPK: usize = 2;
const S: usize = 2; // tokens

// Deterministic pseudo-random-ish weight, distinct per `seed`.
fn wv(n: usize, seed: usize) -> Vec<f32> {
    (0..n).map(|i| (((i + seed) as f32 * 0.41).sin() * 0.5) + 0.05).collect()
}

// Gateless ReLU² FFN reference: `down(relu(up·x)²)`, mirroring `nemotron_moe`'s experts
// and shared expert. Takes the QTensors by reference straight off an `Expert`/`Layer`.
fn relu2(up: &QTensor, down: &QTensor, x: &[f32], nr: usize) -> Vec<f32> {
    let inter = up.o as usize;
    let mut uu = vec![0f32; nr * inter];
    matmul_qt(&mut uu, x, up, nr);
    for u in uu.iter_mut() {
        let r = u.max(0.0);
        *u = r * r;
    }
    let dn = down.o as usize;
    let mut out = vec![0f32; nr * dn];
    matmul_qt(&mut out, &uu, down, nr);
    out
}

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

fn temp_dir() -> PathBuf {
    static N: AtomicU64 = AtomicU64::new(0);
    let base = std::env::var("TMPDIR").unwrap_or_else(|_| "/tmp".into());
    let p = PathBuf::from(base).join(format!(
        "colibri-nemomoe-{}-{}",
        std::process::id(),
        N.fetch_add(1, Ordering::Relaxed)
    ));
    std::fs::create_dir_all(&p).unwrap();
    p
}

// A single-MoE-layer Nemotron-H config (raw top-k weights: norm off, scale 1.0).
fn nemotron_cfg() -> Config {
    let dir = temp_dir();
    let json = format!(
        r#"{{
        "model_type":"nemotron_h",
        "hidden_size":{HIDDEN}, "num_hidden_layers":1,
        "num_attention_heads":2, "num_key_value_heads":1, "head_dim":2,
        "vocab_size":8, "hybrid_override_pattern":"E",
        "n_routed_experts":{N_EXPERTS}, "num_experts_per_tok":{TOPK},
        "moe_intermediate_size":{MOE_INTER}, "moe_latent_size":{LATENT},
        "moe_shared_expert_intermediate_size":{SHARED_INTER},
        "norm_topk_prob":false, "routed_scaling_factor":1.0, "mlp_hidden_act":"relu2",
        "ssm_state_size":2, "conv_kernel":2, "mamba_num_heads":2, "mamba_head_dim":2,
        "n_groups":1, "chunk_size":2, "layer_norm_epsilon":1e-5
    }}"#
    );
    File::create(dir.join("config.json")).unwrap().write_all(json.as_bytes()).unwrap();
    let cfg = Config::load(&dir).unwrap();
    std::fs::remove_dir_all(&dir).ok();
    cfg
}

#[test]
fn nemotron_moe_matches_latent_flow_reference() {
    let cfg = nemotron_cfg();
    // Gateless ReLU² for both routed + shared experts (see module note on isolation).
    set_activation(&cfg);

    // --- layer weights (f32 / exact so this tests wiring, not quantization) ---------
    let mut l = Layer::default();
    l.router = wv(N_EXPERTS * HIDDEN, 1); // f32 router
    l.router_bias = vec![0.0; N_EXPERTS];
    l.fc1_latent = Some(qtensor_from_f32(&wv(LATENT * HIDDEN, 2), LATENT, HIDDEN, 16));
    l.fc2_latent = Some(qtensor_from_f32(&wv(HIDDEN * LATENT, 3), HIDDEN, LATENT, 16));
    // Shared expert (gateless ReLU²): up_proj/down_proj on the hidden state.
    l.up_proj = qtensor_from_f32(&wv(SHARED_INTER * HIDDEN, 4), SHARED_INTER, HIDDEN, 16);
    l.down_proj = qtensor_from_f32(&wv(HIDDEN * SHARED_INTER, 5), HIDDEN, SHARED_INTER, 16);

    // Routed experts: latent-space up (LATENT->MOE_INTER) + down (MOE_INTER->LATENT), no
    // gate. The `Expert.gate` is left at its empty default and ignored under ReLU².
    let mut experts: HashMap<(usize, usize), Arc<Expert>> = HashMap::new();
    for e in 0..N_EXPERTS {
        let up = qtensor_from_f32(&wv(MOE_INTER * LATENT, 20 + e), MOE_INTER, LATENT, 16);
        let down = qtensor_from_f32(&wv(LATENT * MOE_INTER, 40 + e), LATENT, MOE_INTER, 16);
        experts.insert((0, e), Arc::new(Expert { up, down, ..Default::default() }));
    }
    let provider = MapProvider { experts };

    let x = wv(S * HIDDEN, 100);

    // --- reference: replay the flow from primitives -------------------------------
    let mut logits = vec![0f32; S * N_EXPERTS];
    matmul_f32(&mut logits, &x, &l.router, S, HIDDEN, N_EXPERTS);
    let mut h_lat = vec![0f32; S * LATENT];
    matmul_qt(&mut h_lat, &x, l.fc1_latent.as_ref().unwrap(), S);
    let mut moe_lat = vec![0f32; S * LATENT];
    for t in 0..S {
        let (idx, w) = route(&cfg, &logits[t * N_EXPERTS..(t + 1) * N_EXPERTS], &l.router_bias);
        assert_eq!(idx.len(), TOPK);
        for (j, &e) in idx.iter().enumerate() {
            let ex = &provider.experts[&(0, e)];
            let eo = relu2(&ex.up, &ex.down, &h_lat[t * LATENT..(t + 1) * LATENT], 1);
            for c in 0..LATENT {
                moe_lat[t * LATENT + c] += w[j] * eo[c];
            }
        }
    }
    let mut expect = vec![0f32; S * HIDDEN];
    matmul_qt(&mut expect, &moe_lat, l.fc2_latent.as_ref().unwrap(), S);
    let sh = relu2(&l.up_proj, &l.down_proj, &x, S);
    for i in 0..S * HIDDEN {
        expect[i] += sh[i];
    }

    // --- the mixer under test ------------------------------------------------------
    let mut out = vec![0f32; S * HIDDEN];
    nemotron_moe(&cfg, &l, 0, &x, S, &mut out, &provider).unwrap();

    for i in 0..S * HIDDEN {
        assert!(
            (out[i] - expect[i]).abs() < 1e-4,
            "at {i}: nemotron_moe {} vs reference {}",
            out[i],
            expect[i]
        );
    }
    assert!(out.iter().any(|v| v.abs() > 1e-6), "moe produced all-zero output");
}
