//! Expert-parallel correctness for the Nemotron-H latent MoE ([`nemotron_moe`]).
//!
//! Nemotron's routed experts run in `moe_latent` space, so its EP split ships the
//! post-`fc1` latent activation rather than the hidden state. This asserts the sharded
//! result is identical to the single-node one: node 0 owns experts {0,1}, node 1 owns
//! {2,3}, and node 1's shard is served over a real TCP loopback whose handler is the
//! same `compute_experts_partial` a `coli worker` runs.
//!
//! Its own integration binary because BOTH `set_activation` (gateless ReLU²) and
//! `set_cluster` are process-global `OnceLock`s — installing a 2-node cluster here must
//! not leak into the single-node latent-flow test in `nemotron_moe.rs`.

use colibri_cluster::{serve_experts, ExpertResponse, ExpertSharding, NodeId, TcpTransport};
use colibri_engine::{
    compute_experts_partial, nemotron_moe, qtensor_from_f32, set_activation, set_cluster,
    ClusterCtx, Config, Expert, ExpertProvider, Layer,
};
use std::collections::HashMap;
use std::fs::File;
use std::io::{self, Write};
use std::path::PathBuf;
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::Arc;

const HIDDEN: usize = 4;
const LATENT: usize = 3;
const MOE_INTER: usize = 5;
const SHARED_INTER: usize = 6;
const N_EXPERTS: usize = 4;
const TOPK: usize = 2;
const S: usize = 2;

fn wv(n: usize, seed: usize) -> Vec<f32> {
    (0..n).map(|i| (((i + seed) as f32 * 0.41).sin() * 0.5) + 0.05).collect()
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
        "colibri-nemoep-{}-{}",
        std::process::id(),
        N.fetch_add(1, Ordering::Relaxed)
    ));
    std::fs::create_dir_all(&p).unwrap();
    p
}

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
fn nemotron_moe_sharded_two_nodes_equals_single_node() {
    let cfg = nemotron_cfg();
    set_activation(&cfg);

    let mut l = Layer::default();
    l.router = wv(N_EXPERTS * HIDDEN, 1);
    l.router_bias = vec![0.0; N_EXPERTS];
    l.fc1_latent = Some(qtensor_from_f32(&wv(LATENT * HIDDEN, 2), LATENT, HIDDEN, 16));
    l.fc2_latent = Some(qtensor_from_f32(&wv(HIDDEN * LATENT, 3), HIDDEN, LATENT, 16));
    l.up_proj = qtensor_from_f32(&wv(SHARED_INTER * HIDDEN, 4), SHARED_INTER, HIDDEN, 16);
    l.down_proj = qtensor_from_f32(&wv(HIDDEN * SHARED_INTER, 5), HIDDEN, SHARED_INTER, 16);

    let mut experts: HashMap<(usize, usize), Arc<Expert>> = HashMap::new();
    for e in 0..N_EXPERTS {
        let up = qtensor_from_f32(&wv(MOE_INTER * LATENT, 20 + e), MOE_INTER, LATENT, 16);
        let down = qtensor_from_f32(&wv(LATENT * MOE_INTER, 40 + e), LATENT, MOE_INTER, 16);
        experts.insert((0, e), Arc::new(Expert { up, down, ..Default::default() }));
    }
    // Both "nodes" read one map here — this test is about the split/exchange/fold math,
    // not about ownership enforcement (covered by `provider_refuses_experts_owned_by_another_node`).
    let provider = Arc::new(MapProvider { experts });

    let x = wv(S * HIDDEN, 100);

    // Reference: single node, every expert local. Must run BEFORE `set_cluster` — the
    // cluster context is a process-global OnceLock and cannot be removed once installed.
    let mut out_single = vec![0f32; S * HIDDEN];
    nemotron_moe(&cfg, &l, 0, &x, S, &mut out_single, &*provider).unwrap();
    assert!(out_single.iter().any(|v| v.abs() > 1e-6), "reference produced all-zero output");

    // 2 nodes, contiguous blocks: node 0 owns {0,1}, node 1 owns {2,3}.
    let sharding = ExpertSharding::new(2, N_EXPERTS as u32);

    // Node 1's expert server over loopback TCP — the same handler `coli worker` installs.
    // It is dimension-agnostic (`req.hidden` is the *expert input* width), so it serves
    // the latent space here without any Nemotron-specific knowledge.
    let hp = provider.clone();
    let addr = serve_experts("127.0.0.1:0".parse().unwrap(), sharding.fingerprint(), move |req| {
        let outputs = compute_experts_partial(
            &*hp,
            req.layer as usize,
            &req.experts,
            &req.weights,
            &req.activations,
            req.n_tokens,
            req.hidden,
        )
        .unwrap();
        ExpertResponse { outputs, n_tokens: req.n_tokens, hidden: req.hidden }
    })
    .unwrap();

    let mut peers = HashMap::new();
    peers.insert(NodeId(1), addr);
    let transport = TcpTransport::new(NodeId(0), peers, sharding.fingerprint());
    set_cluster(ClusterCtx { sharding, transport: Box::new(transport) });

    let mut out_sharded = vec![0f32; S * HIDDEN];
    nemotron_moe(&cfg, &l, 0, &x, S, &mut out_sharded, &*provider).unwrap();

    for i in 0..S * HIDDEN {
        assert!(
            (out_single[i] - out_sharded[i]).abs() < 1e-5,
            "at {i}: single {} vs sharded {}",
            out_single[i],
            out_sharded[i]
        );
    }
}
