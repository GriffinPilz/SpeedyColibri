//! Model hyperparameters — port of the `Cfg` struct and `load_cfg` from
//! `c/glm.c`.
//!
//! Loaded from a snapshot's `config.json`. The validation ranges (`CKR` in C)
//! are a single choke point: `config.json` arrives from untrusted mirrors, so
//! hostile dimensions must not pass this point and reach a downstream alloc.

use colibri_json::Json;
use std::path::Path;

pub const MAX_STOP_IDS: usize = 8;
pub const MAX_LAYERS_IDX: usize = 128;

/// GLM-5.2 (`glm_moe_dsa`) hyperparameters.
#[derive(Debug, Clone)]
pub struct Config {
    pub hidden: i32,
    pub n_layers: i32,
    pub n_heads: i32,
    pub n_experts: i32,
    pub topk: i32,
    pub moe_inter: i32,
    pub dense_inter: i32,
    pub first_dense: i32,
    pub q_lora: i32,
    pub kv_lora: i32,
    pub qk_nope: i32,
    pub qk_rope: i32,
    /// derived: `qk_nope + qk_rope`
    pub qk_head: i32,
    pub v_head: i32,
    pub n_shared: i32,
    pub vocab: i32,
    /// model's max context (`max_position_embeddings`); 0 if the config omits it
    pub max_ctx: i32,
    pub n_group: i32,
    pub topk_group: i32,
    pub norm_topk: bool,
    /// stop tokens (GLM-5.2 has three: endoftext, user, observation)
    pub stop_ids: Vec<i32>,
    /// DSA lightning indexer params
    pub index_topk: i32,
    pub index_nh: i32,
    pub index_hd: i32,
    /// per-layer indexer type: `true` = full (compute), `false` = shared (reuse)
    pub idx_type: Vec<bool>,
    pub eps: f32,
    pub theta: f32,
    pub attn_scale: f32,
    pub routed_scale: f32,
}

/// Error from loading/validating a config.
#[derive(Debug)]
pub enum ConfigError {
    Io(std::io::Error),
    Parse(String),
    /// A field is outside its accepted `[lo, hi]` range (the `CKR` checks).
    Range {
        name: &'static str,
        value: i64,
        lo: i64,
        hi: i64,
    },
    Unsupported(String),
}

impl std::fmt::Display for ConfigError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            ConfigError::Io(e) => write!(f, "io error: {e}"),
            ConfigError::Parse(s) => write!(f, "parse error: {s}"),
            ConfigError::Range {
                name,
                value,
                lo,
                hi,
            } => write!(f, "config: {name}={value} is outside [{lo},{hi}]"),
            ConfigError::Unsupported(s) => write!(f, "unsupported config: {s}"),
        }
    }
}

impl std::error::Error for ConfigError {}

fn gi(r: &Json, k: &str) -> i32 {
    r.get(k).and_then(Json::as_i64).unwrap_or(0) as i32
}

macro_rules! ckr {
    ($name:literal, $v:expr, $lo:expr, $hi:expr) => {{
        let v = $v as i64;
        if v < ($lo as i64) || v > ($hi as i64) {
            return Err(ConfigError::Range {
                name: $name,
                value: v,
                lo: $lo as i64,
                hi: $hi as i64,
            });
        }
    }};
}

impl Config {
    /// Load and validate `<snap>/config.json`.
    pub fn load(snap: impl AsRef<Path>) -> Result<Config, ConfigError> {
        let path = snap.as_ref().join("config.json");
        let text = std::fs::read_to_string(&path).map_err(ConfigError::Io)?;
        let root = Json::parse(&text)
            .ok_or_else(|| ConfigError::Parse(format!("{}: empty or invalid", path.display())))?;
        Config::from_json(&root)
    }

    /// Build a `Config` from an already-parsed `config.json` root object.
    pub fn from_json(r: &Json) -> Result<Config, ConfigError> {
        let mut c = Config {
            hidden: gi(r, "hidden_size"),
            n_layers: gi(r, "num_hidden_layers"),
            n_heads: gi(r, "num_attention_heads"),
            n_experts: gi(r, "n_routed_experts"),
            topk: gi(r, "num_experts_per_tok"),
            moe_inter: gi(r, "moe_intermediate_size"),
            dense_inter: gi(r, "intermediate_size"),
            first_dense: gi(r, "first_k_dense_replace"),
            q_lora: gi(r, "q_lora_rank"),
            kv_lora: gi(r, "kv_lora_rank"),
            qk_nope: gi(r, "qk_nope_head_dim"),
            qk_rope: gi(r, "qk_rope_head_dim"),
            qk_head: 0,
            v_head: gi(r, "v_head_dim"),
            n_shared: gi(r, "n_shared_experts"),
            vocab: gi(r, "vocab_size"),
            max_ctx: gi(r, "max_position_embeddings"),
            n_group: gi(r, "n_group"),
            topk_group: gi(r, "topk_group"),
            norm_topk: r.get("norm_topk_prob").and_then(Json::as_bool).unwrap_or(false),
            stop_ids: Vec::new(),
            index_topk: gi(r, "index_topk"),
            index_nh: gi(r, "index_n_heads"),
            index_hd: gi(r, "index_head_dim"),
            idx_type: Vec::new(),
            eps: r.get("rms_norm_eps").and_then(Json::as_f64).unwrap_or(1e-5) as f32,
            theta: 10000.0,
            attn_scale: 0.0,
            routed_scale: r
                .get("routed_scaling_factor")
                .and_then(Json::as_f64)
                .unwrap_or(1.0) as f32,
        };

        // rope theta lives under rope_parameters.rope_theta
        if let Some(th) = r
            .get("rope_parameters")
            .and_then(|rp| rp.get("rope_theta"))
            .and_then(Json::as_f64)
        {
            c.theta = th as f32;
        }

        // Stop tokens: a scalar or an array. GLM-5.2 ships three.
        match r.get("eos_token_id") {
            Some(Json::Num(n)) => c.stop_ids.push(*n as i32),
            Some(Json::Arr(a)) => {
                for v in a.iter().take(MAX_STOP_IDS) {
                    if let Some(id) = v.as_i64() {
                        c.stop_ids.push(id as i32);
                    }
                }
            }
            _ => {}
        }

        // Per-layer indexer type: explicit list, or a freq/offset formula.
        let n_layers_capped = (c.n_layers.max(0) as usize).min(MAX_LAYERS_IDX);
        c.idx_type = vec![false; n_layers_capped];
        {
            let it = r.get("indexer_types").and_then(Json::as_array);
            let mut freq = gi(r, "index_topk_freq");
            if freq < 1 {
                freq = 1;
            }
            let off = r.get("index_skip_topk_offset").and_then(Json::as_i64).unwrap_or(2) as i32;
            for (i, slot) in c.idx_type.iter_mut().enumerate() {
                let ii = i as i32;
                if let Some(arr) = it {
                    if let Some(s) = arr.get(i).and_then(Json::as_str) {
                        *slot = s == "full";
                        continue;
                    }
                }
                let v = (ii - off + 1).max(0);
                *slot = v % freq == 0;
            }
        }

        c.qk_head = c.qk_nope + c.qk_rope;
        c.attn_scale = 1.0 / (c.qk_head as f32).sqrt();

        if c.n_group != 1 {
            return Err(ConfigError::Unsupported(
                "this engine requires n_group=1 (GLM-5.2)".into(),
            ));
        }

        // Validation choke point (PR #25). Same ranges as the C `CKR` macros.
        ckr!("hidden_size", c.hidden, 1, 1 << 20);
        ckr!("num_hidden_layers", c.n_layers, 1, 128);
        ckr!("num_attention_heads", c.n_heads, 1, 1024);
        ckr!("n_routed_experts", c.n_experts, 1, 4096);
        ckr!("num_experts_per_tok", c.topk, 1, 64);
        ckr!("moe_intermediate_size", c.moe_inter, 1, 1 << 20);
        ckr!("intermediate_size", c.dense_inter, 1, 1 << 24);
        ckr!("first_k_dense_replace", c.first_dense, 0, c.n_layers);
        ckr!("q_lora_rank", c.q_lora, 0, 1 << 20);
        ckr!("kv_lora_rank", c.kv_lora, 1, 1 << 20);
        ckr!("qk_nope_head_dim", c.qk_nope, 1, 1 << 16);
        ckr!("qk_rope_head_dim", c.qk_rope, 1, 1 << 16);
        ckr!("v_head_dim", c.v_head, 1, 1 << 16);
        ckr!("n_shared_experts", c.n_shared, 0, 64);
        ckr!("vocab_size", c.vocab, 1, 1 << 24);
        ckr!("index_topk", c.index_topk, 0, 1 << 20);
        ckr!("index_n_heads", c.index_nh, 0, 1024);
        ckr!("index_head_dim", c.index_hd, 0, 1 << 16);

        Ok(c)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    // A minimal GLM-5.2-shaped config (values from the README architecture notes).
    fn glm_json() -> Json {
        let text = r#"{
            "hidden_size": 6144,
            "num_hidden_layers": 78,
            "num_attention_heads": 64,
            "n_routed_experts": 256,
            "num_experts_per_tok": 8,
            "moe_intermediate_size": 2048,
            "intermediate_size": 12288,
            "first_k_dense_replace": 3,
            "q_lora_rank": 2048,
            "kv_lora_rank": 512,
            "qk_nope_head_dim": 128,
            "qk_rope_head_dim": 64,
            "v_head_dim": 128,
            "n_shared_experts": 1,
            "vocab_size": 151552,
            "n_group": 1,
            "topk_group": 1,
            "norm_topk_prob": true,
            "rms_norm_eps": 1e-5,
            "routed_scaling_factor": 2.5,
            "rope_parameters": {"rope_theta": 10000.0},
            "eos_token_id": [151329, 151336, 151338],
            "index_topk": 2048,
            "index_n_heads": 64,
            "index_head_dim": 128
        }"#;
        Json::parse(text).unwrap()
    }

    #[test]
    fn loads_glm_shape() {
        let c = Config::from_json(&glm_json()).unwrap();
        assert_eq!(c.hidden, 6144);
        assert_eq!(c.n_layers, 78);
        assert_eq!(c.qk_head, 128 + 64);
        assert_eq!(c.stop_ids, vec![151329, 151336, 151338]);
        assert!(c.norm_topk);
        assert!((c.attn_scale - 1.0 / (192f32).sqrt()).abs() < 1e-6);
        assert_eq!(c.idx_type.len(), 78);
    }

    #[test]
    fn rejects_out_of_range() {
        let mut text = glm_json();
        if let Json::Obj(_) = &text {
            // Rebuild with a hostile layer count.
            text = Json::parse(
                &r#"{"hidden_size":6144,"num_hidden_layers":9999,"num_attention_heads":64,
                    "n_routed_experts":256,"num_experts_per_tok":8,"moe_intermediate_size":2048,
                    "intermediate_size":12288,"first_k_dense_replace":3,"q_lora_rank":2048,
                    "kv_lora_rank":512,"qk_nope_head_dim":128,"qk_rope_head_dim":64,"v_head_dim":128,
                    "n_shared_experts":1,"vocab_size":151552,"n_group":1,"index_topk":0,
                    "index_n_heads":0,"index_head_dim":0}"#
                    .to_string(),
            )
            .unwrap();
        }
        match Config::from_json(&text) {
            Err(ConfigError::Range { name, .. }) => assert_eq!(name, "num_hidden_layers"),
            other => panic!("expected range error, got {other:?}"),
        }
    }

    #[test]
    fn requires_n_group_1() {
        let text = Json::parse(r#"{"n_group": 8, "num_hidden_layers": 4}"#).unwrap();
        assert!(matches!(
            Config::from_json(&text),
            Err(ConfigError::Unsupported(_))
        ));
    }
}
