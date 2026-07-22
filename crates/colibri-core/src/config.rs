//! Model hyperparameters — port of the `Cfg` struct and `load_cfg` from
//! `c/glm.c`.
//!
//! Loaded from a snapshot's `config.json`. The validation ranges (`CKR` in C)
//! are a single choke point: `config.json` arrives from untrusted mirrors, so
//! hostile dimensions must not pass this point and reach a downstream alloc.
//!
//! Two architectures are supported, discriminated by [`Config::arch`]:
//!   - [`Arch::GlmMoeDsa`] — GLM-5.2: MLA attention + DSA lightning indexer.
//!   - [`Arch::MinimaxM3`] — MiniMax-M3: standard GQA (partial RoPE, per-head
//!     QK-norm), Gemma-norm, clamped SwiGLU, sigmoid+bias MoE router. The
//!     GQA head geometry reuses the `qk_nope`/`qk_rope`/`v_head` fields
//!     (`qk_rope` = the rotary sub-dim, `qk_nope` = head_dim − rotary).

use colibri_json::Json;
use std::path::Path;

pub const MAX_STOP_IDS: usize = 8;
pub const MAX_LAYERS_IDX: usize = 128;

/// Which model architecture a [`Config`] describes. Selects the attention core,
/// the MoE router, the activation, and the norm variant in the forward pass.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Arch {
    /// GLM-5.2: Multi-head Latent Attention + DSA sparse indexer.
    GlmMoeDsa,
    /// MiniMax-M3: grouped-query attention (partial RoPE, per-head QK-norm),
    /// Gemma-style RMSNorm, clamped OpenAI-SwiGLU, sigmoid+bias top-k router.
    MinimaxM3,
}

/// GLM-5.2 / MiniMax-M3 hyperparameters.
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

    // ---- architecture discriminator + MiniMax-M3-specific fields ----
    // (GLM leaves these at the defaults set in `from_json_glm`.)
    /// Which architecture this config describes.
    pub arch: Arch,
    /// GQA key/value head count (MiniMax-M3). For GLM/MLA this mirrors `n_heads`.
    pub n_kv_heads: i32,
    /// Shared-expert intermediate size. MiniMax-M3 sets it explicitly
    /// (`shared_intermediate_size`); GLM derives `n_shared * moe_inter`.
    pub shared_inter: i32,
    /// Per-head QK RMSNorm applied before RoPE (MiniMax-M3 `use_qk_norm`).
    pub qk_norm: bool,
    /// Gemma-style `(1 + weight)` RMSNorm (MiniMax-M3 `use_gemma_norm`).
    pub gemma_norm: bool,
    /// Clamped OpenAI-SwiGLU activation (MiniMax-M3 `hidden_act == "swigluoai"`);
    /// `false` = plain SiLU-gated SwiGLU (GLM).
    pub swiglu_oai: bool,
    /// SwiGLU gate scale (`swiglu_alpha`, MiniMax-M3) — used only when `swiglu_oai`.
    pub swiglu_alpha: f32,
    /// SwiGLU clamp limit (`swiglu_limit`, MiniMax-M3) — used only when `swiglu_oai`.
    pub swiglu_limit: f32,
    /// Sigmoid expert scoring with an additive routing bias (MiniMax-M3
    /// `scoring_func == "sigmoid"` + `e_score_correction_bias`); `false` = GLM.
    pub sigmoid_route: bool,
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

/// Read integer field `k` from object `r` (0 if absent/non-integer).
fn gi_in(r: &Json, k: &str) -> i32 {
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

/// Collect `eos_token_id` (scalar or array) into `out`.
fn parse_stop_ids(r: &Json, out: &mut Vec<i32>) {
    match r.get("eos_token_id") {
        Some(Json::Num(n)) => out.push(*n as i32),
        Some(Json::Arr(a)) => {
            for v in a.iter().take(MAX_STOP_IDS) {
                if let Some(id) = v.as_i64() {
                    out.push(id as i32);
                }
            }
        }
        _ => {}
    }
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

    /// Build a `Config` from an already-parsed `config.json` root object,
    /// dispatching on architecture. MiniMax-M3 nests its hyperparameters under
    /// `text_config` and advertises `model_type == "minimax_m3_vl"`; everything
    /// else is treated as GLM-5.2.
    pub fn from_json(r: &Json) -> Result<Config, ConfigError> {
        let is_minimax = r.get("model_type").and_then(Json::as_str) == Some("minimax_m3_vl")
            || r.get("text_config").is_some();
        if is_minimax {
            Config::from_json_minimax(r)
        } else {
            Config::from_json_glm(r)
        }
    }

    /// GLM-5.2 (`glm_moe_dsa`) parse — the original path.
    fn from_json_glm(r: &Json) -> Result<Config, ConfigError> {
        let gi = |k: &str| gi_in(r, k);
        let mut c = Config {
            hidden: gi("hidden_size"),
            n_layers: gi("num_hidden_layers"),
            n_heads: gi("num_attention_heads"),
            n_experts: gi("n_routed_experts"),
            topk: gi("num_experts_per_tok"),
            moe_inter: gi("moe_intermediate_size"),
            dense_inter: gi("intermediate_size"),
            first_dense: gi("first_k_dense_replace"),
            q_lora: gi("q_lora_rank"),
            kv_lora: gi("kv_lora_rank"),
            qk_nope: gi("qk_nope_head_dim"),
            qk_rope: gi("qk_rope_head_dim"),
            qk_head: 0,
            v_head: gi("v_head_dim"),
            n_shared: gi("n_shared_experts"),
            vocab: gi("vocab_size"),
            max_ctx: gi("max_position_embeddings"),
            n_group: gi("n_group"),
            topk_group: gi("topk_group"),
            norm_topk: r.get("norm_topk_prob").and_then(Json::as_bool).unwrap_or(false),
            stop_ids: Vec::new(),
            index_topk: gi("index_topk"),
            index_nh: gi("index_n_heads"),
            index_hd: gi("index_head_dim"),
            idx_type: Vec::new(),
            eps: r.get("rms_norm_eps").and_then(Json::as_f64).unwrap_or(1e-5) as f32,
            theta: 10000.0,
            attn_scale: 0.0,
            routed_scale: r
                .get("routed_scaling_factor")
                .and_then(Json::as_f64)
                .unwrap_or(1.0) as f32,
            // GLM defaults for the MiniMax-only fields.
            arch: Arch::GlmMoeDsa,
            n_kv_heads: gi("num_attention_heads"),
            shared_inter: gi("n_shared_experts") * gi("moe_intermediate_size"),
            qk_norm: false,
            gemma_norm: false,
            swiglu_oai: false,
            swiglu_alpha: 0.0,
            swiglu_limit: 0.0,
            sigmoid_route: false,
        };

        // rope theta lives under rope_parameters.rope_theta
        if let Some(th) = r
            .get("rope_parameters")
            .and_then(|rp| rp.get("rope_theta"))
            .and_then(Json::as_f64)
        {
            c.theta = th as f32;
        }

        parse_stop_ids(r, &mut c.stop_ids);

        // Per-layer indexer type: explicit list, or a freq/offset formula.
        let n_layers_capped = (c.n_layers.max(0) as usize).min(MAX_LAYERS_IDX);
        c.idx_type = vec![false; n_layers_capped];
        {
            let it = r.get("indexer_types").and_then(Json::as_array);
            let mut freq = gi("index_topk_freq");
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

        c.validate_common()?;
        // GLM/MLA-specific ranges.
        ckr!("q_lora_rank", c.q_lora, 0, 1 << 20);
        ckr!("kv_lora_rank", c.kv_lora, 1, 1 << 20);
        ckr!("qk_nope_head_dim", c.qk_nope, 1, 1 << 16);
        ckr!("qk_rope_head_dim", c.qk_rope, 1, 1 << 16);
        ckr!("index_topk", c.index_topk, 0, 1 << 20);
        ckr!("index_n_heads", c.index_nh, 0, 1024);
        ckr!("index_head_dim", c.index_hd, 0, 1 << 16);
        Ok(c)
    }

    /// MiniMax-M3 (`minimax_m3_vl`) parse. Hyperparameters live under
    /// `text_config`; the vision tower is ignored (text-only inference). The GQA
    /// head geometry is folded onto `qk_nope`/`qk_rope`/`v_head`: `qk_rope` is the
    /// rotary sub-dimension (`rotary_dim`) and `qk_nope = head_dim − rotary_dim`.
    fn from_json_minimax(r: &Json) -> Result<Config, ConfigError> {
        let t = r
            .get("text_config")
            .ok_or_else(|| ConfigError::Parse("minimax_m3: missing text_config".into()))?;
        let gt = |k: &str| gi_in(t, k);

        let head_dim = gt("head_dim");
        let rotary_dim = gt("rotary_dim");
        // Partial RoPE: rotate the first `rotary_dim` of each head, leave the rest.
        let qk_rope = rotary_dim;
        let qk_nope = head_dim - rotary_dim;

        // first-dense count = leading zeros of `moe_layer_freq` (dense layers precede
        // the MoE stack); fall back to `first_k_dense_replace` if the list is absent.
        let first_dense = match t.get("moe_layer_freq").and_then(Json::as_array) {
            Some(arr) => arr
                .iter()
                .take_while(|v| v.as_i64() == Some(0))
                .count() as i32,
            None => gt("first_k_dense_replace"),
        };

        let act = t.get("hidden_act").and_then(Json::as_str).unwrap_or("");
        let scoring = t.get("scoring_func").and_then(Json::as_str).unwrap_or("");

        let mut c = Config {
            hidden: gt("hidden_size"),
            n_layers: gt("num_hidden_layers"),
            n_heads: gt("num_attention_heads"),
            n_experts: gt("num_local_experts"),
            topk: gt("num_experts_per_tok"),
            moe_inter: gt("intermediate_size"), // expert FFN width
            dense_inter: gt("dense_intermediate_size"),
            first_dense,
            q_lora: 0, // GQA: no query LoRA
            kv_lora: 0, // GQA: no latent KV
            qk_nope,
            qk_rope,
            qk_head: head_dim,
            v_head: head_dim,
            n_shared: gt("n_shared_experts"),
            vocab: gt("vocab_size"),
            max_ctx: gt("max_position_embeddings"),
            n_group: 1,
            topk_group: 1,
            // MiniMax normalizes the top-k gate weights before `routed_scaling`.
            norm_topk: true,
            stop_ids: Vec::new(),
            // Sparse attention is deferred (dense GQA for the MVP): no DSA indexer.
            index_topk: 0,
            index_nh: 0,
            index_hd: 0,
            idx_type: vec![false; (gt("num_hidden_layers").max(0) as usize).min(MAX_LAYERS_IDX)],
            eps: t.get("rms_norm_eps").and_then(Json::as_f64).unwrap_or(1e-6) as f32,
            theta: t.get("rope_theta").and_then(Json::as_f64).unwrap_or(10000.0) as f32,
            attn_scale: if head_dim > 0 { 1.0 / (head_dim as f32).sqrt() } else { 0.0 },
            routed_scale: t
                .get("routed_scaling_factor")
                .and_then(Json::as_f64)
                .unwrap_or(1.0) as f32,
            arch: Arch::MinimaxM3,
            n_kv_heads: gt("num_key_value_heads"),
            shared_inter: gt("shared_intermediate_size"),
            qk_norm: t.get("use_qk_norm").and_then(Json::as_bool).unwrap_or(false),
            gemma_norm: t.get("use_gemma_norm").and_then(Json::as_bool).unwrap_or(false),
            swiglu_oai: act == "swigluoai",
            swiglu_alpha: t.get("swiglu_alpha").and_then(Json::as_f64).unwrap_or(1.702) as f32,
            swiglu_limit: t.get("swiglu_limit").and_then(Json::as_f64).unwrap_or(7.0) as f32,
            sigmoid_route: scoring == "sigmoid",
        };

        // eos/stop ids may sit in text_config or at the root.
        parse_stop_ids(t, &mut c.stop_ids);
        if c.stop_ids.is_empty() {
            parse_stop_ids(r, &mut c.stop_ids);
        }

        c.validate_common()?;
        // GQA-specific ranges.
        ckr!("head_dim", head_dim, 1, 1 << 16);
        ckr!("rotary_dim", rotary_dim, 1, head_dim);
        ckr!("num_key_value_heads", c.n_kv_heads, 1, c.n_heads);
        ckr!("shared_intermediate_size", c.shared_inter, 0, 1 << 24);
        Ok(c)
    }

    /// Validation shared by both architectures (the C `CKR` choke point).
    fn validate_common(&self) -> Result<(), ConfigError> {
        ckr!("hidden_size", self.hidden, 1, 1 << 20);
        ckr!("num_hidden_layers", self.n_layers, 1, 128);
        ckr!("num_attention_heads", self.n_heads, 1, 1024);
        ckr!("n_routed_experts", self.n_experts, 1, 4096);
        ckr!("num_experts_per_tok", self.topk, 1, 64);
        ckr!("moe_intermediate_size", self.moe_inter, 1, 1 << 20);
        ckr!("intermediate_size", self.dense_inter, 1, 1 << 24);
        ckr!("first_k_dense_replace", self.first_dense, 0, self.n_layers);
        ckr!("v_head_dim", self.v_head, 1, 1 << 16);
        ckr!("n_shared_experts", self.n_shared, 0, 64);
        ckr!("vocab_size", self.vocab, 1, 1 << 24);
        Ok(())
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

    // A minimal MiniMax-M3-shaped config (values from nvidia/MiniMax-M3-NVFP4).
    fn minimax_json() -> Json {
        let text = r#"{
            "model_type": "minimax_m3_vl",
            "text_config": {
                "hidden_size": 6144,
                "intermediate_size": 3072,
                "num_hidden_layers": 60,
                "num_attention_heads": 64,
                "num_key_value_heads": 4,
                "head_dim": 128,
                "vocab_size": 200064,
                "max_position_embeddings": 1048576,
                "rms_norm_eps": 1e-06,
                "use_gemma_norm": true,
                "rope_theta": 5000000,
                "rotary_dim": 64,
                "partial_rotary_factor": 0.5,
                "hidden_act": "swigluoai",
                "use_qk_norm": true,
                "dense_intermediate_size": 12288,
                "shared_intermediate_size": 3072,
                "num_local_experts": 128,
                "num_experts_per_tok": 4,
                "n_shared_experts": 1,
                "scoring_func": "sigmoid",
                "use_routing_bias": true,
                "moe_layer_freq": [0,0,0,1,1,1],
                "swiglu_alpha": 1.702,
                "swiglu_limit": 7.0,
                "routed_scaling_factor": 2.0,
                "eos_token_id": [200020]
            }
        }"#;
        Json::parse(text).unwrap()
    }

    #[test]
    fn loads_glm_shape() {
        let c = Config::from_json(&glm_json()).unwrap();
        assert_eq!(c.arch, Arch::GlmMoeDsa);
        assert_eq!(c.hidden, 6144);
        assert_eq!(c.n_layers, 78);
        assert_eq!(c.qk_head, 128 + 64);
        assert_eq!(c.stop_ids, vec![151329, 151336, 151338]);
        assert!(c.norm_topk);
        assert!(!c.gemma_norm && !c.swiglu_oai && !c.sigmoid_route);
        assert!((c.attn_scale - 1.0 / (192f32).sqrt()).abs() < 1e-6);
        assert_eq!(c.idx_type.len(), 78);
    }

    #[test]
    fn loads_minimax_shape() {
        let c = Config::from_json(&minimax_json()).unwrap();
        assert_eq!(c.arch, Arch::MinimaxM3);
        assert_eq!(c.hidden, 6144);
        assert_eq!(c.n_layers, 60);
        assert_eq!(c.n_heads, 64);
        assert_eq!(c.n_kv_heads, 4);
        // GQA head geometry folded onto qk_nope/qk_rope: head_dim 128, rotary 64.
        assert_eq!(c.qk_head, 128);
        assert_eq!(c.qk_rope, 64);
        assert_eq!(c.qk_nope, 64);
        assert_eq!(c.v_head, 128);
        assert_eq!(c.n_experts, 128);
        assert_eq!(c.topk, 4);
        assert_eq!(c.moe_inter, 3072);
        assert_eq!(c.dense_inter, 12288);
        assert_eq!(c.shared_inter, 3072);
        assert_eq!(c.first_dense, 3); // three leading zeros in moe_layer_freq
        assert_eq!(c.vocab, 200064);
        assert_eq!(c.max_ctx, 1048576);
        assert!(c.qk_norm && c.gemma_norm && c.swiglu_oai && c.sigmoid_route);
        assert!((c.swiglu_alpha - 1.702).abs() < 1e-6);
        assert!((c.swiglu_limit - 7.0).abs() < 1e-6);
        assert!((c.routed_scale - 2.0).abs() < 1e-6);
        assert!((c.attn_scale - 1.0 / (128f32).sqrt()).abs() < 1e-6);
        assert!((c.theta - 5_000_000.0).abs() < 1.0);
        assert_eq!(c.stop_ids, vec![200020]);
        // Sparse attention deferred: no DSA indexer for the MVP.
        assert_eq!(c.index_topk, 0);
        assert_eq!(c.idx_type.len(), 60);
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
