//! Model and per-layer weight structures — port of the `Model`, `Layer`,
//! `ESlot`, and `KVState` structs from `c/glm.c`.
//!
//! # Status: SKELETON
//!
//! The field layout mirrors the C structs so the loader and forward pass can be
//! filled in against a known shape. Buffers that are hot-path detail (profiling
//! counters, GPU shadow caches) are elided until their subsystem is ported.

use colibri_core::{Config, QTensor};
use colibri_safetensors::Shards;

/// A transformer layer: MLA attention (dense, quantized) plus either a dense MLP
/// (`sparse == false`) or the MoE block (`sparse == true`).
#[derive(Default)]
pub struct Layer {
    pub in_ln: Vec<f32>,
    pub post_ln: Vec<f32>,

    // MLA (dense, quantized)
    pub q_a: QTensor,
    pub q_b: QTensor,
    pub kv_a: QTensor,
    pub kv_b: QTensor,
    pub o: QTensor,
    pub q_a_ln: Vec<f32>,
    pub kv_a_ln: Vec<f32>,

    pub sparse: bool,

    // dense mlp (sparse == false)
    pub gate_proj: QTensor,
    pub up_proj: QTensor,
    pub down_proj: QTensor,

    // moe (sparse == true) — router weights stay f32 (numerically sensitive)
    pub router: Vec<f32>,
    pub router_bias: Vec<f32>,
    pub sh_gate: QTensor,
    pub sh_up: QTensor,
    pub sh_down: QTensor,
}

/// The compressed MLA KV-cache for one layer's context.
///
/// Only the normalized latent `[kv_lora]` and the rotary key `[qk_rope]` are
/// kept per token (576 vs 32768 values/token); k_nope and value are
/// reconstructed on the fly via `kv_b`. This is what makes the context tractable
/// in ~10 GB (64 heads, no GQA).
#[derive(Default)]
pub struct KvState {
    /// per-token latent, `[max_t * kv_lora]` flattened per layer
    pub latent: Vec<f32>,
    /// per-token rotary key, `[max_t * qk_rope]`
    pub k_rot: Vec<f32>,
    pub max_t: usize,
    /// first valid position (MTP partial caches start mid-sequence)
    pub kv_start: usize,
}

/// A fully loaded model.
///
/// Fields present so the loader has a target; heavy runtime state (expert LRU,
/// pinned hot-store, DSA indexer, MTP head, profiling) is added as each
/// subsystem is ported. See PORTING.md.
pub struct Model {
    pub cfg: Config,
    pub shards: Shards,
    /// bits/param for experts and for the dense part
    pub ebits: i32,
    pub dbits: i32,

    pub embed: QTensor,
    pub lm_head: QTensor,
    pub final_norm: Vec<f32>,
    pub layers: Vec<Layer>,

    /// whether the DSA lightning indexer weights are present
    pub has_dsa: bool,
    /// whether the native MTP speculative head is present
    pub has_mtp: bool,
}

impl Model {
    /// Convenience accessor for the config.
    pub fn config(&self) -> &Config {
        &self.cfg
    }
}
