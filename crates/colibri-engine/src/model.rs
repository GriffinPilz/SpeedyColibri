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

/// The compressed MLA KV-cache — port of the `Lc`/`Rc` per-layer buffers in
/// `c/glm.c`.
///
/// Only the normalized latent `[kv_lora]` and the rotary key `[qk_rope]` are kept
/// per token (576 vs 32768 values/token for GLM-5.2); k_nope and value are
/// reconstructed on the fly via `kv_b`. This is what makes the context tractable
/// in ~10 GB (64 heads, no GQA).
pub struct KvCache {
    pub max_t: usize,
    kv_lora: usize,
    qk_rope: usize,
    /// per-layer latent buffer, each `[max_t * kv_lora]`
    latent: Vec<Vec<f32>>,
    /// per-layer rotary-key buffer, each `[max_t * qk_rope]`
    k_rot: Vec<Vec<f32>>,
    /// first valid position per layer (MTP partial caches start mid-sequence)
    pub kv_start: Vec<usize>,
    /// device-side KV shadow (persistent-KV GPU decode path); lazily allocated
    #[cfg(feature = "cuda")]
    dev: Option<crate::gpu::DeviceKv>,
}

impl KvCache {
    /// Allocate a cache for `n_layers` layers holding up to `max_t` tokens.
    pub fn new(n_layers: usize, kv_lora: usize, qk_rope: usize, max_t: usize) -> KvCache {
        KvCache {
            max_t,
            kv_lora,
            qk_rope,
            latent: vec![vec![0.0; max_t * kv_lora]; n_layers],
            k_rot: vec![vec![0.0; max_t * qk_rope]; n_layers],
            kv_start: vec![0; n_layers],
            #[cfg(feature = "cuda")]
            dev: None,
        }
    }

    /// Sync the device KV shadow for `layer` up to `tk` rows and return the
    /// device `(latent, rope)` base pointers. Uploads only the missing rows.
    #[cfg(feature = "cuda")]
    pub fn sync_device(
        &mut self,
        layer: usize,
        pos_base: usize,
        tk: usize,
    ) -> Option<(*const f32, *const f32)> {
        let n_layers = self.latent.len();
        let (max_t, kvl, r) = (self.max_t, self.kv_lora, self.qk_rope);
        let dev = self
            .dev
            .get_or_insert_with(|| crate::gpu::DeviceKv::new(n_layers, max_t));
        dev.sync(layer, &self.latent[layer], &self.k_rot[layer], kvl, r, pos_base, tk)
    }

    pub fn kv_lora(&self) -> usize {
        self.kv_lora
    }
    pub fn qk_rope(&self) -> usize {
        self.qk_rope
    }

    /// Normalized-latent row for `(layer, pos)`.
    pub fn latent_row(&self, layer: usize, pos: usize) -> &[f32] {
        &self.latent[layer][pos * self.kv_lora..(pos + 1) * self.kv_lora]
    }
    pub fn latent_row_mut(&mut self, layer: usize, pos: usize) -> &mut [f32] {
        &mut self.latent[layer][pos * self.kv_lora..(pos + 1) * self.kv_lora]
    }

    /// Roped k_rot row for `(layer, pos)`.
    pub fn krot_row(&self, layer: usize, pos: usize) -> &[f32] {
        &self.k_rot[layer][pos * self.qk_rope..(pos + 1) * self.qk_rope]
    }
    pub fn krot_row_mut(&mut self, layer: usize, pos: usize) -> &mut [f32] {
        &mut self.k_rot[layer][pos * self.qk_rope..(pos + 1) * self.qk_rope]
    }

    /// Contiguous latent rows `[start, end)` for a layer — a single slice the
    /// batched `kv_b` reconstruction multiplies against.
    pub fn latent_rows(&self, layer: usize, start: usize, end: usize) -> &[f32] {
        &self.latent[layer][start * self.kv_lora..end * self.kv_lora]
    }

    /// Contiguous roped-key rows `[start, end)` for a layer.
    pub fn krot_rows(&self, layer: usize, start: usize, end: usize) -> &[f32] {
        &self.k_rot[layer][start * self.qk_rope..end * self.qk_rope]
    }
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
