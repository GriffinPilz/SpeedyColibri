# colibrì → Rust port status (SpeedyColibri)

This repository is being converted from the original C engine (`c/`) to Rust.
The Rust workspace lives at the repo root (`Cargo.toml` + `crates/`); the C
sources stay in the tree as the reference implementation until each module is
fully ported and validated.

**Goal:** a full 1:1 rewrite of the whole engine (CPU forward pass, kernels,
grammar, tokenizer, backends, tools) that runs GLM-5.2 token-exact against the
same model files.

**Deployment target:** a Docker container on **NVIDIA DGX Spark** (GB10 Grace
Blackwell, aarch64 + CUDA), single node first, designed to split across nodes
**expert-parallel** over an RDMA/RoCE link. Backend priority is **CUDA → CPU
(Grace/aarch64 NEON)**; Apple-Silicon Metal is off the critical path. See
[DEPLOYMENT.md](DEPLOYMENT.md).

**Approach:** bottom-up. Leaf modules first (no dependencies, easy to validate),
then the forward pass, then the GPU backends and tooling. Every ported module
ships with unit tests; the C code is the oracle.

## Workspace layout

| Rust crate | Ports (C source) | Status |
|---|---|---|
| `colibri-json` | `c/json.h` | ✅ ported + tested |
| `colibri-core` | `c/glm.c` (Cfg/QT), `c/st.h` (dtypes), `c/tier.h` | ✅ ported + tested |
| `colibri-safetensors` | `c/st.h` | ✅ ported + tested (¹) |
| `colibri-tokenizer` | `c/tok.h`, `c/tok_unicode.h` | ✅ ported + tested |
| `colibri-kernels` | `c/glm.c` (idot/quant/dequant) | 🟡 **NEON int4·f32 / int8·f32 dots (5.4× vs scalar)** wired into `matmul_qt`; int2 + IDOT int8-activation path pending |
| `colibri-grammar` | `c/grammar.h`, `c/schema_gbnf.h` | ⬜ skeleton |
| `colibri-engine` | `c/glm.c` (forward, MoE, MLA, KV, gen) | 🟡 **full CPU forward pass + greedy decode + resident expert cache**; DSA/SIMD/speculation deferred |
| `colibri-backend` | `c/backend_loader.c`, `backend_cuda.*` | 🟡 CPU trait live; **CUDA FFI binding GPU-verified on a DGX Spark** (GB10, sm_121, CUDA 13 — builds/links/inits, GPU matmul smoke test passes); not yet wired into forward; Metal deprioritized |
| `colibri-cluster` | (new — multi-node) | 🟡 expert-parallel sharding tested; RDMA transport stubbed |
| `coli` (bin) | `c/glm.c` `main()`, `c/coli` launcher | 🟡 tokenize/config/load/gen/repack work; chat (tokenizer-wired)/serve pending |
| Docker / deploy | (new — DGX Spark) | ✅ aarch64+CUDA image, compose, entrypoint |
| — | `c/olmoe.c` | ⬜ not started (second model variant) |
| — | `c/openai_server.py`, `c/tools/*`, `web/` | ⬜ not started |

¹ `colibri-safetensors` omits the `posix_fadvise(DONTNEED)` + `O_DIRECT` twin-fd
behavior for now (performance/RSS, not correctness). Reintroduce via
`libc::posix_fadvise` behind a `cfg(unix)` gate.

Legend: ✅ done · 🟡 partial · ⬜ not started

## Milestone order

1. **Foundation (done):** json, config, dtypes, quant container, tier eviction,
   safetensors, tokenizer, sampling. All tested.
2. **Kernels:** ✅ **aarch64 NEON** `dot_i4_f32` / `dot_i8_f32` (two-accumulator
   `vfmaq`/`vaddvq`, mirroring the C `matmul_i4`/`matmul_q`) wired into
   `matmul_qt` — **5.4× over scalar** at n=6144 (17.2 vs 3.2 GFLOP/s), byte-exact
   with the scalar reference, harness still passes. The f32 path stays scalar
   (byte-exact with the C f32 kernel). ⬜ pending: int2 NEON, the IDOT
   int8-activation dot (`dot_i8i8`), and an x86 AVX2 path for dev boxes.
3. **CPU forward pass (`colibri-engine`):**
   - ✅ primitives: RMSNorm/LayerNorm, interleaved-partial RoPE, `matmul_qt`
     (exact scalar for f32/int8/int4/int2), `embed_row`, weight quantizers, and
     the `qt_from_disk` weight loader — all unit-tested
   - ✅ weight-loader driver (`load_model`): materializes embed/lm_head/final_norm
     + per-layer attention & dense-MLP / shared-expert / router by GLM tensor name,
     detects DSA/MTP; tested end-to-end on a synthetic tiny model. (Expert LRU
     sizing still ⬜ — experts stream at forward time.)
   - ✅ MLA attention (`attention.rs`) with compressed KV-cache (`KvCache`) — both
     the reconstruction reference and the DeepSeek weight-absorption decode core;
     tested that the two agree, and that batched prefill == step-by-step decode.
     (DSA sparse-indexer top-k selection still ⬜ — this is the dense path.)
   - ✅ MoE block (`moe.rs`): sigmoid router + bias top-K (noaux_tc), SwiGLU
     experts, shared expert; experts streamed via an `ExpertProvider` whose
     `ShardsExpertProvider` checks `colibri-cluster` ownership (single-node local
     now, RDMA-remote later). Router/FFN/shared tested independently.
   - ✅ resident expert cache (`cache.rs`): `ExpertCache` keeps loaded experts in
     RAM (returns `Arc<Expert>`), LFRU eviction (`colibri-core::tier`) only when
     over a byte budget, optional pinned hot-store; hit/miss/eviction stats.
     `coli gen` shows e.g. `32 hits / 2 misses` across decode.
   - ✅ pinned hot-store warm-up / AUTOPIN (`usage.rs` + `cache.warm_pin`): a
     persistent `.coli_usage` history (C-compatible `layer eid count` format,
     `UsageHistory`) is loaded at startup and the globally-hottest experts are
     pinned resident (`COLI_PIN_GB` budget); the session's selections are merged
     back and saved. Port of `usage_load`/`usage_save`/`pin_load`.
   - ✅ parallel expert preload (`preload.rs`): `preload_parallel` reads experts
     **directly from the original safetensors** across `num_cores` threads (no
     repack, no second copy) — sorted by on-disk offset, contiguous chunk per
     thread, per-thread byte budget → resident `PreloadStore`, zero per-token disk
     I/O. `COLI_PRELOAD=1 coli gen` uses it. Optional `coli repack` still exists
     (writes core-sharded blobs + a `Manifest` for max sequentiality;
     `COLI_PRELOAD=<dir>` uses those). Both tested byte-identical to the disk path
     incl. generation output.
   - ✅ capacity/KV planning (`capacity` module + `coli capacity <snap> [ram] [ctx]`):
     18 MB/expert int4; KV = 175.5 KB/token (compressed MLA, 78 layers) so 256K
     ctx ≈ 44 GB KV. One 128 GB Spark: ~3,980 experts at 256K ctx (~6,000 at ≤47K).
     (CACHE_ROUTE/top-p routing variants ⬜.)
   - ✅ per-layer forward (`forward.rs`): in_ln → MLA attention → residual →
     post_ln → MoE/dense → residual, then final norm + lm_head; greedy decode
     loop (`generate_greedy`). Runs end-to-end on a synthetic model with routed
     experts (integration test + `coli gen`); deterministic. **The CPU forward
     pass generates.** Next: wire a tokenizer into `coli chat`, then perf (SIMD,
     expert LRU cache) and the deferred pieces (DSA, speculation, CUDA).
4. **CUDA (Blackwell) backend:** primary GPU tier for DGX Spark.
   - ✅ FFI binding (`colibri-backend/src/cuda.rs` + `build.rs`): compiles
     `c/backend_cuda.cu` with nvcc (`--features cuda`, `CUDA_ARCH=native`/`sm_121`),
     links `cudart`+`stdc++`; safe wrappers for init/mem_info/tensor_upload/matmul/
     expert_mlp/lifecycle; `CudaBackend::probe()` (init-based); `coli backend`.
     **GPU-VERIFIED on a DGX Spark** (GB10, sm_121, CUDA 13.0): builds+links, inits
     the GPU (130.7 GB VRAM), and a GPU matmul smoke test (`cargo test -p
     colibri-backend --features cuda`) passes. build.rs skips nvcc gracefully on
     non-CUDA hosts.
   - ✅ wired into the forward pass (`colibri-engine/src/gpu.rs`, feature `cuda`):
     `matmul_qt` routes GPU-eligible (resident) weights — dense + preloaded
     experts (`QTensor.gpu_eligible`) — to `coli_cuda_matmul` (upload-once, reuse
     by data-pointer slot), CPU fallback otherwise. **Validated on the GB10**:
     `COLI_PRELOAD=1 coli gen` runs matmuls on the GPU and produces the SAME
     tokens as the CPU path. **Measured 17.9–18.6× vs 1-core CPU-NEON** on an
     int4 `[8192,6144]` matmul (429–448 vs 24 GFLOP/s).
   - ✅ fused expert FFN (`coli_cuda_expert_mlp`): `moe::ffn` routes resident
     experts / shared / dense MLP to one on-device `down(silu(gate·x)⊙up·x)` call
     (one upload+download vs 3 GEMMs). Validated on the GB10 (same tokens as CPU;
     `36 fused FFNs` replaced ~108 matmuls). **Measured 19.2×** vs 1-core CPU-NEON
     at hidden 6144 / moe_inter 2048 (165 µs vs 3171 µs per expert).
   - ✅ GPU MLA attention (`coli_cuda_attention_absorb_batch`): `attention_with`
     runs the weight-absorption core on the GPU for resident kv_b (ctx from
     q+latent+rope), then o_proj (also GPU). Validated on the GB10 (same tokens;
     `18 attention cores`, kv_b reconstruction gone). **Measured 31.9×** vs
     1-core CPU-NEON at H=64/T=2048 (1349 µs vs 43 ms), matching the CPU to
     `max|Δ|≈3.5e-10`. **The whole hot path (projections + attention + expert FFN
     + lm_head) is on-device.**
   - ✅ persistent device KV (`coli_cuda_attention_absorb_kvdev` + `DeviceKv`
     shadow): decode uploads only the new KV row per token; validated same tokens.
     **Finding:** on the GB10's *unified* memory the H2D re-upload is a fast local
     memcpy (~57 GB/s), so this is only ~1.07× vs re-uploading — the attention
     *kernel* dominates decode here, not KV transfer. (Would matter more on a
     discrete PCIe GPU.)
   - 🟡 attention-kernel profiling (the decode bottleneck, ~2.5 ms/core at T=4096):
     source analysis — the absorb kernel launches one block per (head, query), 64
     blocks × 256 threads for decode (~12% occupancy). Bumped to 1024 threads
     (`ATTN_TPB`, all 5 absorb launches) → ~2% (occupancy wasn't the main limit).
     A multi-head-per-block variant (read the shared latent once, reuse across G
     heads) was **correct but ~1.5× slower** — halving the block count hurt more
     than the redundant reads. **Finding: the kernel is parallelism-sensitive on
     the GB10, not memory-bandwidth-bound**; the real win is flash-attention-style
     T-parallelism (more blocks + online softmax + coalescing fixes), which needs
     `ncu` perf-counter access (admin-gated on the shared DGX) to do well.
   - ⬜ next: VRAM eviction for the full model; end-to-end tok/s on a real-sized
     model; the flash-attention absorb rewrite; then port kernels FFI→Rust.
     (Metal deprioritized — not a target.)
5. **Speculative + grammar:** MTP head, grammar-forced drafts, GBNF engine,
   schema→GBNF.
6. **Persistence & serving:** KV-cache `.coli_kv`, `.coli_usage` learning cache,
   OpenAI-compatible server, web dashboard.
7. **Multi-node (expert-parallel):** real `num_nodes > 1` sharding + RDMA/RoCE
   transport over ConnectX-7 (GPUDirect); split-model on-disk layout per node.
8. **Second model:** `olmoe.c`.

## Validation strategy

- Unit tests per crate (the C behavior is the spec). 87 tests currently pass.
- **C-vs-Rust harness (`scripts/validate_c_vs_rust.py`, see [VALIDATION.md](VALIDATION.md)):**
  runs both engines on the same tiny synthetic model (real GLM architecture, no
  torch / no 370 GB model) and diffs greedy generation + teacher-forcing at f32
  and int4. **Currently PASSES** — byte-exact at f32, token-exact at int4, on
  both modes. The C engine is forced onto the exact CPU path
  (`IDOT=0 ABSORB=0 DRAFT=0`). Since the C engine is itself token-exact vs a
  `transformers` oracle, this transitively validates the Rust dense path.
- Not yet covered by the harness: DSA indexer, MTP speculation, CUDA (unported).

## Notes

- `scripts/gen_unicode.py` regenerates `crates/colibri-tokenizer/src/unicode_tables.rs`
  from `c/tok_unicode.h` — do not hand-edit the generated file.
- Clippy style-lints (e.g. `needless_range_loop`) are deferred where the Rust
  deliberately mirrors a C index loop; they do not affect correctness.
- Comments in `c/glm.c` are mixed Italian/English (upstream); ported comments
  are in English.
