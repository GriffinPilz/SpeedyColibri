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
| `colibri-kernels` | `c/glm.c` (idot/quant/dequant) | 🟡 scalar reference + `qrow_i8`-exact activation quant; SIMD pending |
| `colibri-grammar` | `c/grammar.h`, `c/schema_gbnf.h` | ⬜ skeleton |
| `colibri-engine` | `c/glm.c` (forward, MoE, MLA, KV, gen) | 🟡 **full CPU forward pass + greedy decode runs end-to-end**; DSA/SIMD/expert-cache/speculation deferred |
| `colibri-backend` | `c/backend_loader.c`, `backend_cuda.*` | 🟡 CPU trait live; CUDA primary (stub), Metal deprioritized |
| `colibri-cluster` | (new — multi-node) | 🟡 expert-parallel sharding tested; RDMA transport stubbed |
| `coli` (bin) | `c/glm.c` `main()`, `c/coli` launcher | 🟡 tokenize/config/load/gen work; chat (tokenizer-wired)/serve pending |
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
2. **Kernels:** int8/int4 dot + the shape-dependent rounding that makes quantized
   output byte-exact. SIMD target is **aarch64 NEON** (Grace CPU, the DGX Spark
   CPU fallback); the AVX2 (`maddubs`) path is kept for x86 dev boxes. The scalar
   reference in `colibri-kernels` is the oracle.
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
     `coli gen` shows e.g. `32 hits / 2 misses` across decode. `capacity` module
     + `coli capacity` size residency (GLM-5.2: 18 MB/expert, ~6.2k experts per
     128 GB Spark ≈ 33%, ~4 nodes to hold all). (CACHE_ROUTE/top-p variants ⬜.)
   - ✅ per-layer forward (`forward.rs`): in_ln → MLA attention → residual →
     post_ln → MoE/dense → residual, then final norm + lm_head; greedy decode
     loop (`generate_greedy`). Runs end-to-end on a synthetic model with routed
     experts (integration test + `coli gen`); deterministic. **The CPU forward
     pass generates.** Next: wire a tokenizer into `coli chat`, then perf (SIMD,
     expert LRU cache) and the deferred pieces (DSA, speculation, CUDA).
4. **CUDA (Blackwell) backend:** primary GPU tier for DGX Spark — bind
   `c/backend_cuda.cu` via FFI first, then port; target sm_121. (Metal is
   deprioritized — not a deployment target.)
5. **Speculative + grammar:** MTP head, grammar-forced drafts, GBNF engine,
   schema→GBNF.
6. **Persistence & serving:** KV-cache `.coli_kv`, `.coli_usage` learning cache,
   OpenAI-compatible server, web dashboard.
7. **Multi-node (expert-parallel):** real `num_nodes > 1` sharding + RDMA/RoCE
   transport over ConnectX-7 (GPUDirect); split-model on-disk layout per node.
8. **Second model:** `olmoe.c`.

## Validation strategy

- Unit tests per crate (the C behavior is the spec). 75 tests currently pass.
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
