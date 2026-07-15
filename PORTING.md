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
| `colibri-engine` | `c/glm.c` (forward, MoE, MLA, KV, gen) | 🟡 primitives + dense weight-loader driver done (`load_model`); attention+MoE assembly next |
| `colibri-backend` | `c/backend_loader.c`, `backend_cuda.*` | 🟡 CPU trait live; CUDA primary (stub), Metal deprioritized |
| `colibri-cluster` | (new — multi-node) | 🟡 expert-parallel sharding tested; RDMA transport stubbed |
| `coli` (bin) | `c/glm.c` `main()`, `c/coli` launcher | 🟡 tokenize/config/load work; chat/serve pending |
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
   - ⬜ MLA attention with weight absorption + compressed KV-cache
   - ⬜ MoE block (sigmoid router / noaux_tc, shared expert, streaming experts) —
     route each expert through `colibri-cluster` (`is_local`/`owner`) so the
     single-node path and the future split share one code path
   - ⬜ single-token decode loop → wire up `coli chat`
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

- Unit tests per crate (the C behavior is the spec). 63 tests currently pass.
- Byte-exactness: the C engine validates token-exact against a `transformers`
  oracle (TF 32/32, greedy 20/20). The Rust engine must reproduce the C engine's
  greedy stream under `DRAFT=0 IDOT=0 COLI_CUDA=0`.
- Cross-check: keep the C `coli` buildable (`make -C c`) to diff outputs during
  the port.

## Notes

- `scripts/gen_unicode.py` regenerates `crates/colibri-tokenizer/src/unicode_tables.rs`
  from `c/tok_unicode.h` — do not hand-edit the generated file.
- Clippy style-lints (e.g. `needless_range_loop`) are deferred where the Rust
  deliberately mirrors a C index loop; they do not affect correctness.
- Comments in `c/glm.c` are mixed Italian/English (upstream); ported comments
  are in English.
