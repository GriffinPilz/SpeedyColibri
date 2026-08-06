# Development

<!-- Extracted from README.md 2026-08-06 to keep the README to the six things a
reader needs first: speeds, models, how to run them, where they come from, what was
changed, and quality. Nothing here was trimmed in the move. -->

### Test suite status

`cargo test --workspace` (CPU) is **396 passing, 0 failing**. Adding `--features cuda` on a
GB10 brings up the kernel tests: **401 passing, 0 failing** as of **2026-08-05**, at cargo's
default parallelism. The suite is fully green — the two long-standing caveats that used to
live here are both resolved:

| was | resolution |
|---|---|
| `moe::…::shards_provider_loads_gateless_nemotron_expert` failed by 2.5e-4 | **The test, not the loader.** `matmul_qt` dispatches on `QTensor::gpu_eligible`; the provider sets it for NVFP4 and the in-memory reference did not, so a GPU result was being compared against a CPU one under a 1e-5 tolerance. Same binary passed with `COLI_CUDA=0`. |
| "run CUDA tests with `--test-threads=1`" | **Per-thread re-initialisation.** `gpu::available()` was a *thread-local* `OnceCell`, so every thread re-ran `coli_cuda_init` — which resets process-global state and recreates the stream underneath threads already launching on it. Init is now idempotent and `AVAIL` process-global. |

Two K3 failures listed here previously — `kimi_stack_runs_end_to_end` ("forward must be
deterministic") and `kimi_prefill_matches_incremental_decode` — also pass, fixed by
`1928094`: a device weight cache keyed by *host pointer* served streamed int8/f32 experts the
wrong weights when the pooled buffer was reused.

One trap worth knowing if you are checking this yourself: cargo **stops at the first failing
test binary** unless you pass `--no-fail-fast`, so a naive run reports one failure and hides
any others; and piping through `head` truncates the per-binary summaries that would have
shown them.


### Without Docker

The workspace has **no crates.io dependencies** (std + path crates only), so a
direct build needs only the CUDA toolkit and rustup:

```bash
# Build (~3–5 min): PREFER the wrapper — it locates nvcc, sets the arch, adds the CUDA
# lib path (cudart is under targets/<arch>-linux/lib on ARM/DGX, lib64 on x86), and
# VERIFIES the result is a CUDA binary. A plain `cargo build -p coli` WITHOUT
# `--features cuda` silently builds a CPU-only binary (`coli backend` -> cpu) that runs
# the expert FFN single-threaded, ~16-40x slower with the GPU idle — the wrapper refuses
# to produce that.
scripts/build.sh

# Or the raw command (equivalent; build.rs now finds cudart on ARM automatically):
NVCC=/usr/local/cuda/bin/nvcc CUDA_HOME=/usr/local/cuda CUDA_ARCH=sm_121 \
  cargo build --release -p coli --features cuda
# Always confirm: `coli backend` must print `backend: cuda (Cuda(0))`, not `cpu`.

# Which models are registered (scripts/models.toml) — serve any of them by name:
scripts/model.py list                     # name + arch notes (abbreviated here)
#   deepseek-v4-flash 43 layers all-MoE, 256 exp top-6 + 1 shared, MXFP4 routed;
#                     Hyper-Connections, Compressor 41/43, DSA-family Indexer 21/43…
#   glm-5.2           MLA + DSA lightning indexer, 256 experts top-8, NVFP4 routed…
#   kimi-k3           Hybrid: 93 layers = 69 KDA + 24 gated MLA, 896 exp top-16, MXFP4…
#   minimax-m2.7      GQA (48Q/8KV, head_dim 128, partial rope 64), 256 experts top-8…
#   minimax-m3        GQA (64Q/4KV, head_dim 128, partial rope 64), 128 experts top-4…
#   nemotron-3-super  Hybrid: 88 layers = 40 Mamba2 + 40 latent-MoE + 8 GQA…

# Where a model comes from and where it lands on this host:
scripts/model.py env nemotron-3-super     # CONTAINER / SOURCE / HF_REPO / convert flags

# Serve a specific registered model by NAME — resolves its container from the
# registry, waits until it's loaded + listening, and prints the client curl:
SERVE_DETACH=1 scripts/serve.sh minimax-m2.7 8081     # any registered name, any free port

# …or the raw form with an explicit container path (what serve.sh calls under the hood):
./target/release/coli serve /path/to/container 8080 "warm-up prompt"

# Convert an HF FP8/NVFP4 checkpoint into a colibrì container. Experts are NVFP4 by
# default (4-bit block-scaled); COLI_XFP8=1 for 8-bit e4m3 experts instead:
./target/release/coli convert nvidia/GLM-5.2-NVFP4 /path/to/container

# Re-quantize an existing e4m3 container's experts to NVFP4 (in place, ~18 min, ~2× faster
# decode + prefill at <1% perplexity — see the Expert quantization section below).
# The pass is idempotent: a weight that is already NVFP4 (it has a `.g` global scale beside
# it) is copied through, so re-running is safe and a container whose experts are ALREADY
# NVFP4 — like Nemotron's — can still have its RESIDENT tier converted with
# COLI_RESIDENT_NVFP4=1 (+9.4% decode on Nemotron, quality-neutral):
./target/release/coli requant-nvfp4 /path/to/e4m3-container /path/to/nvfp4-container
```

### Low-level: `gen` (forward-pass smoke test)

`coli gen <snap> [token_id...]` runs the raw forward pass and greedy-generates a
continuation. Its arguments are **token ids, not text** — e.g. `gen 100 200 300 400`
feeds the four-token prompt `[100, 200, 300, 400]` and prints the generated ids.
It's a benchmark/debug driver that bypasses the tokenizer (the server is the
text-in/text-out path); pass any valid ids (`< 154880`), or none to default to
`[1]`. `COLI_TIMING=1` and `COLI_PROFILE=1` print per-token latency and the
attention/MoE/load breakdown; `COLI_NGEN=N` sets how many tokens to generate
(default 16).

```bash
COLI_TIMING=1 COLI_PROFILE=1 docker/run-dgx.sh gen 100 200 300 400
```

### Low-level: `genbatch` (batched-decode benchmark)

`coli genbatch <snap> <B> <ngen> [token_id...]` advances **B sequences one token per
step through a single MoE call**, so the routed-expert union streams from disk once and
amortizes across the batch (decode is bytes-bound — this is the throughput lever). It
reports aggregate tok/s; `COLI_BATCH_VERIFY=1` also checks that a batched sequence is
token-identical to decoding it alone. See [the measured curve](#speculative-decoding-mtp--batched-decode)
— on a single node it's U-shaped (worse at moderate B, ~1.34× at B=64).

```bash
COLI_BATCH_VERIFY=1 ./target/release/coli genbatch /path/to/container 64 16 785 6722 315
```

## Switching models

One `coli` process serves one model — the model *is* the container you point it at. Beyond
the six registered short names, `-m` also takes an arbitrary HF repo, and `COLI_MODEL_DIR`
takes a container you already have on disk:

```bash
docker/run-dgx.sh -h <hf_token> -p 8080 -m unsloth/GLM-5.2-FP8   # any HF checkpoint
COLI_MODEL_DIR=/path/to/container docker/run-dgx.sh -p 8080      # a local snapshot
```

Without Docker, the registry ([`scripts/models.toml`](../scripts/models.toml)) maps a short
name to its container path, so `serve.sh` takes the name directly and several can run on
different ports:

```bash
scripts/model.py list                      # what's registered
scripts/serve.sh minimax-m2.7 8081         # resolves the container, waits until ready
scripts/serve.sh glm-5.2 8080
```

The startup banner echoes which model loaded (`(model: MiniMax-M2.7-container)`) and its
arch, so a wrong container is obvious immediately; `GET /v1/models` reports it at runtime.

## Repository layout

```
crates/          the Rust workspace (core, safetensors, tokenizer, kernels,
                 engine, backend, cluster, and the `coli` binary). The CUDA
                 kernels live in crates/colibri-backend/cuda/backend_cuda.{cu,h},
                 compiled by that crate's build script.
docker/          Dockerfile, entrypoint, and run-dgx.sh (the one-command launch)
scripts/         benchmark + codegen helpers
PORTING.md       per-module port history and milestone order
DEPLOYMENT.md    DGX Spark deployment guide
```

