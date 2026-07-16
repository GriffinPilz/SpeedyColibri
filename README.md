<p align="center">
  <img src="assets/colibri.svg" width="440" alt="SpeedyColibri — colibrì, in Rust, on the DGX Spark">
</p>

<h1 align="center">SpeedyColibri</h1>

**Run GLM-5.2 (744B-parameter MoE) on a single NVIDIA DGX Spark** — a Rust port
of the [colibrì](https://github.com/JustVugg/colibri) streaming engine, tuned for
the GB10 Grace-Blackwell superchip, with the whole hot path on the GPU.

> ### Built on colibrì — with gratitude
>
> SpeedyColibri exists because of **[JustVugg](https://github.com/JustVugg)** and
> the **[colibrì](https://github.com/JustVugg/colibri)** project. Every idea that
> makes this work — treating VRAM, RAM, and disk as one managed memory hierarchy;
> streaming a 744B model's routed experts on demand while keeping the dense part
> resident at int4; the faithful, quality-preserving GLM-5.2 forward pass — is
> theirs. This repository is a Rust rewrite of that engine: the C sources in
> [`c/`](c/) remain in-tree as the **reference oracle**, and the port is validated
> **token-exact** against them ([VALIDATION.md](VALIDATION.md)). None of this would
> be possible without JustVugg's original work. **Thank you.** If you want the
> mature, portable, multi-platform engine, start there — colibrì is the real thing;
> SpeedyColibri is one deployment target taken deep.

---

## What this is

colibrì's insight: a 744B Mixture-of-Experts model activates only ~40B parameters
per token, and only ~11 GB of those (the routed experts) change from token to
token. So the **dense part** (attention, shared expert, embeddings — ~10 GB at
int4) stays resident in RAM, and the **19,456 routed experts** (~358 GB) live on
disk and are **streamed on demand**.

SpeedyColibri takes that design and specializes it for **one box: the DGX Spark**
(GB10, aarch64 Grace CPU + Blackwell GPU, 128 GB coherent unified memory,
sm_121). On unified memory the GPU reads pageable host RAM directly, so routed
experts are computed on the GPU with **zero copies** — no VRAM double-store, no
`cudaMemcpy`, no eviction churn. Attention, the fused expert FFN, and the
projections all run on-device and are **token-exact vs the CPU path**.

The result runs the real model end-to-end today. It is **disk-streaming-bound**,
not compute-bound — which is exactly the problem multi-Spark is meant to solve
(see [Roadmap](#roadmap-multi-spark)).

## Quick start (DGX Spark)

The deliverable is an **OpenAI-compatible inference server**. One command builds
the container on first use, wires up the GPU (even on a stock shared Spark with no
`--gpus` runtime and no root), fetches the model into your Hugging Face cache if
needed, and starts serving:

```bash
# `serve [port] [warm-up prompt...]` — publishes the port and starts the HTTP API.
# The HF token (first arg, or HF_TOKEN) is only needed for the first, uncached download.
docker/run-dgx.sh hf_xxxxxxxxxxxxxxxxxxxx serve 8080 "hello, warm up the cache"

# already downloaded (or a public repo)? no token needed:
COLI_RAM_GB=85 docker/run-dgx.sh serve 8080
```

Then talk to it like any OpenAI endpoint — streaming works, which matters at
~1 tok/s:

```bash
# chat, streamed (SSE)
curl -N http://localhost:8080/v1/chat/completions -H 'Content-Type: application/json' -d '{
  "model": "glm-5.2", "stream": true, "max_tokens": 64,
  "messages": [{"role": "user", "content": "Explain MoE routing in one sentence."}]
}'

# raw completion, non-streamed
curl http://localhost:8080/v1/completions -H 'Content-Type: application/json' -d '{
  "prompt": "The capital of France is", "max_tokens": 16
}'

curl http://localhost:8080/v1/models      # what is served
curl http://localhost:8080/health         # liveness
```

**Warm-up prompts** (positional args after the port, and/or `COLI_WARMUP="a|b"`)
run through a short generation at startup so the hottest experts are already
resident before the first real request. The **port** is a positional arg after
`serve`, or `COLI_PORT`, default `8080`.

The model resolves in order: a snapshot mounted at `/model`
(`COLI_MODEL_DIR=<dir> docker/run-dgx.sh ...`) → the Hugging Face cache (the
launcher mounts the host's `~/.cache/huggingface`, so the 358 GB download happens
**at most once** and is shared with non-container runs) → `hf download` of
`$COLI_MODEL_REPO` (default `mateogrgic/GLM-5.2-colibri-int4-with-int8-mtp`).

Full deployment notes — GPU passthrough modes, every environment variable,
building by hand or with compose — are in **[DEPLOYMENT.md](DEPLOYMENT.md)**.

### Without Docker

The workspace has **no crates.io dependencies** (std + path crates only):

```bash
CUDA_HOME=/usr/local/cuda CUDA_ARCH=sm_121 \
  cargo build --release -p coli --features cuda

COLI_RAM_GB=85 ./target/release/coli serve /path/to/snapshot 8080 "warm-up prompt"
```

### Low-level: `gen` (forward-pass smoke test)

`coli gen <snap> [token_id...]` runs the raw forward pass and greedy-generates a
continuation. Its arguments are **token ids, not text** — e.g. `gen 100 200 300 400`
feeds the four-token prompt `[100, 200, 300, 400]` and prints the generated ids.
It's a benchmark/debug driver that bypasses the tokenizer (the server is the
text-in/text-out path); pass any valid ids (`< 154880`), or none to default to
`[1]`. `COLI_TIMING=1` and `COLI_PROFILE=1` print per-token latency and the
attention/MoE/load breakdown.

```bash
COLI_RAM_GB=85 COLI_TIMING=1 COLI_PROFILE=1 docker/run-dgx.sh gen 100 200 300 400
```

## Where it stands

Running the real 358 GB model on **one** DGX Spark (GB10), int4 experts, 85 GB
expert cache — honest, disk-streaming-bound numbers:

| | |
|---|---|
| Decode (steady state) | **~1.2 tok/s** mean, ~1.5 best |
| Prefill (8-token prompt) | ~18 s (one-time cold expert load) |
| Correctness | GPU tokens **byte-identical** to the CPU path |
| Resident footprint | ~10 GB dense + capped expert cache; no OOM |

The bottleneck is **loading**, not math: every token streams ~180 fresh experts
(~3.4 GB) from disk, and the model is far larger than RAM, so experts can't all
stay resident and load can't overlap the per-layer-sequential routing. On this
single node, decode ≈ load + compute.

What's landed to push on that wall:

- **Zero-copy GPU experts** on unified memory — the kernel reads the RAM copy in
  place (~2× the copy path, 0.5 GB VRAM, no eviction).
- **Flash-attention decode** and the fused expert FFN on-device (GB10-validated).
- **Chunked parallel reads** — each expert's 18 MB read is split across cores so a
  single cache miss saturates the NVMe (2.4× cold-load throughput).
- **Recycled read buffers** (`SharedBuf` pool) — kills per-expert allocation churn
  and a hidden 18 MB copy; warm expert loads got **21.7× faster**, decode 2.6×.

Per-module port status and the milestone order live in **[PORTING.md](PORTING.md)**.

## Roadmap: multi-Spark

A single Spark holds only ~5% of the experts resident, so it streams the rest from
disk every token — that's the whole ceiling. The design answer, and the reason
this port exists, is **expert-parallel across multiple Sparks**:

- **Shard the 256 experts/layer across nodes.** Each Spark owns a contiguous block,
  streams and computes only its shard, and keeps that shard *resident* — so the
  disk-streaming wall disappears. The dense part (attention, shared expert,
  embeddings) is replicated per node, so only expert I/O crosses the wire.
- **Transport: RDMA/RoCE over the ConnectX-7 200 GbE link**, ideally with GPUDirect
  so the Blackwell GPU DMAs activations straight to the fabric.
- The sharding math already lives in the [`colibri-cluster`](crates/colibri-cluster)
  crate (tested); the RDMA transport is designed and stubbed. Wiring it is the
  **next milestone**.

Two Sparks with the experts split and pinned turn a disk-bound ~1.2 tok/s into a
compute-bound engine. That's the target.

## Repository layout

```
crates/          the Rust workspace (core, safetensors, tokenizer, kernels,
                 engine, backend, cluster, and the `coli` binary)
c/               the original colibrì C engine — kept as the validation oracle;
                 c/backend_cuda.{cu,h} are compiled by the Rust CUDA backend
docker/          Dockerfile, entrypoint, and run-dgx.sh (the one-command launch)
scripts/         validate_c_vs_rust.py (the oracle harness) + codegen helpers
PORTING.md       per-module C→Rust status and milestone order
DEPLOYMENT.md    DGX Spark deployment guide
VALIDATION.md    how the port is checked token-exact against the C engine
```

## Credits & license

- **Original engine, design, and model runtime:**
  [colibrì](https://github.com/JustVugg/colibri) by
  [JustVugg](https://github.com/JustVugg). SpeedyColibri is a derivative work —
  the architecture, the streaming approach, and the GLM-5.2 forward pass are all
  from the upstream project. Please star and follow the original.
- **License:** Apache 2.0, inherited from colibrì (see [LICENSE](LICENSE)).
- GLM-5.2 is a model by Zhipu AI / Z.ai; this is an independent runtime for it.
