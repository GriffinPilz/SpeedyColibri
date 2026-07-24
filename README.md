<p align="center">
  <img src="assets/colibri.svg" width="440" alt="SpeedyColibri — colibrì, in Rust, on the DGX Spark">
</p>

<h1 align="center">SpeedyColibri</h1>

**Run huge Mixture-of-Experts models on a single NVIDIA DGX Spark** — GLM-5.2
(744B), MiniMax-M3, and MiniMax-M2.7 today — with a Rust engine tuned for the GB10
Grace-Blackwell superchip and the whole hot path on the GPU. Experts stream from
disk on demand; the dense part stays resident in low precision; add a second Spark
and the experts split across both.

> ### Rooted in colibrì — with gratitude
>
> SpeedyColibri began as a Rust port of **[JustVugg](https://github.com/JustVugg)**'s
> **[colibrì](https://github.com/JustVugg/colibri)**, and the foundation is theirs:
> the core insight of treating VRAM, RAM, and disk as one managed memory hierarchy —
> streaming a MoE model's routed experts on demand while keeping the dense part
> resident in low precision — and the original, quality-preserving GLM-5.2 forward
> pass. That idea is what everything here is built on. **Thank you.**
>
> It has since grown into its own engine. The C sources have been retired, correctness
> is carried by the port's own Rust test suite, and the work has gone well past a GLM
> port: on-GPU zero-copy experts on unified memory, flash- and tensor-core attention
> kernels, a NVFP4 4-bit expert format, adaptive RAM residency, multi-Spark
> expert-parallel over RoCE/RDMA, and support for model families colibrì doesn't
> cover — the MiniMax GQA models (M3, M2.7), each a different attention shape, norm,
> activation, and router. If you want the mature, portable, multi-platform original,
> start with colibrì; SpeedyColibri is the DGX-Spark line taken deep and broadened to
> more models.

---

## What this is

colibrì's insight: a 744B Mixture-of-Experts model activates only ~40B parameters
per token, and only ~11 GB of those (the routed experts) change from token to
token. So the **dense part** (attention, shared expert, embeddings — ~19 GB at
int8) stays resident in RAM, and the **19,456 routed experts** (NVFP4, ~436 GB) live
on disk and are **streamed on demand**.

SpeedyColibri takes that design and specializes it for **one box: the DGX Spark**
(GB10, aarch64 Grace CPU + Blackwell GPU, 128 GB coherent unified memory,
sm_121). On unified memory the GPU reads pageable host RAM directly, so routed
experts are computed on the GPU with **zero copies** — no VRAM double-store, no
`cudaMemcpy`, no eviction churn. Attention, the fused expert FFN, and the
projections all run on-device and are **token-exact vs the CPU path**.

The result runs the real model end-to-end today. On one Spark it is
**disk-streaming-bound**, not compute-bound — which is exactly what
[multi-Spark](#multi-spark-expert-parallel) solves: splitting the experts across two
Sparks measured **2.6× faster** decode, with bit-identical output.

The engine is **model-general** across MoE families — the streaming/residency machinery
is shared, and each model plugs in its own attention shape, norms, activation, and
router. GLM-5.2 is the original and largest target; the MiniMax GQA models were added
on top of the same core.

## Supported models

Registered in [`scripts/models.toml`](scripts/models.toml) — serve any by name
(`scripts/serve.sh <name>`) or list them with `scripts/model.py list`.

| name | params | attention | experts | routed format | notes |
|---|---|---|---|---|---|
| **`glm-5.2`** | 744B | MLA + DSA lightning indexer | 256, top-8 | NVFP4 | the original target; ≫ RAM, streams from disk |
| **`minimax-m3`** | — | GQA (64Q/4KV, head_dim 128, partial rope 64) | 128, top-4 | NVFP4 | gemma-norm, swigluoai, per-head QK-norm; ~229 GB |
| **`minimax-m2.7`** | — | GQA (48Q/8KV, head_dim 128, partial rope 64) | 256, top-8 | NVFP4 | per-layer QK-norm, plain SwiGLU, no shared expert, all-MoE; ~122 GB (**fits RAM**) |

**Which one loads** is chosen by the container you point `serve` at — one model per
process. With Docker, set `COLI_MODEL_REPO` (and `COLI_MODEL_DIR` for a local snapshot);
without Docker, pass a registry name to `scripts/serve.sh` (it resolves the container
path). See [Switching models](#switching-models). Adding a new model is an `Arch`
variant + a convert mapping + one registry block — the checklist is in
[scripts/README.md](scripts/README.md).

## Quick start (DGX Spark)

An **OpenAI-compatible inference server** in two steps. `docker/run-dgx.sh` handles
everything — the image build, GPU passthrough (even on a stock shared Spark with no
`--gpus` runtime and no root), the model download **and conversion**, and serving.

> **Prerequisites:** a DGX Spark (GB10) with Docker and the NVIDIA driver ≥ 580 (both
> ship with DGX OS), a Hugging Face token for the first download, and free disk for the
> model (~130 GB `m2.7` · ~230 GB `m3` · ~360 GB `glm`). No root required.

**1 — Get the code**

```bash
git clone https://github.com/GriffinPilz/SpeedyColibri.git
cd SpeedyColibri
```

**2 — Download and serve, one command**

```bash
docker/run-dgx.sh -h hf_xxxxxxxxxxxxxxxxxxxx -p 8080 -m m2.7
```

| flag | meaning | default |
|---|---|---|
| `-h <token>` | Hugging Face token — first download only (or the `HF_TOKEN` env var) | none |
| `-p <port>` | port to serve on | `8080` |
| `-m <model>` | `m2.7` · `m3` · `glm`  (or any `org/repo` checkpoint) | `glm` |

First run builds the image (~10 min) and downloads + converts the model (a one-time
30–90 min, cached in your HF cache); later runs skip both and reach a ready server in
~1–2 min — and once cached you can drop `-h`. Wait for:

```
[serve] OpenAI-compatible server on http://0.0.0.0:8080  (model: MiniMax-M2.7-container)
```

The advanced positional form — `docker/run-dgx.sh [hf_TOKEN] <coli-command> [args...]` —
and every `COLI_*` knob still work; see [Switching models](#switching-models), the
environment-variable table below, and the `docker/run-dgx.sh` header comments.

**3 — Query it**

Any OpenAI client works. Streaming (`"stream": true`) sends tokens as they are
produced — worth using at sub-1 tok/s so output appears live instead of after the
whole completion. Current measured throughput and long-context prefill costs are in
the [Performance & quality record](#performance--quality-record); a short prompt's
first token lands in a few seconds, but a long prompt pays a large prefill first.

```bash
# Liveness + what's served (instant)
curl http://localhost:8080/health
curl http://localhost:8080/v1/models

# Chat, streamed as Server-Sent Events (a 64-token reply ≈ 2+ min at ~0.5 tok/s)
curl -N http://localhost:8080/v1/chat/completions -H 'Content-Type: application/json' -d '{
  "stream": true, "max_tokens": 64,
  "messages": [{"role": "user", "content": "Explain MoE routing in one sentence."}]
}'

# Raw text completion, returned in one JSON object (waits for all tokens)
curl http://localhost:8080/v1/completions -H 'Content-Type: application/json' -d '{
  "prompt": "The capital of France is", "max_tokens": 16
}'
```

**Request fields** (JSON body):

| Field | Applies to | Meaning | Default |
|---|---|---|---|
| `messages` | `/v1/chat/completions` | array of `{"role": "user"\|"assistant"\|"system", "content": "..."}`; assembled with the **served model's** chat template (GLM-5.2's, or MiniMax's `<think>`-reasoning format for M3/M2.7), stopping at that model's own end-of-turn token | required |
| `prompt` | `/v1/completions` | raw text, tokenized and continued verbatim | required |
| `max_tokens` | both | tokens to generate; the only bound is that prompt + completion ≤ the served context (`COLI_CTX`) | `128` |
| `stream` | both | `true` → SSE token stream ending in `data: [DONE]`; `false` → one JSON object | `false` |
| `model` | both | accepted and echoed; ignored for routing (one model is served) | — |

**4 — Stop it**

`Ctrl-C` in the foreground, or `docker rm -f <container>` (find it with `docker ps`).

To serve a different model, stop it and re-run step 2 with a different `-m` (see
[Switching models](#switching-models)).

---

**How the model is found** — `run-dgx.sh` resolves it in order: a snapshot you
mount (`COLI_MODEL_DIR=<host-dir> docker/run-dgx.sh serve ...` → `/model`) → the
Hugging Face cache (the launcher mounts the host's `~/.cache/huggingface`, so the
358 GB download happens **at most once** and is shared with non-container runs) →
`hf download` of `$COLI_MODEL_REPO` (default
`nvidia/GLM-5.2-NVFP4`).

**Environment variables** (all optional; pass as `VAR=value docker/run-dgx.sh ...`):

| Var | Meaning | Default |
|---|---|---|
| `HF_TOKEN` | Hugging Face token for the first download (alt. to the `hf_...` arg) | none |
| `COLI_RAM_GB` | **manual override** of the adaptive default ([RAM residency](#ram-residency-adaptive-by-default) below). Forces a fixed expert-cache budget and disables the adaptive monitor. Rarely needed; on a ≫-RAM model, setting it *higher* drives the box into swap (measured on GLM: 85 GB → ~0.11 tok/s vs ~0.46 at the safe budget). | adaptive (see below) |
| `COLI_PORT` | listen port (a positional `port` arg overrides it) | `8080` |
| `COLI_WARMUP` | warm-up prompts, `\|`-separated | none |
| `COLI_CTX` | served context length (prompt + completion), e.g. `64k`. Clamped to what RAM can hold as KV and printed at startup; a request whose KV won't fit is rejected (507), never an OOM. Memory-bound on one node — see [Context & output length](#context--output-length): M3 ~400k · GLM ~290k · M2.7 ~190k | `32768` |
| `COLI_MODEL_DIR` | host path to a pre-downloaded snapshot → mounted at `/model` | none |
| `COLI_MODEL_REPO` | HF repo to download when nothing is mounted/cached | `nvidia/GLM-5.2-NVFP4` |
| `COLI_VRAM_GB` | cap the VRAM expert store | all free VRAM |
| `COLI_PIN_GB` | pin the hottest experts resident from the usage history so they never churn out of the cache. A number = that many GB; `auto` = size it to the knee of the usage curve (capped at 80% of the cache, leaving room for the cold tail to stream). Costs a one-time warm-up that reads every pinned expert — minutes, at `auto` scale | off |
| `COLI_PROFILE` | `1` → print the attention/MoE/expert-load time breakdown | off |
| `COLI_TIMING` | `1` → print per-token latency + steady-state tok/s | off |
| `COLI_EXPERT_LOG` | path → log every routing decision (`step layer pos e0..e7`) for `scripts/expert_hotset_analysis.py` | off |
| `COLI_PREFETCH` | speculative next-layer expert prefetch. **Leave off**: measured *slower* at every degree (0.82–0.99 vs 1.01 tok/s) — speculative loads evict the working set and contend for an already-saturated NVMe | off |
| `DRAFT` | MTP speculative decoding: draft this many tokens per step with the model's own next-token (MTP) head, then verify them in one main-model forward. **Measured break-even at best on single-sequence NVFP4** (decode is bytes-bound, not compute-bound — drafting *adds* expert reads), and **not bit-exact while drafting** (`DRAFT=0` is exact; drafting's multi-token verify runs a different attention path than S=1 decode, so ~1 token in 16 can differ). Only pays in batched serving. Auto-disables below 10% acceptance. See [Speculative decoding + batched decode](#speculative-decoding-mtp--batched-decode). | off (`0`) |
| `MTP` | `0` force-disables the MTP head even if the container ships one (equivalent to `DRAFT=0`) | on when present |

Multi-node variables (`COLI_NUM_NODES`, `COLI_PEERS`, …) are in
[Multi-Spark](#multi-spark-expert-parallel) below.

Full deployment notes — GPU passthrough modes, building by hand or with compose,
the CUDA base image — are in **[DEPLOYMENT.md](DEPLOYMENT.md)**.

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
scripts/model.py list
#   glm-5.2       MLA + DSA lightning indexer, 256 experts top-8
#   minimax-m3    GQA (64Q/4KV), 128 experts top-4
#   minimax-m2.7  GQA (48Q/8KV), 256 experts top-8

# Serve a specific registered model by NAME — resolves its container from the
# registry, waits until it's loaded + listening, and prints the client curl:
SERVE_DETACH=1 scripts/serve.sh minimax-m2.7 8081     # any registered name, any free port

# …or the raw form with an explicit container path (what serve.sh calls under the hood):
./target/release/coli serve /path/to/container 8080 "warm-up prompt"

# Convert an HF FP8/NVFP4 checkpoint into a colibrì container. Experts are NVFP4 by
# default (4-bit block-scaled); COLI_XFP8=1 for 8-bit e4m3 experts instead:
./target/release/coli convert nvidia/GLM-5.2-NVFP4 /path/to/container

# Re-quantize an existing e4m3 container's experts to NVFP4 (in place, ~18 min, ~2× faster
# decode + prefill at <1% perplexity — see the Expert quantization section below):
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

One `coli` process serves one model — the model is the container you point it at.

**Docker.** Pick the model with `-m`; everything else is the same command:

```bash
docker/run-dgx.sh -h <hf_token> -p 8080 -m glm     # GLM-5.2 (default)
docker/run-dgx.sh -h <hf_token> -p 8080 -m m3      # MiniMax-M3
docker/run-dgx.sh -h <hf_token> -p 8080 -m m2.7    # MiniMax-M2.7

# any other checkpoint by its HF repo:
docker/run-dgx.sh -h <hf_token> -p 8080 -m unsloth/GLM-5.2-FP8

# a snapshot you already downloaded/converted:
COLI_MODEL_DIR=/path/to/container docker/run-dgx.sh -p 8080
```

**Without Docker.** The registry ([`scripts/models.toml`](scripts/models.toml)) maps a
short name to its container path, so `serve.sh` takes the name directly:

```bash
scripts/model.py list                      # what's registered
scripts/serve.sh minimax-m2.7 8081         # resolves the container, waits until ready
scripts/serve.sh glm-5.2 8080
```

The startup banner echoes which model loaded (`(model: MiniMax-M2.7-container)`) and its
arch, so it's obvious if the wrong one is up. `GET /v1/models` reports it at runtime.

## RAM residency (adaptive by default)

The expert cache **fills RAM and defends it** — no flag, for every model. A background
monitor polls `MemAvailable` every 100 ms and evicts LRU experts the moment free memory
approaches a hard floor (~3 GB), *whatever* consumed it — more experts, a longer KV cache,
the GPU's own working set on GB10's unified pool. A cache that gives memory back under
pressure **cannot OOM**, which is what lets a fill-RAM policy point at a model of any size:

- **Near-fit** (experts ≈ RAM — e.g. MiniMax-M2.7, ~122 GB on a 121 GB Spark): fill RAM and
  hold the whole working set resident, dropping the page-cache double-copy (`fadvise`) so
  `MemAvailable` is honest. Measured **1.94×** over the old static default (median 4.83 vs
  2.49 tok/s, diverse-prompt serving) — the working set stays resident instead of streaming.
- **≫ RAM** (experts larger than RAM — e.g. MiniMax-M3 ~216 GB, GLM-5.2 ~436 GB): fill RAM
  with as many experts as fit, keep the OS page cache as a reclaimable second tier, and let
  the monitor evict the LRU tail under pressure. Measured **1.22×** on M3 (2.05 vs 1.68 tok/s)
  — more resident experts, higher hit rate — and, crucially, **no crash**: the same box that
  OOM-died under a fixed `COLI_RAM_GB=100` now fills to 121 GB used and holds `avail` at the
  floor for the whole run.

`COLI_RAM_GB=<n>` sets a fixed fill *target* (e.g. a smaller per-node budget for
[multi-Spark](#multi-spark-expert-parallel)); the pressure monitor still runs underneath it,
so even an oversized value can no longer walk the box into swap.

### Context & output length

Each request reserves *its own* KV cache dynamically — the expert cache evicts to make room
before the allocation, and a request whose KV genuinely can't fit is **rejected with HTTP 507**
rather than OOMing the box. So `COLI_CTX` (prompt + completion) can be raised safely, but the
real ceiling is **memory, not the model's architectural max**: on one 121 GB Spark the KV cache
must fit alongside the resident weights, so the server clamps `COLI_CTX` to what RAM can hold
and prints the limit at startup.

The KV per token depends on the attention shape. GLM's **MLA** stores a compressed latent
that is mirrored on the GPU (host + device); the MiniMax **GQA** models store full K and V
per kv-head **on the host only** (the GPU reads them over unified memory), plus a small roped
key that is mirrored. So the cost scales with kv-heads × layers, and M3's 4 kv-heads actually
make it *lighter* per token than GLM's latent:

| model | attention | KV / token (host + GB10 device shadow) | max `COLI_CTX` on a 121 GB Spark |
|---|---|---|---|
| **MiniMax-M3** | GQA, 4 kv-heads × 60 layers | ~260 KB | **~400k** (measured clamp 402,690) |
| **GLM-5.2** | MLA (compressed), mirrored | ~350 KB | ~290k |
| **MiniMax-M2.7** | GQA, 8 kv-heads × 62 layers | ~530 KB | ~190k |

(architectural maxima are 1M / 1M / 196,608 respectively; only M2.7 is near-reachable.) These are ceilings where the experts are nearly all evicted, so throughput there is low;
practical high-throughput context is lower. The default `COLI_CTX` stays **32,768** — a small KV
keeps the most RAM for resident experts and the lowest latency. `max_tokens` defaults to 128 and
is bounded only by the remaining context (no fixed cap).

**The KV grows on demand.** The cache is sized to the window in *address space* but committed
lazily (zero-on-demand pages) — its resident RAM tracks the tokens actually produced, not
`max_tokens`. So a request that sets a large `max_tokens` but stops early never pays for the
tail: only the prompt's KV is reserved up front, and the generation grows a token at a time
while the expert cache evicts against it. A prompt that can't fit is still rejected (507)
rather than OOM'd.

## Where it stands

Running the real 358 GB model on **one** DGX Spark (GB10). The bottleneck is
**loading**, not math: every token streams ~180 fresh experts (~3.4 GB) from disk,
and the model is far larger than RAM, so experts can't all stay resident and load
can't overlap the per-layer-sequential routing. On this single node, decode ≈ load +
compute. For the current, measured throughput and quality numbers see the
[Performance & quality record](#performance--quality-record) below — they supersede
every tok/s figure that used to live here (those came from a repeated-single-prompt
benchmark that read ~1.5–2× high, on the pre-8/4 model).

What's landed to push on that wall:

- **Zero-copy GPU experts** on unified memory — the kernel reads the RAM copy in
  place (~2× the copy path, 0.5 GB VRAM, no eviction).
- **Flash-attention decode** and the fused expert FFN on-device (GB10-validated).
- **Chunked parallel reads** — each expert's 18 MB read is split across cores so a
  single cache miss saturates the NVMe (2.4× cold-load throughput).
- **Recycled read buffers** (`SharedBuf` pool) — kills per-expert allocation churn
  and a hidden 18 MB copy; warm expert loads got **21.7× faster**, decode 2.6×.

Per-module port status and the milestone order live in **[PORTING.md](PORTING.md)**.

## Performance & quality record

A living, measured record of throughput and quality per node size — **starting →
current** — so progress (and regressions) stay visible. Update `current` as it
changes; leave `starting` fixed so the trajectory reads at a glance.

**Read the conditions, not just the digits** — they move the number more than any
optimization does:
- A *repeated-single-prompt* benchmark reads ~1.5–2× higher than *diverse* prompts,
  because the expert cache hits on the repeat. All numbers here use 12 diverse prompts.
- A RAM budget past the swap cliff collapses throughput ~4× (measured: 87 GB → 0.11,
  40 GB → 0.46). All current numbers use the auto budget (`MemTotal/3` ≈ 41 GB/node).
- Output is **bit-identical** across node counts, so **quality is node-independent** —
  it's tracked once, not per size.

Config: GLM-5.2 744B MoE, **int8 resident + NVFP4 experts** (int4 support has been
removed from the engine entirely), GB10 Grace-Blackwell, greedy decode. The 2026-07-17 rows below were on the
earlier int4-experts build and establish the *resident* bit-width choice; the NVFP4-vs-e4m3
*expert*-format A/B is in [Expert quantization](#expert-quantization-nvfp4-default-e4m3-opt-out).

### Quality (model-level, all node sizes)

| | perplexity ↓ | top-1 ↑ | when |
|---|---|---|---|
| starting — int4 resident (reference 4/4) | 48.665 | 32.1% | baseline |
| **int8 resident (shipped)** | **6.189** | **57.9%** | 2026-07-17 |

int4 attention was wrecking the model; int8 resident recovers it for +~7 GB RAM. Perplexity
from `coli ppl`; lower is better. These rows fix the resident format; the experts were int4
here and are NVFP4 now — the same-text NVFP4-vs-e4m3 expert A/B (4.707 vs 4.670, +0.8%) is
in the Expert quantization section (a different held-out text, so not comparable to 6.189).

### Throughput — decode, diverse prompts, short context

| nodes | starting tok/s | current tok/s | how measured |
|---|---|---|---|
| 1 | 0.46 | **0.46** | counterbalanced, n ≥ 6, auto budget |
| 2 | — | *not yet measured on 8/4* | prior repeated-prompt runs read ~1.95, but on the old model and inflated — not comparable; re-measure with RDMA-A |

The single-node number is flat from a 20 GB to a 55 GB cache (diverse traffic barely
reuses experts), so cache size is not a throughput lever here — headroom and avoiding
swap are.

### RAM residency by model (fill + OOM-safe eviction) — 2026-07-23

Every model fills RAM and evicts LRU experts under pressure. Measured on a 121 GiB Spark,
`bench_serve.py` diverse prompts, single node:

| model | routed experts vs RAM | policy (auto) | serving throughput |
|---|---|---|---|
| **MiniMax-M2.7** | ~122 GB ≈ 121 GB (near-fit) | fill ~101 GB + fadvise, hold working set | **4.83 tok/s** median — **1.94×** over the old 41 GB static default (2.49) |
| **MiniMax-M3** | ~216 GB (1.8× RAM) | fill RAM, keep page cache, LRU-evict | **2.05 tok/s** median — **1.22×** over the old static 1.68; box fills to 121 GB used and holds `avail` at the 3 GB floor for the whole run, **no OOM** |
| **GLM-5.2** | ~436 GB (3.6× RAM) | fill RAM, keep page cache, LRU-evict | not re-run (container offloaded to HF); ≫-RAM, expect a small gain like M3 at most — the page cache already served the hot set |

Takeaway: filling RAM helps whenever more experts fit — a lot when the model fits (~2× on
M2.7), modestly when it doesn't (~1.2× on M3). The **eviction is what makes it safe**: the
earlier build OOM-crashed when a fixed 100 GB budget grew into the GPU's working set; the
monitor now defends a hard floor, so filling RAM never crosses the swap line and the same
config just caps itself. That safety is also what lets `COLI_CTX` reach each model's full
context maximum — the KV cache grows, experts evict, the box stays up.

### Long context — single node, 8/4, varied input (in progress)

| input tokens | prefill (time to first token) | decode at that context |
|---|---|---|
| 512 | 202 s | 0.58 tok/s |
| 2048 | 618 s | 0.45 tok/s |
| 32k (target-adjacent) | ~2.5 h *(extrapolated, unmeasured)* | lower |
| 64k (bare-minimum target) | ~5 h *(extrapolated, unmeasured)* | lower |

Prefill is ~linear (~0.27 s/token + ~63 s fixed) and dominates at long context, which
is why 64k single-node is impractical on time (memory fits fine, no swap). This is the
case for the multi-node work below: sharding experts cuts per-node prefill streaming.
The 32k/64k rows are **extrapolations from the two measured points**, not measurements
— they will be replaced with real numbers or struck out.

### Speculative decoding (MTP) & batched decode

Both are throughput levers aimed at the bytes-bound decode. Measured 2026-07-22, single
node, NVFP4, warm.

**MTP speculative decoding (`DRAFT=n`)** — the model ships a next-token (MTP) head; the
converter keeps it by default (`has_mtp=true`), and `DRAFT=n` drafts *n* tokens per step
and verifies them in one forward. On **single-sequence** decode it is **break-even at
best** and a loss beyond DRAFT=2:

| `DRAFT` | draft acceptance | effective tok/s | vs baseline |
|---|---|---|---|
| 0 (baseline) | — | 0.81 | — (bit-exact) |
| 2 | 57% | 0.81 | break-even |
| 4 | 30% | 0.67 | −17% |
| 8 | 8% | auto-disabled | — |

Why it doesn't pay: decode is **bytes-bound** (each token streams the routed experts from
disk), and a verify pass over *k* drafts routes each token to its own top-8, *growing* the
per-layer expert union — so drafting reads *more* bytes to make the same tokens. Acceptance
improves with quantization quality (an int4 head auto-disabled at <10%; e4m3 45%; NVFP4
57%) but never enough to win single-sequence. Drafting is also **not bit-exact** on NVFP4
(`DRAFT=0` is exact; the multi-token verify runs a different attention path than S=1 decode
and flips ~1 token in 16). **Keep `DRAFT=0` unless batching.**

**Batched decode (`coli genbatch`)** — B sequences advance one token/step through one MoE
call, so the expert union loads once and amortizes across the batch. Aggregate tok/s is
**U-shaped** on a single node — batching loses in the middle (union grown, 40 GB cache
thrashed) and wins once the union saturates:

| B | aggregate tok/s | ms/token | vs B=1 |
|---|---|---|---|
| 1 | 0.82 | 1213 | 1.0× |
| 8 | 0.50 | 2000 | 0.61× |
| 16 | 0.59 | 1681 | 0.72× |
| 32 | 0.77 | 1295 | 0.94× |
| **64** | **1.10** | **908** | **1.34×** |

This is with near-worst-case routing diversity (each sequence offset to route almost
disjointly) — realistic traffic overlaps more, so it crosses earlier and peaks higher. The
ceiling is set by disk bandwidth: even at saturation the union (~all 256 experts) never
fits the cache, so every step still streams ~the whole expert set. The real lever is
**RAM-resident experts across a cluster**, which lifts the whole curve; a continuous-batching
scheduler pairs with that, not with a single node.

### Expert quantization: NVFP4 (default), e4m3 opt-out

The routed experts (97% of the weights, and what every token streams) are stored as
**NVFP4** — 4-bit block-scaled. Resident weights (attention / dense / shared) stay 8-bit
int. Two source checkpoints feed the experts: modelopt **NVFP4**
[`nvidia/GLM-5.2-NVFP4`](https://huggingface.co/nvidia/GLM-5.2-NVFP4) (the default) and
block-scaled **FP8** [`unsloth/GLM-5.2-FP8`](https://huggingface.co/unsloth/GLM-5.2-FP8).

| expert format | bytes/wt | experts on disk | build (from source checkpoint) |
|---|---|---|---|
| **NVFP4** (e2m1 + per-16 ue4m3 block scale + global) — **default** | **0.5625** | **~436 GB** | `coli convert nvidia/GLM-5.2-NVFP4 <out>` |
| e4m3 fp8 (per-row) — 8-bit opt-out | 1.0 | ~735 GB | `COLI_XFP8=1 coli convert unsloth/GLM-5.2-FP8 <out>` |

**NVFP4 is a 4-bit block-scaled format** — 4-bit weights with a shared scale per 16
inputs, so it is int4-small while nearly matching e4m3's accuracy. It is the default output
of `coli convert` for **any** source (a modelopt NVFP4 source stays NVFP4 with no
dequant/requant loss; an FP8 source is quantized straight to NVFP4). **int4 has been
removed from the engine entirely** (NVFP4 is the 4-bit format now). The one command:

```bash
docker/run-dgx.sh <hf_token> serve 8080 "warm up"   # defaults to nvidia/GLM-5.2-NVFP4 → NVFP4
```

Switching to the 8-bit e4m3 experts is `--model unsloth/GLM-5.2-FP8 COLI_XFP8=1`. To turn
an existing e4m3 container into NVFP4 without a re-download, `coli requant-nvfp4 <e4m3-dir>
<out-dir>` (~18 min for the 744B model).

**Measured NVFP4 vs e4m3, single node GB10, GPU, warm cache** (2026-07-21; a same-session
A/B — the *ratio* is the robust result, the absolute tok/s uses a short warm prompt and
is not comparable to the diverse-prompt record above):

| | e4m3 (8-bit) | NVFP4 (4-bit) | NVFP4 win |
|---|---|---|---|
| decode | 2571 ms/tok (0.39 tok/s) | **1186 ms/tok (0.84 tok/s)** | **2.17×** |
| decode + `COLI_PIN_GB=30` | — | 1049 ms/tok (0.95 tok/s) | 2.45× |
| prefill @1024 (+prefetch+tc) | 5.6 tok/s | **11.1 tok/s** | **1.98×** |
| perplexity (128 tok) ↓ | 4.670 | 4.707 | +0.8% |
| top-1 ↑ | 58.3% | 59.8% | +1.5 pt |

**~2× faster on both prefill and decode at under 1% perplexity cost** — NVFP4 wins on
both the halved streamed bytes *and* a dedicated single-row `nvfp4_gemv` decode kernel
(1.59× faster than the tiled path at batch 1). A device-copy staging variant of the
prefill kernel was tested and did not help (left off by default). NVFP4 experts are
stored as one coalesced blob (nibbles ++ block-scales) so the loader's gate/up/down read
grabs the scales for free — a separate scale sidecar cost an uncoalesced read per expert.

## Multi-Spark (expert-parallel)

**Working, and it's the single biggest win available.** The 256 experts/layer are
split across nodes: each Spark owns half, loads and computes only its own half, and
answers its peers over the ConnectX/RoCE fabric. The dense part (attention, shared
expert, embeddings) is replicated per node, so only expert activations cross the wire
(~24 KB each way, not expert weights).

> ⚠️ **These 2-Spark numbers are superseded and not comparable to the record above.**
> They were taken with a *repeated single prompt* (which reads ~1.5–2× high because the
> cache hits on the repeat) on the pre-8/4 model. They are kept only because they are
> the sole 2-node data that exists; the shape (2-node ≈ 2× 1-node, from residency not
> compute) is believed to hold, but the magnitudes must be re-measured with diverse
> prompts on 8/4 (RDMA-A). Treat as illustrative, not current.

Measured on two DGX Sparks, 32-token greedy decode, 6 consecutive repeats against a
warm server, **40 GB expert cache per node**:

| | cold (1st run) | warm (converged) |
|---|---|---|
| 1 Spark | 0.71 tok/s | **0.76 tok/s** |
| 2 Sparks | 1.15 tok/s | **~1.95 tok/s** (**2.6×**) |

The cold/warm gap is the point: the first request pays for filling the cache, and a
`serve` warm-up prompt buys that back before real traffic arrives.

Output is **bit-identical** to single-node (all 32 tokens), verified on the real 744 B
model. The win is *residency, not compute*: at the same per-node budget each Spark
caches a 128-expert shard instead of all 256, so it hits disk far less. Fabric latency
is a rounding error by comparison — RoCE RTT is ~0.36 ms, so all 75 layers of
round-trips cost ~27 ms of a ~510 ms token (~5%).

### Running it

Start the **workers first** — the driver verifies every peer at startup and exits if
one is unreachable.

```bash
# --- on each worker node (rank 1..N-1) ---
COLI_NUM_NODES=2 COLI_NODE_RANK=1 COLI_RAM_GB=40 \
  docker/run-dgx.sh worker                    # serves its shard on :48800

# --- on the driver (rank 0) — this is the node you send requests to ---
COLI_NUM_NODES=2 COLI_NODE_RANK=0 \
  COLI_PEERS=1=192.168.100.10:48800 \
  COLI_RAM_GB=40 docker/run-dgx.sh serve 8080
```

`docker/run-dgx.sh cluster` scans the fabric and prints the Sparks it can see, with
their RoCE addresses — use it to fill in `COLI_PEERS`.

Both nodes print a **sharding fingerprint** at startup; they must match. They also
print their build revision (`coli v0.1.0 (abc1234)`) — that must match too, or one
node is running stale code. Nodes that disagree about the expert map are refused at
connect time rather than silently producing wrong tokens, so a mismatch is a startup
failure, never a wrong answer.

| Var | Meaning | Default |
|---|---|---|
| `COLI_NUM_NODES` | cluster size; `1` disables expert-parallel entirely | `1` |
| `COLI_NODE_RANK` | this node's rank, `0..NUM_NODES-1`. Rank 0 is the driver (runs `serve`); the rest run `worker` | `0` |
| `COLI_PEERS` | `rank=host:port` for **every** other rank, comma-separated. Required on the driver; a missing rank is a startup error | none |
| `COLI_EXPERT_PORT` | port a `worker` listens on | `48800` |
| `COLI_SHARD` | `hot` → assign experts to balance *traffic* rather than count, from the usage history. **Measured no gain on 2 nodes** (~1.96 vs ~1.95 tok/s) and it requires every node to share a byte-identical `.coli_usage` or the handshake refuses. Leave unset. | contiguous |
| `COLI_USAGE` | path to the usage history. Point every node at the *same* file (shared storage) if using `COLI_SHARD=hot` | `<snap>/.coli_usage` |

**Scaling past 2.** Per-node cache (~5 900 experts at 106 GB) versus 19 200 total
routed experts means at **4+ Sparks each node's whole shard is resident** and expert
streaming stops entirely. Sharding does *not* reduce disk footprint — every node
holds the full 358 GB snapshot and simply reads less of it.

**Next:** the RDMA transport (`colibri-cluster`, stubbed behind the same `Transport`
trait) would cut the ~27 ms/token of fabric latency — worth ~5%. The larger remaining
lever is still expert residency, i.e. more nodes.

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

## Credits & license

- **Foundation:** [colibrì](https://github.com/JustVugg/colibri) by
  [JustVugg](https://github.com/JustVugg). SpeedyColibri is a derivative work — the
  managed memory hierarchy, the on-demand expert-streaming approach, and the original
  GLM-5.2 forward pass come from the upstream project. The Rust engine, DGX-Spark
  specialization, GPU kernels, multi-Spark expert-parallel, adaptive RAM residency,
  NVFP4 experts, and the MiniMax GQA model support were built here on that foundation.
  Please star and follow the original.
- **License:** Apache 2.0, inherited from colibrì (see [LICENSE](LICENSE)).
- Models are by their respective authors — GLM-5.2 by Zhipu AI / Z.ai, MiniMax-M3 /
  M2.7 by MiniMax; this is an independent runtime for them.
