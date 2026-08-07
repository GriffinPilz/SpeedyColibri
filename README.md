# SpeedyColibri

A Rust MoE inference engine for a single **DGX Spark** (GB10, 121.7 GiB unified memory).

Routed experts stream from NVMe on demand, the dense tier stays resident in low precision,
and RAM is filled adaptively with LRU eviction that cannot OOM. That is what lets one box
run a 744B model — or a 1.5T one — without sharding it across a rack.

Six models run today, from a 69 GB hybrid to a 1.4 TB container.

---

## 1 — How fast is it?

One Spark, single sequence, greedy, 512-token prompt. Every figure is a median of repeated
runs on **one build**, gated on token identity so a "faster" number that changed the output
fails loudly instead of being reported as a win.

| model | on disk | prefill | decode | serving |
|---|---|---|---|---|
| **`nemotron-3-super`** | 69 GB | **41.0 tok/s** (12.5 s) | **12.8 tok/s** | **10.0 tok/s** |
| **`minimax-m2.7`** | 122 GB | **18.3 tok/s** (28.0 s) | **5.1 tok/s** | **5.9 tok/s** |
| **`deepseek-v4-flash`** | 145 GB | **13.6 tok/s** (37.6 s) | **4.7 tok/s** | **4.3 tok/s** |
| **`minimax-m3`** | 229 GB | **16.1 tok/s** (31.8 s) | **2.6 tok/s** | **2.6 tok/s** |
| **`glm-5.2`** (744B) | 403 GB | **6.6 tok/s** (78.1 s) | **0.9 tok/s** | **0.8 tok/s** |
| **`kimi-k3`** (1.5T) | 1.4 TB | **1.8 tok/s** (4.6 min) | **0.35 tok/s** | — |

Measured 2026-08-06 on branch `docs-retest-4model-2026-08-06`, which is ahead of `main`.

**The five rows above predate the coverage-gated FFN staging** (`ffn-devcopy-coverage-gate`).
That change measures **nemotron −10.0%, M2.7 −6.5%, M3 −2.4% on prefill** against exactly
this build, so those three prefill figures are conservative until the suite is re-run. Decode,
serving and GLM are unaffected. K3 is MXFP4 and the MXFP4 expert paths never staged, so its
row is current either way.

K3's row *was* the stale one — carried over at "~1.1 tok/s (~8 min) / ~0.4" from a build
nobody had re-measured because the suite takes hours. Re-run 2026-08-06: **1.8 tok/s**
(median of 3, 1.6% spread, token gate PASS) and **0.35 tok/s** decode (two reps, both
0.35 median / 0.45 best). Prefill is far better than the old figure; decode is unchanged
once a reporting bug is undone — `median()` printed `%.1f` on even-count inputs, rendering
0.35 as "0.3" and making a flat result look like a 12% regression. Fixed in `scripts/lib.sh`.
K3 remains the SSD-bound model: `expert-load` is 126.8 s of its 278.6 s prefill.

**Prefill is what this build moved, and only for M3 and GLM — both ~1.22×.** Measured against
a freshly built `main`, ABBA-interleaved in one session, tokens gated identical:

| | `main` | this build | ratio |
|---|---|---|---|
| **`minimax-m3`** (n=6/arm) | 38458 ms | 31559 ms | **1.22×** — arm ranges 2.0% and 2.4%, disjoint |
| **`glm-5.2`** (n=4/arm) | 90497 ms | 73980 ms | **1.22×** — direction certain, magnitude soft |

M3's is as clean as this box gets: 6.3 s separates the slowest new run from the fastest old
one. GLM's *direction* is equally certain — the ranges don't overlap either — but its new arm
spans 17% and drifts slower run over run while the `main` arm holds flat, which no
explanation here accounts for, so read GLM as "roughly 1.2×" rather than 1.223×.

The other three are unchanged, for reasons the profile explains: nemotron's experts are fully
resident (`expert-load` = 2 ms, so there was no CPU-side cost to recover), M2.7 spends 54% of
prefill waiting on NVMe, and V4 takes neither of the paths the fix touched (measured 1.35%
apart at n=6/arm, against a 9.6% spread). Decode and serving are unchanged everywhere.

**These ratios deliberately do not come from comparing this table against an older one.** The
same unmodified binary measured V4 prefill at 41 s in July and 36 s today: **cross-day drift
on this box is worth ~12% before any code changes**, which is larger than most of the wins
here. The fix itself was pooling per-expert scratch buffers whose page faults were being
billed to gather and scatter — that story, and the eight wrong hypotheses that preceded it,
is in [docs/PERFORMANCE.md](docs/PERFORMANCE.md).

**Read the decode column as a disk-streaming ladder, not a quality ranking.** Decode is bound
by how many expert bytes each token pulls, so it tracks *model size against 121 GB of RAM*
almost perfectly. Prefill does not — M3 beats the smaller V4 there, because V4's attention
costs 14.7 s to M3's 6.8 s (Hyper-Connections, the Compressor and the Indexer are not cheap),
which more than pays back the expert bytes it saves.

- **prefill** — one-shot `coli gen`, which refills the in-process expert cache every
  invocation. A server pays that once.
- **decode** — steady-state tok/s. **This column understates the largest models**: the
  window is 24 tokens, which at ~1 tok/s is mostly warm-up, so it charges GLM a fixed cost
  over very few tokens. Run out to 100 and GLM's real steady state is **~1.23–1.33**, against
  the 0.9 reported here. The window is kept at 24 anyway, because changing it would break
  comparability with every figure previously recorded.
- **serving** — real HTTP, twelve **diverse** prompts. Deliberately not one prompt repeated,
  which warms the cache on a tiny working set and flatters throughput ~2×. The median
  includes each model's warm-up, and how much that costs depends on whether the model
  converges on a resident working set.

Reproduce any cell:

```bash
BENCH_REPS=5 scripts/bench.sh <model> prefill
BENCH_REPS=8 scripts/bench.sh <model> decode    # 4 for glm-5.2
scripts/bench.sh <model> serve
```

Full history, including every negative result and the measurement traps that produced them,
is in [docs/PERFORMANCE.md](docs/PERFORMANCE.md) and
[docs/MODEL-TEST-MATRIX.md](docs/MODEL-TEST-MATRIX.md).

---

## 2 — Models

| name | params | attention | experts | routed format | on disk |
|---|---|---|---|---|---|
| **`nemotron-3-super`** | 120B-A12B | **hybrid**: 88 layers = 40 Mamba2 + 40 latent-MoE + 8 GQA (NoPE, no QK-norm) | 512, top-22 | NVFP4 | 69 GB |
| **`minimax-m2.7`** | — | GQA (48Q/8KV, partial rope 64), per-layer QK-norm | 256, top-8 | NVFP4 | 122 GB |
| **`deepseek-v4-flash`** | — | latent + **O-LoRA** output proj, 128-tok sliding window, Compressor 41/43, Indexer 21/43 | 256, top-6 + 1 shared | MXFP4 | 145 GB |
| **`minimax-m3`** | — | GQA (64Q/4KV, partial rope 64), per-head QK-norm | 128, top-4 | NVFP4 | 229 GB |
| **`glm-5.2`** | 744B | MLA + DSA lightning indexer | 256, top-8 | NVFP4 | 403 GB |
| **`kimi-k3`** | 1.5T | **hybrid**: 93 layers = 69 KDA (delta-rule linear attn) + 24 gated MLA | 896, top-16 (latent) | MXFP4 | 1.4 TB |

Two are not transformers. **Nemotron-3-Super** is a state-space/attention hybrid with
gateless ReLU² experts in a 1024-wide latent space. **Kimi-K3** has no ordinary residual —
attention residuals thread `prefix_sum`/`block_residual` through the stack. **DeepSeek-V4**
replaces the residual entirely with Hyper-Connections: four copies of the hidden state,
`[b,s,4,4096]`.

Only nemotron fits RAM outright; the rest stream. Coverage — the share of a model's experts
that can stay resident — is the axis that predicts nearly everything about its behaviour.

---

## 3 — How to run each model

Every model is published as a ready-to-run container, so a fresh host downloads one instead
of paying a multi-hour conversion.

```bash
docker/run-dgx.sh -m nemotron -p 8080     # or: m2.7 · m3 · glm · k3 · v4
```

Add `-h <hf_token>` (or set `HF_TOKEN`) on the **first** run only — it is needed to pull the
container, not to serve it.

Without Docker, by registry name:

```bash
scripts/serve.sh nemotron-3-super         # resolves the container, waits for listen
scripts/model.py list                     # what's registered
```

Then any OpenAI client:

```bash
curl -s localhost:8080/v1/chat/completions -H 'Content-Type: application/json' -d '{
  "messages": [{"role": "user", "content": "The capital of France is"}], "max_tokens": 16
}'
```

**One model per process** — which one loads is decided by the container you point at.
`kimi-k3` wants `COLI_O_DIRECT=1` (1.09× prefill / 1.13× decode expert-load, tokens
identical; off by default because it *loses* on GLM and the mechanism is unexplained).

Build, the low-level `gen`/`genbatch` tools, and the repo layout are in
[docs/DEVELOPMENT.md](docs/DEVELOPMENT.md). Every `COLI_*` knob is in
[docs/CONFIGURATION.md](docs/CONFIGURATION.md). Memory behaviour and context limits are in
[docs/MEMORY-AND-CONTEXT.md](docs/MEMORY-AND-CONTEXT.md). Multi-box expert parallelism is in
[docs/MULTI-SPARK.md](docs/MULTI-SPARK.md).

---

## 4 — Where each model comes from

`hf_repo` in [`scripts/models.toml`](scripts/models.toml) is the **container** — what to
re-materialize from, no conversion needed. The upstream is the original checkpoint, if you
would rather convert it yourself.

| model | container (ready to run) | upstream checkpoint |
|---|---|---|
| `nemotron-3-super` | [`Kanposer/Nemotron-3-Super-120B-speedy-colibri-nvfp4`](https://huggingface.co/Kanposer/Nemotron-3-Super-120B-speedy-colibri-nvfp4) | `nvidia/NVIDIA-Nemotron-3-Super-120B-A12B-NVFP4` |
| `minimax-m2.7` | [`Kanposer/MiniMax-M2.7-speedy-colibri-nvfp4`](https://huggingface.co/Kanposer/MiniMax-M2.7-speedy-colibri-nvfp4) | `nvidia/MiniMax-M2.7-NVFP4` |
| `minimax-m3` | [`Kanposer/MiniMax-M3-speedy-colibri-nvfp4`](https://huggingface.co/Kanposer/MiniMax-M3-speedy-colibri-nvfp4) | `nvidia/MiniMax-M3-NVFP4` |
| `glm-5.2` | [`Kanposer/GLM-5.2-speedy-colibri-nvfp4`](https://huggingface.co/Kanposer/GLM-5.2-speedy-colibri-nvfp4) | `nvidia/GLM-5.2-NVFP4` |
| `deepseek-v4-flash` | [`Kanposer/DeepSeek-V4-Flash-0731-speedy-colibri-mxfp4`](https://huggingface.co/Kanposer/DeepSeek-V4-Flash-0731-speedy-colibri-mxfp4) | `unsloth/DeepSeek-V4-Flash-0731` |
| `kimi-k3` | [`Kanposer/Kimi-K3-speedy-colibri-mxfp4`](https://huggingface.co/Kanposer/Kimi-K3-speedy-colibri-mxfp4) | `unsloth/Kimi-K3` |

All six containers are complete and verified on the Hub. `scripts/models.toml` is the only
registry — `run-dgx.sh` reads it rather than keeping its own copy.

---

## 5 — What was changed, per model

**No fine-tuning, distillation, or modification of model behaviour anywhere.** Weights are
repacked into colibrì's container layout: routed experts become coalesced per-expert spans
so they can be streamed, and the dense tier is quantized for residency.

| model | experts | resident tier | notes |
|---|---|---|---|
| `nemotron-3-super` | NVFP4, **as published upstream** — repacked, not requantized | int8, optionally NVFP4 (`COLI_RESIDENT_NVFP4`, **+9.4% decode** at 0 ± 1.5% perplexity) | `.mixer.` marker auto-classifies the hybrid stack |
| `minimax-m2.7` · `minimax-m3` · `glm-5.2` | requantized fp8 → **NVFP4** (4-bit block-scaled) | int8 | GLM keeps its DSA indexer weights (`COLI_KEEP_INDEXER=1`) so sparse attention still runs |
| `deepseek-v4-flash` · `kimi-k3` | **MXFP4 passthrough — bit-exact**, QAT-native upstream | int8 | nothing is re-encoded; convert copies the expert bytes through |

The MXFP4 models pass through untouched because their upstreams are already
quantization-aware-trained at 4-bit — requantizing them would only lose information.

Adding a model is an `Arch` variant + a convert mapping + one registry block; the checklist
is in [scripts/README.md](scripts/README.md).

---

## 6 — Quality

Quantization here is chosen on measured perplexity, not on what compresses best.

**Resident tier — int4 was wrecking the model:**

| | perplexity ↓ | top-1 ↑ |
|---|---|---|
| int4 resident | 48.665 | 32.1% |
| **int8 resident (shipped)** | **6.189** | **57.9%** |

int8 recovers it for ~7 GB more RAM. That is why the resident tier is int8 and not smaller.

**Experts — NVFP4 costs almost nothing:**

| | perplexity ↓ | vs e4m3 |
|---|---|---|
| e4m3 experts (8-bit) | 4.670 | — |
| **NVFP4 experts (4-bit, shipped)** | **4.707** | **+0.8%** |

Half the bytes for +0.8% perplexity, which is why NVFP4 is the default. Note the two tables
use different held-out texts and are not comparable to each other.

**A caution that cost real time:** reconstruction RMS does **not** rank formats across
families. NVFP4 perturbs weights *more* than int6 by RMS and yet costs zero perplexity —
the error it introduces is shaped in a way the model tolerates. Price a format by
perplexity (`coli ppl`), never by RMS.

Full quality record, including the MXFP4 passthrough rationale and the resident-NVFP4
sweep, is in [docs/PERFORMANCE.md](docs/PERFORMANCE.md).

---

## Rooted in colibrì — with gratitude

SpeedyColibri began as a Rust port of **[JustVugg](https://github.com/JustVugg)**'s
**[colibrì](https://github.com/JustVugg/colibri)**, and the foundation is theirs: the core
insight of treating VRAM, RAM, and disk as one managed memory hierarchy — streaming a MoE
model's routed experts on demand while keeping the dense part resident in low precision —
and the original, quality-preserving GLM-5.2 forward pass. That idea is what everything here
is built on. **Thank you.**

It has since grown into its own engine: on-GPU zero-copy experts on unified memory, flash-
and tensor-core attention kernels, an NVFP4 4-bit expert format, adaptive RAM residency,
multi-Spark expert-parallel over RoCE/RDMA, and five model families colibrì doesn't cover.
If you want the mature, portable, multi-platform original, start with colibrì.

## Licence

Each container inherits its upstream model's licence, linked on its Hub page. Engine code is
in this repository; see [docs/DEVELOPMENT.md](docs/DEVELOPMENT.md) for layout and build.
