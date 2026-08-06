# Environment variables

<!-- Extracted from README.md 2026-08-06 to keep the README to the six things a
reader needs first: speeds, models, how to run them, where they come from, what was
changed, and quality. Nothing here was trimmed in the move. -->

`nvidia/GLM-5.2-NVFP4`).

**Environment variables** (all optional; pass as `VAR=value docker/run-dgx.sh ...`):

| Var | Meaning | Default |
|---|---|---|
| `HF_TOKEN` | Hugging Face token for the first download (alt. to the `hf_...` arg) | none |
| `COLI_PORT` | listen port (a positional `port` arg overrides it) | `8080` |
| `COLI_WARMUP` | warm-up prompts, `\|`-separated | none |
| `COLI_CTX` | served context length (prompt + completion), e.g. `64k`. Clamped to what RAM can hold as KV and printed at startup; a request whose KV won't fit is rejected (507), never an OOM. Memory-bound on one node — see [Context & output length](#context--output-length): nemotron 262,144 (its own max) · **V4 1,048,576 (its own max)** · M3 ~450k · GLM ~290k · M2.7 ~200k. V4 and nemotron are the two whose ceiling is the *model's* limit rather than the box's | `32768` |
| `COLI_ALLOW_LONG_CTX` | `1` → serve past the model's advertised `max_position_embeddings`. Meaningful only for a **NoPE** model like Nemotron-3-Super, which has no positional table to overflow (its 262,144 is an advisory default; the model card documents up to 1M). The RAM clamp still applies. Quality past the validated length is not guaranteed — this is a quality decision, not a memory one | off |
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
| `COLI_RESIDENT_NVFP4` | **converter** knob (`coli requant-nvfp4`): `1` → also re-encode the **resident** weights (attention q/k/v/o, Mamba in/out-proj, fc1/fc2-latent, shared experts) from int8 to NVFP4. Embeddings and `lm_head` are never touched. Cuts Nemotron's decode traffic 8.87 → 6.18 GB/token and measures **+9.4% decode** (10.09 → 11.04 tok/s) at **0 ± 1.5% perplexity** across three corpora. Note the win is far below what the byte count alone suggests: NVFP4 decodes slower per byte than int8, and resident matmuls are only part of a decode step | off |
| `COLI_NVFP4_GEMV` | resident-NVFP4 decode kernel: `0` original, `1` wide read (one byte/lane), `2` wide read without shared staging of `x`, `3` uint32/lane (full cache line). The original read only **16 B per warp** (lanes 2j and 2j+1 fetched the same byte) and staged `x` in 16–32 KB of shared memory, capping an SM at ~3 blocks. Fixing both is most of the gain above; width stops paying past 32 B/warp | `3` |
| `COLI_QSIM` | **quality-sweep tool**, not a serving knob. `class:scheme[,…]` (e.g. `mamba:nvfp4`, `resident:6`) round-trips the named resident tensors through a target precision at load time, so `coli ppl` can price a quantization choice without rebuilding a container. Storage is unchanged, so it measures **quality only, never speed**. Each rule reports the RMS perturbation it actually applied and warns below 1% — an arm that perturbs nothing yields a perplexity that reads as "this precision is free" | off |

**DeepSeek-V4 only** — the mechanisms that give it long context. All default on; each is a
switch because its success case is invisible (correct output either way), so an A/B is the
only way to tell one apart from a silent no-op:

| variable | meaning | default |
|---|---|---|
| `COLI_DSV4_KV_BUDGET_MB` | how much raw KV a prefill may retain before it is chunked. Below the budget the prompt runs in one call; above it, chunks are as coarse as the budget allows. Chunking is **not free** — a 2048-token prompt chunked at 512 measured **1.41× slower prefill** for 133 MB saved — so the budget exists to make sure the cost is only paid when the memory is actually at stake. See [Context & output length](#context--output-length) | `1024` |
| `COLI_DSV4_CHUNK` | override the above with a fixed token chunk; `0` never chunks (the pre-chunking behaviour, for an A/B) | budget-derived |
| `COLI_DSV4_COMPRESS` | `0` disables the Compressor. Context past the 128-token sliding window then simply **is not there** — a hard edge, not a graceful degradation | on |
| `COLI_DSV4_INDEXER` | `0` makes every query attend to all closed compressed rows instead of the Indexer's top-k. The two arms are identical below ~2048 tokens of context and diverge only past it | on |

Diagnostics, any model:

| variable | meaning | default |
|---|---|---|
| `COLI_DEBUG_ACT` | `1` → per-layer L2 norm of the residual stream, to localise where a forward pass degenerates. On V4 it also reports the Hyper-Connection copies separately, since an HC failure shows as the copies diverging from each other rather than any one blowing up | off |
| `COLI_TRACE_STATE` | `1` → FNV-1a hash of the residual stream after every layer. Bitwise, so two states differing in the last ULP hash differently — this is for finding **where** two runs of the same input first diverge, which token identity cannot tell you | off |

Multi-node variables (`COLI_NUM_NODES`, `COLI_PEERS`, …) are in
[Multi-Spark](#multi-spark-expert-parallel) below.

Full deployment notes — GPU passthrough modes, building by hand or with compose,
the CUDA base image — are in **[DEPLOYMENT.md](DEPLOYMENT.md)**.

