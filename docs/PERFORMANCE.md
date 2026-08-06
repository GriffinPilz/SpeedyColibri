# Performance & quality record

<!-- Extracted from README.md 2026-08-06 to keep the README to the six things a
reader needs first: speeds, models, how to run them, where they come from, what was
changed, and quality. Nothing here was trimmed in the move. -->

## Where it stands

Running the real 403 GB model on **one** DGX Spark (GB10). The bottleneck is
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

**Where the wall actually is now, measured rather than assumed.** Sampling `nvidia-smi` and
`/sys/block/nvme0n1/stat` once a second through a real DeepSeek-V4 run gives, at decode:
**~10% of the 11.6 GB/s disk ceiling, 20% GPU, ~24% of 20 CPU cores, ~15% of memory
bandwidth.** Nothing is saturated. Decode is **latency-bound** — the per-layer chain is
strictly serial (attn → route → expert-load → expert-FFN), and each stage idles the
resource the previous one used. That reframes the remaining work:

- **Reading *faster* is finished.** GLM prefill already runs at **81.7%** of the device
  ceiling. Two attempts to beat it are recorded as negatives so they are not retried:
  buffered io_uring measured ~half our threaded `pread`, and `O_DIRECT` is regime-split
  (−18% GLM decode, but a win on K3) so it stayed an explicit flag rather than a default.
  What remains is reading **less** — residency and coverage, i.e. more nodes.
- **Overlap is harder than it looks.** The one dependency-free concurrency in a V4 step —
  running the shared expert during the routed experts' disk wait — hides 100% of its cost
  and is still a **15× loss**, because concurrent GPU work from a second thread takes an
  illegal memory access and a sticky CUDA error drops every expert to the CPU.
- **4-bit experts are dequant-bound, not bandwidth-bound.** `coli gpubench` puts 4-bit at
  ~190 GB/s against int8's 400–580 on the same shapes — slower in absolute time while
  reading half the bytes. The wins came instead from read width and cheaper dequant.
  Grouping their dispatches is **regime-split** and is now gated on mean rows per expert —
  see [the prefill fix](#prefill-the-mallopt-regression-and-its-fix--2026-08-06).
- **Long context is a storage problem before it is a compute one.** DeepSeek-V4's raw KV is
  now a **ring** — a query reaches back only `sliding_window`, so retaining more was dead
  storage — and prefill is **chunked** above a KV budget, which makes the retained rows
  constant in context rather than proportional to the prompt. Together: a 512-token prompt
  plus 40k generated holds 48.4 MiB instead of 3.78 GiB. The chunking is bit-exact, and it
  is budget-gated because it is *not* free — a fixed 512-token chunk measured **1.41× slower
  prefill** at 2048 tokens, for memory that was never at stake. See
  [Context & output length](#context--output-length).

Per-module port status and the milestone order live in **[PORTING.md](PORTING.md)**.

## Prefill: the `mallopt` regression and its fix — 2026-08-06

A re-measurement found MiniMax-M3 prefill at **1.358×** its July figure. Eight mechanisms
were proposed from code reading and commit archaeology; **all eight were wrong.** A `git
bisect` over 34 commits — one M3 prefill per step — named the commit in about an hour, and
the profile timers named the line.

The commit was `750fd10`, which pinned `M_MMAP_THRESHOLD` to 2 MiB. That `mallopt` is
*deliberate and load-bearing* (removing it costs Nemotron serve 2.4×), but it changes what a
multi-MB `vec![0f32; …]` costs: the allocation goes straight to `mmap` and is faulted in on
**first touch**, so the page-fault cost is billed to whichever code writes first. Three hot
paths were still allocating per expert or per layer:

| what | allocated | billed to | after |
|---|---|---|---|
| `xg` / `hh`, per expert per layer (~7680 per prefill) | ~478 allocs | `gather` 1126 ms | 148 ms |
| `out`, `[n_tokens, hidden]` per layer (~500 MB/prefill) | ~8 MB × 60 | `scatter` 2267 ms | 181 ms |
| `sh`, the shared expert, per layer | ~500 MB/prefill | `shared` 1088 ms | 732 ms |

Fixes: thread-local grow-only pools for `xg`/`hh` (`0fefba9`), a
`compute_experts_partial_into(.., out: &mut [f32])` that **accumulates** into the caller's
buffer instead of returning its own (`70f4a3b`), and pooling the shared-expert buffer in the
SwiGLU paths, which an earlier fix had wired into the Nemotron and Kimi paths only
(`15776ff`). Every CPU-side MoE timer then matched July: router 157 vs 166, gather 148 vs
148, scatter 182 vs 185, shared 732 vs 654, `other` 53 vs ~500.

A fourth change (`abb3d64`) is independent of the regression: the NVFP4 grouped SwiGLU path
is now gated on **mean rows per expert** (`COLI_NVFP4_GROUP_ROWS`, default 8) rather than
being always-on. Grouping trades a per-expert round-trip for staging every routed expert
through a pinned buffer — 148.6 GB on one M3 prefill. At 1 row/expert (decode) the round-trip
dominates and grouping wins; at ~25 rows (512-token prefill) the kernel amortises it and only
the staging is left. Per-expert at large S measured **1.28× on GLM**, 1.12× on M3, 1.04× on
M2.7, tokens identical. The crossover inside (1, 25.6) is bounded, not measured.

**Net effect of the whole branch**, measured against a freshly built `main` (a82ecdf),
ABBA-interleaved in one session, one variable — the binary. Provenance was probed rather than
assumed: the old binary must lack a string only the new one has, and vice versa.

| model | `main` | branch | ratio | |
|---|---|---|---|---|
| `minimax-m3` | 38458 ms | 31559 ms | **1.22×** | n=6/arm, ranges 2.0% / 2.4%, disjoint by 6.3 s |
| `glm-5.2` | 90497 ms | 73980 ms | **~1.22×** | n=4/arm, disjoint, but the branch arm spans 17% |
| `deepseek-v4-flash` | 36460 ms | 36952 ms | — | n=6/arm, 1.35% apart, *t* = 0.91 — no effect |

GLM's branch arm drifts slower run over run (66.1 → 73.2 → 78.9 → 77.7 s) while its `main`
arm holds flat at ~90 s across the same interleaved positions. Nothing here explains that, so
the direction is reported as certain and the magnitude as approximate. V4 was expected to be
flat and is: it is MXFP4, so the rows-per-expert gate never applies to it, and `dsv4_moe`
hands its result to `fc2`, so it keeps the owning `compute_experts_partial` and never
received the out-in-place accumulate.

**Three traps this cost, worth not repeating.** First, an ablation must move exactly one thing:
freeing evicted experts off the cache lock removed a 5136 ms lock-wait that matched a 5300 ms
`evict free` almost perfectly, and changed wall time *not at all* — `fetch()` is called by the
compute thread, so contention and critical path coincided and only one of them was a cost.
Second, a 24-token decode window measures warm-up, not steady state: a knob that only changed
*prefill* duration appeared to move GLM decode 1.19×, purely by giving the background
prefetcher 19 s more warm-up. At 100 tokens the arms converge. GLM's real steady-state decode
is **~1.23–1.33 tok/s**, not the ~0.9 a 24-token run reports.

Third — and this one invalidated a number that had already been written down — **a ratio
taken against a table row from another day is not a measurement.** Rebuilding `main` and
running it interleaved shows the *unmodified* binary doing DeepSeek-V4 prefill in 36.1 s
today against the 41 s the same code produced for the previous table. That 12% of cross-day
drift is larger than most of the wins recorded here, and it turned a genuine "V4 is
unchanged" into a published "V4 1.09× faster" until the interleaved run caught it. Every
ratio in this document should come from arms measured in one session, one variable apart.

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

GLM-5.2 specifically. For all four models side by side see
[How fast is it?](#how-fast-is-it) at the top.

| nodes | starting tok/s | current tok/s | how measured |
|---|---|---|---|
| 1 | 0.46 | **0.59** | `bench.sh glm-5.2 serve`, 2026-07-26 (was 0.46 counterbalanced, n ≥ 6, auto budget) |
| 2 | — | *not yet measured on 8/4* | prior repeated-prompt runs read ~1.95, but on the old model and inflated — not comparable; re-measure with RDMA-A |

The single-node number is flat from a 20 GB to a 55 GB cache (diverse traffic barely
reuses experts), so cache size is not a throughput lever here — headroom and avoiding
swap are.

### RAM residency by model (fill + OOM-safe eviction) — 2026-07-23

Every model fills RAM and evicts LRU experts under pressure. Measured on a 121 GiB Spark,
`bench_serve.py` diverse prompts, single node:

| model | routed experts vs RAM | policy (auto) | serving throughput |
|---|---|---|---|
| **MiniMax-M2.7** | ~105 GB ≈ 121 GB (near-fit) | fill ~101 GB + fadvise, hold working set | **4.83 tok/s** median — **1.94×** over the old 41 GB static default (2.49) |
| **MiniMax-M3** | ~193 GB (1.6× RAM) | fill RAM, keep page cache, LRU-evict | **2.05 tok/s** median — **1.22×** over the old static 1.68; box fills to 121 GB used and holds `avail` at the 3 GB floor for the whole run, **no OOM** |
| **GLM-5.2** | ~338 GB (2.8× RAM) | fill RAM, keep page cache, LRU-evict | not re-run (container offloaded to HF); ≫-RAM, expect a small gain like M3 at most — the page cache already served the hot set |

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

### Expert quantization: NVFP4 (default), e4m3 opt-out, MXFP4 passthrough

**A third format, for checkpoints that arrive 4-bit already.** DeepSeek-V4 and Kimi-K3 ship
QAT-trained **MXFP4** experts (block-32 nibbles + one E8M0 power-of-two scale per block, vs
NVFP4's block-16 with an e4m3 scale). Those are **copied through convert bit-exact** rather
than requantized — a dequant→requantize round trip measured **6.40% rel-RMS of pure loss and
5.9% more bytes**, i.e. strictly worse on both axes. `convert` detects them by their scale
sidecar and takes the passthrough path automatically; nothing in the pipeline ever quantizes
*to* MXFP4.

Both 4-bit formats are **dequant-bound rather than bandwidth-bound** — `coli gpubench 1 300`
prints a per-call table for each, and 4-bit tops out near 190 GB/s where int8 does 400–580 on
the same shapes *while reading half the bytes*. Consequences worth knowing before optimising
them: the levers are read width and cheaper dequant, not fewer launches (grouping dispatches
301→43 measured ~10% **slower**). One example of how sharp that is — the E8M0 scale decode is
re-run once per weight *byte*, and two endpoint comparisons inside it cost **14% of the whole
kernel**.

The rest of this section is the NVFP4 path, which is what the four older models use.

The routed experts (97% of the weights, and what every token streams) are stored as
**NVFP4** — 4-bit block-scaled. Resident weights (attention / dense / shared) stay 8-bit
int. Two source checkpoints feed the experts: modelopt **NVFP4**
[`nvidia/GLM-5.2-NVFP4`](https://huggingface.co/nvidia/GLM-5.2-NVFP4) (the default) and
block-scaled **FP8** [`unsloth/GLM-5.2-FP8`](https://huggingface.co/unsloth/GLM-5.2-FP8).

| expert format | bytes/wt | experts on disk | build (from source checkpoint) |
|---|---|---|---|
| **NVFP4** (e2m1 + per-16 ue4m3 block scale + global) — **default** | **0.5625** | **~338 GB** | `coli convert nvidia/GLM-5.2-NVFP4 <out>` |
| e4m3 fp8 (per-row) — 8-bit opt-out | 1.0 | ~601 GB | `COLI_XFP8=1 coli convert unsloth/GLM-5.2-FP8 <out>` |

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

