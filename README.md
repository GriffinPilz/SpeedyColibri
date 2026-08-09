# SpeedyColibri

A Rust MoE inference engine for a single **DGX Spark** (GB10, 121.7 GiB unified memory).

Routed experts stream from NVMe on demand, the dense tier stays resident in low precision,
and RAM is filled adaptively with LRU eviction that cannot OOM. That is what lets one box
run a 744B model — or a 1.5T one — without sharding it across a rack.

Seven models run today, from a 5.3 GB ternary MoE to a 1.4 TB container.

---

## 1 — How fast is it?

One Spark, single sequence, greedy, 512-token prompt. Every figure is a median of repeated
runs on **one build**, gated on token identity so a "faster" number that changed the output
fails loudly instead of being reported as a win.

| model | on disk | prefill | decode | serving |
|---|---|---|---|---|
| **`maple-preview`** (20B-A1B) | 5.9 GB | **160.2 tok/s** (3.2 s) | **157.0 tok/s** (112.8 e2e) | **91.1 tok/s** |
| **`nemotron-3-super`** | 69 GB | **45.7 tok/s** (11.2 s) | **12.9 tok/s** | **9.97 tok/s** |
| **`minimax-m2.7`** | 122 GB | **21.0 tok/s** (24.4 s) | **5.1 tok/s** | **6.0 tok/s** |
| **`deepseek-v4-flash`** | 145 GB | **13.8 tok/s** (37.1 s) | **4.7 tok/s** | **4.4 tok/s** |
| **`minimax-m3`** | 229 GB | **16.7 tok/s** (30.6 s) | **2.5 tok/s** | **2.6 tok/s** |
| **`glm-5.2`** (744B) | 403 GB | **6.9 tok/s** (74.6 s) | **0.9 tok/s** | **0.92 tok/s** |
| **`kimi-k3`** (1.5T) | 1.4 TB | **1.8 tok/s** (4.6 min) | **0.35 tok/s** | — |

Measured 2026-08-07 on branch `ffn-devcopy-coverage-gate`, which is ahead of `main`. All 15
suites exited 0 and every token gate passed. K3 was measured separately on 2026-08-06 — its
suite alone takes ~1.5 h — and is unaffected by anything that changed since. Maple was added
and measured 2026-08-07 on the same box; its numbers are not comparable to a prior release
because there isn't one.

**Read the `decode` column as a FORWARD-PASS rate, not a token rate.** `coli gen`'s decode
timer brackets `forward()` and stops before the `lm_head` matmul, so every decode figure in
this table excludes the output projection. On the streaming models that is a rounding error
(GLM spends ~1100 ms/token, the head is single-digit ms). **On Maple it is not**: the head is
2.5 ms of an 8.9 ms token, so Maple's honest end-to-end decode is **112.8 tok/s** against the
157.0 forward-only — which is why its row carries both. `coli gen` and `scripts/bench.sh` now
print the pair for every model, so the omission is visible rather than inferred. Maple's row
was re-measured in full on 2026-08-08 (BF16 IO tier + warp-per-row expert kernels + split-K
decode attention + zero-copy KV); the other six rows are unchanged from 2026-08-07.

**And read the `serving` column as a 32-token REQUEST rate, not a token rate.** It divides
by the whole HTTP round trip, so a per-request cost paid once lands on all 32 tokens.
Maple's 91.1 is `52 ms + 9.36 ms x 32 tok`; the 9.36 is the same engine the decode column
measures, and `bench_serve.py` now prints the split so the two columns can be reconciled
instead of guessed at. That number was **70.5** until the accept loop stopped napping
100 ms between connections — below.

**All six serve figures were re-measured 2026-08-08 after that fix, and only Maple moved.**
That is the expected shape, not a disappointment: the ~100 ms is a constant, so it is 24% of
Maple's request and 0.3% of GLM's. m2.7 went 5.9 → 6.0 (+1.9%, close to the ~2% predicted
from its token cost), v4 4.4 → 4.38 and m3 2.6 → 2.60 — both flat. **GLM read 0.84 → 0.92,
and that is NOT the fix**: its own 12 prompts spread 0.73–1.24 tok/s in the same run, so 0.84
sits inside the noise of 0.92. K3 was not re-run; its suite takes hours and it would gain
0.1%.

**The biggest single win came from the phase nobody was looking at.** Everything above about
Maple concerns the expert path, because that is where a short-context profile said the time
went. Measured properly — decode only, prefill excluded, at the 512-token context the table
actually uses — the split is:

| phase | ms/token | share |
|---|---|---|
| **attn** | **10.33** | **66%** |
| moe | 2.68 | 17% |
| logits | 2.62 | 17% |

Attention, not experts. And the cause was the same mistake as the expert kernels, one level
up: at `S == 1` both GQA paths launch `grid = (H, S)` — **16 blocks for an entire layer** —
and the WMMA path fills a 16-row query tile with **one** real row. They are prefill-shaped
kernels running the decode regime. Inside the scalar one the tail is worse than the head: the
final V accumulation strides `d < D`, so with D=128 against 1024 threads only 128 are live and
each walks all 536 keys.

**Split-K (flash-decoding)** partitions the key range across `nsplit` blocks per head, each
taking a local softmax, then rescales and combines against a global max. Blocks go from H to
H×nsplit and each thread's V loop shrinks by `nsplit`. ABBA, one knob, tokens identical:

| | prefill-shaped | split-K |
|---|---|---|
| decode (forward-only) | 75.59 / 76.44 | **130.35 / 132.40 tok/s** (1.73×) |
| decode (end-to-end) | 63.58 / 64.18 | **98.31 / 99.47 tok/s** (1.55×) |

**The win scales with context and is ~neutral without it**, which is why `serving` barely
moves (69.1 → 69.8): that suite uses short prompts, so there are few keys to split, `nsplit`
collapses to 1, and the kernel degenerates to the old shape. Prefill is untouched — it is
S>1 and never takes this path. Nemotron, which shares the kernel, measured 12.54 vs 12.68
with identical tokens: neutral, as expected for a model whose decode is expert-streaming
with only 8 GQA layers. `COLI_GQA_SPLIT=0` restores the old kernels.

**Then the KV stopped being copied at all.** `coli_cuda_gqa_attn` re-uploaded the ENTIRE K and
V history on every call — `T*Hkv*D*4` each, ~52.7 MB per decoded token across 24 layers, when
exactly one row had changed. Measured pageable H2D is **43 GB/s** against 256 GB/s for a
device read, so that was ~1.23 ms/token, a quarter of what attention cost after split-K.

The obvious fix is a device-resident KV appended one row at a time, and it is deliberately
**not** what shipped: a cached device copy must be invalidated whenever the host rows change,
and they do — a rejected MTP draft rewrites positions, and `serve` reuses a cache across
requests. A device KV that silently goes stale is *wrong output*, not slow output, and this
repo has already shipped one pointer-keyed device cache that served the wrong bytes. Reading
the host buffer **zero-copy** has no such failure mode, because there is no second copy to go
stale — the kernel sees whatever the host last wrote. It is the same mechanism the expert
weights already use.

Measured, n=5 per arm: **forward-only 129.9 → 158.1 tok/s (1.22×)**, end-to-end ~99.6 →
~112.8. Tokens identical, and bit-exact by construction — the same bytes, just not copied
first. One caveat worth stating: zero-copy is **less consistent** than the upload it replaces
(140–161 tok/s across runs, against 129.6–132.9), and the very first run after a build read
111.85 before settling. It is faster in six runs of seven; `COLI_GQA_ZEROCOPY=0` restores the
uploads.

This is also why `serving` sits below `decode` for every row. The column is left as-is
rather than silently re-baselined, because every historical measurement in this repo is
forward-only and changing the metric would make new runs incomparable without anyone
noticing.

**That per-token head figure was wrong twice before it was right, both times in the
arithmetic rather than the kernel.** It first read 4.4 ms because the running total was
divided by the decode-step count while including the *prefill* logits call — which is the
first touch of `lm_head` and therefore pays a one-time ~38 ms device upload of the whole
622 MB weight. Then it still drifted, because the total was averaged over *every* step while
the `mean` it was being added to used only the warm half. The tell for both was the same and
should have been caught immediately: **the number went UP when the run got SHORTER** —
6.3 ms over 11 steps against 4.4 over 24 — which no genuine per-token cost can do. A fixed
cost divided by a variable denominator is not a rate.

The detour cost four A/Bs that all measured neutral (a warp-per-row GEMV, `uint4` vectorised
reads, a rows-per-block sweep, and removing a 608 KB per-token allocation), every one aimed
at a kernel that was never slow. `coli gpubench` now prints a **measured** streaming-read
ceiling and times `lm_head` in isolation against it, so the next such question is one
command rather than an afternoon.

**Maple is in a different regime from every other row, and the table flatters it.** At 5.3 GB
on a 121 GB box it is ~2300% coverage: nothing streams, nothing evicts, and `expert-load`
measures **1 ms**. The other six are all wholly or partly bound by reading experts off the
NVMe; Maple is bound by neither that nor DRAM. Read it as "what this engine does when the
model fits", not as a speedup over the others.

**Its decode was launch-bound, and 1.29× of that is now recovered.** Maple routes to top-8
experts on 24 layers, and each one used to be its own GPU call: **192 dispatches per decode
token** (counted, not estimated), 4 kernels and 2 blocking copies each, 54.7 µs apiece to
move 0.80 MB — an effective 14.6 GB/s, ~6% of this box's bandwidth. Roughly 38 µs of every
dispatch was launch and sync doing no work. Collapsing a layer's experts into one launch
triple, with the weights still read **zero-copy in place**, gives **decode 44.4 → 57.2 tok/s
and serving 43.7 → 56.6**, tokens bit-identical, ranges disjoint. Prefill is unchanged: the
grouped path is gated on the row count at the decision point and declines outside the decode
shape. `COLI_INT2_GROUP=0` restores the per-expert arm for comparison.

**The server was napping 100 ms between connections, and only Maple was fast enough to
show it.** The accept loop polled a non-blocking listener and slept 100 ms whenever nothing
was pending, justified in its own comment as "nothing next to a multi-second generation" —
true of GLM at ~1100 ms/token, false of Maple at 8.2. A sequential client pays the *whole*
nap, not half of it: its next request lands microseconds into a fresh one. The proof needed
no model at all — `GET /health`, a route that returns a constant, measured **63–97 ms** with
TCP connect at 0.1 ms. Waiting on the socket with `poll` instead of on the clock keeps the
bounded shutdown check and removes the latency: **`/health` 63–97 → 0.09 ms**, a 1-token
request **152.5 → 50.9 ms**, and Maple's serve median **70.5 → 91.1 tok/s**. Every model
gains the same ~100 ms; only a model whose token costs 8 ms notices.

**What is left of the per-request cost is a SHORT PREFILL, and the expert path declines to
group it.** With the nap gone, a request still pays a fixed 51 ms on Maple before the
per-token rate applies — and 270 (m2.7), 472 (m3), 575 (v4), 617 (nemotron), 941 (GLM).
`COLI_SERVE_TIMING=1` marks every stage of a request, and the answer is not where reading
the code suggested: JSON, tokenize, `reserve_ram`, ledger admission and KV allocation
together cost **0.08 ms**. All of it is inside generation, and the engine's own per-request
counters localize it further — for a 5-token prompt, **moe 38.4 ms** against 2.6 ms for a
one-row decode step, fifteen times the cost for five times the rows.

The cause is a gate doing exactly what it says. `try_expert_group_int2_decode` requires
*one row per expert, all the same token*; a 5-token prefill routes each expert a different
row set, so it declines and the per-expert path runs ~30–40 launch triples per layer across
24 layers. At the ~38 µs per dispatch already measured for this model, ~800–960 dispatches
is 30–36 ms — which is the 38 ms. The gate was written against prefill at 512 tokens, where
per-dispatch overhead is ~12% and grouping does not pay; at 5 tokens there is almost no work
to hide it behind and the overhead is nearly all of it. **The right gate is rows-per-expert,
not prefill-versus-decode** — the same lesson as three earlier expert-path defaults. Not yet
fixed: a multi-row grouped kernel is a real change, not a threshold edit.

**Verified against the reference implementation.** Maple's own `modeling_maple.py` was run
unmodified (only `fa3.py` swapped for an SDPA-backed equivalent, since the published one
hard-requires a FlashAttention build), and compared two ways. The residual stream matches
layer by layer — every one of the 23 comparable layers within **0.06–3%**, including all six
NoPE/global layers, which a wrong router, window or rope would not produce. Under teacher
forcing over 32 positions the top-1 choice agrees **31/32**. The single disagreement is a
**0.125-logit near-tie** in the reference's own top-2 (19.750 vs 19.625) — about two ULPs of
the bf16 it computes in, where colibrì runs f32.

**The dense tier is now exact, and it is not free everywhere.** `COLI_IO_BITS=16` stores the
embeddings and `lm_head` as f32 rather than per-row int8, which lifts reference agreement
from **29/32 to 31/32**. Prefill and decode are unchanged (140.3 vs 146.4 and 57.0 vs 57.2,
both inside this box's drift) — decode is bandwidth-underutilised enough to absorb `lm_head`
growing 311 MB → 1.24 GB per token. **Serving is not**, and that is measured rather than
inferred: ABBA-interleaved on one binary, f32 gave 47.44 / 47.13 and int8 gave 55.53 / 56.32
tok/s — **~1.18×**, arms cleanly separated with no drift trend. So the trade is ~6 points of
reference agreement against ~18% of serving throughput. The shipped container takes the
quality side; drop `COLI_IO_BITS` for the 5.3 GB int8 build.

**RESOLVED: serving pays because it MEASURES the `lm_head`, and decode does not.** Adding
per-request phase counters to `serve` (it had none — the production path was the one path
with no instrument) shows `attn` and `moe` identical between the two IO tiers to within
0.1 ms, and the entire gap in `logits`: **4.2–4.6 ms/token at f32 against 1.5–2.0 at int8**,
which is the whole 14.7 vs 12.0 ms difference. The earlier "not the forward math" conclusion
was drawn from `gen`, which excludes the one weight that differs.

**A second win on top: the int2 kernel's read granularity.** `weight_at` decodes fmt 3 one
element at a time, so four consecutive lanes load the same byte. Giving each thread **one
byte — 4 ternary weights** cuts loads 4x while keeping all 256 threads busy: `maple-o`
[2048,2048] goes 30.7 → 20.8 µs, matching int8's 20.9 while reading a quarter of the bytes,
and **decode 57.0 → 64.3 tok/s** with serving 47.2 → 52.4. Token gates pass and reference
agreement is unchanged at 31/32.

Granularity, not width, is the knob — and the difference is not small. The obvious version
of this change, a `uint32` per lane (16 weights), measured **38% WORSE**: at 16 weights per
thread an I=2048 row leaves half the block idle and the expert down-projection (I=512)
leaves 7 of 8 threads idle. An int2 row is only 512 bytes; a wide read empties the block
faster than it fills the bus.

**And the reason that failed was the real bug: the block was the wrong unit, not the read.**
The grouped expert kernels gave one **256-thread block** to each output row. Maple's expert
intermediate is 512, the smallest in the fleet, so:

| projection | row bytes | threads | work per thread |
|---|---|---|---|
| down | `ceil(512/4)` = **128** | 256 | threads 128–255 do **nothing**; the rest do ONE byte |
| gate/up | `ceil(2048/4)` = 512 | 256 | 2 bytes |

…and each of those blocks then ran an **8-round shared tree reduction with a barrier at
every level**. The reduction cost more than the arithmetic it was reducing.

The arithmetic that made this worth checking: a token routes top-8 over 24 layers = 192
experts × 786 KB = **151 MB of expert weight**. At the measured 257 GB/s ceiling that is
**0.59 ms**; the path was spending ~10.7 ms — about **5% of the roof**. A number that far
off is a kernel-shape bug, not a bandwidth story.

**One warp per row** — all 32 lanes busy, five `__shfl_down_sync`, no shared memory, no
block barrier, 8 rows in flight per block. ABBA, one binary, one knob, `lm_head` flat at
2.5 ms as the control, tokens identical in all four runs:

| | block-per-row | warp-per-row |
|---|---|---|
| decode (forward-only) | 63.95 | **76.91 tok/s** (1.20×) |
| decode (end-to-end) | 55.14 | **64.51 tok/s** (1.17×) |

So the check to run before calling any GEMV kernel ALU-bound is **`row_bytes / blockDim`**.
If it is single digits, the kernel is starving and the reduction dominates. That also
retires this repo's earlier "int2 is ALU-bound" conclusion: it was **occupancy**, and the
lever was the thread→row mapping.

The same kernel shape applies to the dense int2 attention projections, and there it is a
real but *small* win: isolated in `gpubench`, `maple-qkv` [3072,2048] goes **34.8 → 21.0 µs
(1.66×, 59 → 125 GB/s)** while `maple-o` [2048,2048] is unchanged. But qkv is 24 calls of a
15.6 ms token — worth ~2%, which is **below what the decode harness resolves**, and the
end-to-end A/B duly came back neutral. It is kept because the kernel is strictly better in
isolation, not because the headline moved.

Two things this is *not*. It is not the existing grouped arm — that one stages weights host
to device at ~0.74 GB/s and measured **10% worse on DeepSeek-V4** while cutting 301
dispatches to 43, because V4's experts are 13.37 MB and already run at 70 GB/s where
launches are noise. And it is not a bandwidth fix: at 57.2 tok/s Maple still moves only
~525 MB/token, so headroom remains.

**Achieved bandwidth is now reported against a measured ceiling, not a spec sheet.**
`coli gpubench` runs a do-nothing streaming read of a 1 GiB device buffer and gets
**257 GB/s** — 94% of the GB10's 273 GB/s paper figure, so the probe is sound. Every GB/s
in these tables should be read against that number. It also settled the largest read in the
engine: `lm_head` at bf16 measures **250 GB/s in isolation, 98% of the ceiling**, and f32
measures 262. The output projection is *done* — there is no kernel work left to find there,
and the four rewrites that chased it were chasing a reporting bug.

Getting that probe right needed one correction of its own. The first version reported
**133,950 GB/s** — 1 GiB in 8 µs, exactly the kernel launch floor — because its
"keep the loads alive" guard was `threadIdx.x == 1025`, which nvcc can prove false at
blockDim 256, so it deleted the branch and the reads with it. The sentinel is a runtime
argument now. A dead-code eliminator always wins against a condition it can evaluate.

**Prefill moved; decode and serving did not.** That split is the point, not a coincidence:
the change behind it gates expert-weight staging on coverage, and the S==1 decode path skips
staging by construction. Had decode moved, the mechanism would be wrong. See
[docs/PERFORMANCE.md](docs/PERFORMANCE.md).

**Do not read the whole prefill gain as that change.** Isolated by an A/B on one binary it is
worth **−10.0% nemotron, −6.5% M2.7, −2.4% M3, and nothing on V4 or GLM**; the table also
shows V4 +1.5% and GLM +4.5%, which the mechanism says must be ~0 — that residue is this
box's day-to-day drift, independently measured at ~12% on an unmodified binary. The A/B
ratios are the attributable part; these absolutes are simply what the machine does today.

Two separate changes account for it, each isolated by its own A/B on one binary,
ABBA-interleaved in a single session with tokens gated identical.

**1 — Pooled scratch + a rows-per-expert dispatch gate**, against a freshly built `main`:

| | `main` | with the fix | ratio |
|---|---|---|---|
| **`minimax-m3`** (n=6/arm) | 38458 ms | 31559 ms | **1.22×** — arm ranges 2.0% and 2.4%, disjoint |
| **`glm-5.2`** (n=4/arm) | 90497 ms | 73980 ms | **~1.22×** — direction certain, magnitude soft |

M3's is as clean as this box gets: 6.3 s separates the slowest new run from the fastest old
one. GLM's *direction* is equally certain — the ranges don't overlap either — but its new arm
spans 17% and drifts slower run over run while the `main` arm holds flat, which nothing here
explains, so read it as "roughly 1.2×". Nemotron, M2.7 and V4 are untouched by this one: the
first two never took the affected path and V4 measured 1.35% apart at n=6/arm against a 9.6%
spread. The fix was pooling per-expert scratch buffers whose page faults were being billed to
`gather` and `scatter`.

**2 — Coverage-gated expert-weight staging**, `COLI_FFN_DEVCOPY=1` (the old always-on
behaviour) against the gate, same binary:

| | coverage | old | gated | |
|---|---|---|---|---|
| **`nemotron-3-super`** | ~155% | 12504 ms | 11254 ms | **−10.0%** |
| **`minimax-m2.7`** | ~84% | 27012 ms | 25269 ms | **−6.5%** |
| **`minimax-m3`** | ~37% | 32306 ms | 31529 ms | **−2.4%** |
| **`glm-5.2`** | ~18% | 81850 ms | 84311 ms | +3.0% = noise; both arms run identical code |

Staging costs GPU time and buys I/O decoupling, so the answer is coverage. At 18% GLM streams
nearly every expert *during* the forward pass and the copy earns its keep; at 155% nemotron
has nothing to decouple from and it is pure waste. GLM's arms are behaviourally identical
under the gate, which makes its +3.0% a calibration of this box's noise rather than a result.

**Neither ratio comes from comparing this table against an older one, deliberately.** The same
unmodified binary measured V4 prefill at 41 s in July and 36 s in August. Both stories, and
the eight wrong hypotheses that preceded the first, are in
[docs/PERFORMANCE.md](docs/PERFORMANCE.md).

**Read the decode column as a disk-streaming ladder, not a quality ranking.** Decode is bound
by how many expert bytes each token pulls, so it tracks *model size against 121 GB of RAM*
almost perfectly. Prefill does not — M3 beats the smaller V4 there, because V4's attention
costs 14.2 s to M3's 7.3 s (Hyper-Connections, the Compressor and the Indexer are not cheap),
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
| **`maple-preview`** | 20B-A1B | GQA (16Q/4KV, partial rope 64), per-layer QK-norm, **3:1 sliding(512)/full interleave with NoPE on the global layers** | 256, top-8 | **int2 (ternary)** | 5.3 GB |
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

**Maple is also the only one with a non-uniform attention span**: 18 of its 24 layers see
only the last 512 tokens and the other 6 see everything with *no positional encoding at all*
(the reference applies RoPE only where there is a sliding window). It is the only model here
that reaches its full architectural context — **131,072 tokens** — with room to spare.

Maple and nemotron fit RAM outright; the rest stream. Coverage — the share of a model's
experts that can stay resident — is the axis that predicts nearly everything about its
behaviour, and Maple sits at ~2300% of it, far outside the range the other six occupy.

---

## 3 — How to run each model

**All seven are published as ready-to-run containers, so a fresh host downloads one instead
of paying a conversion.** `hf_repo` in the registry is the container — the engine loads it
directly, and nothing is converted at any point below.

```bash
docker/run-dgx.sh -m maple -p 8080        # or: nemotron · m2.7 · m3 · glm · k3 · v4
```

Add `-h <hf_token>` (or set `HF_TOKEN`) on the **first** run only — it is needed to pull the
container, not to serve it.

Without Docker, by registry name:

```bash
scripts/fetch.sh maple-preview            # download the container (~5.9 GB)
scripts/serve.sh maple-preview 8080       # …or just this: serve fetches what's missing
scripts/model.py list                     # what's registered
```

`serve.sh` downloads a container that isn't on the host; `fetch.sh` is the same download on
its own, for when you want it to happen at a time you chose. Both are idempotent — over a
complete directory they verify it, over an interrupted one they finish it. `SERVE_NO_FETCH=1`
restores the old "container missing" failure. **The benchmark scripts never fetch**, so a
measurement cannot quietly become a several-hundred-GB transfer.

Then any OpenAI client:

```bash
curl -s localhost:8080/v1/chat/completions -H 'Content-Type: application/json' -d '{
  "messages": [{"role": "user", "content": "The capital of France is"}], "max_tokens": 16
}'
```

**One model per process** — which one loads is decided by the container you point at.
`kimi-k3` wants `COLI_O_DIRECT=1` (1.09× prefill / 1.13× decode expert-load, tokens
identical; off by default because it *loses* on GLM and the mechanism is unexplained).

Converting from the upstream checkpoint is the other route, and it is only worth it if you
want to change how the model is packed. Maple is the cheap one to try — ~25 s for the whole
pass, though the source download is 40.4 GB against the container's 5.9:

```bash
scripts/convert.sh maple-preview          # registry paths + the model's convert_env
```

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
| `maple-preview` | [`Kanposer/maple-preview-speedy-colibri-int2`](https://huggingface.co/Kanposer/maple-preview-speedy-colibri-int2) | [`deepgrove/maple-preview`](https://huggingface.co/deepgrove/maple-preview) |

All seven containers are on the Hub, and `scripts/fetch.sh <model>` pulls any of them by
registry name. `scripts/models.toml` is the only registry — `run-dgx.sh` and `fetch.sh` read
it rather than keeping their own copies.

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
| `maple-preview` | **ternary → int2, bit-exact** (`fmt 3`) | **bf16 IO tier, bit-exact** (`fmt 2`) | the released BF16 weights are ALREADY ternary — see below |

**The IO tier (embeddings + `lm_head`) is BF16, and that is free in both directions.** These
two are genuinely dense — no ternary structure to exploit — so Maple ships them exactly
rather than at int8, which is worth 31/32 teacher-forced top-1 against the reference instead
of 29/32. Exact used to mean F32, which was **2× the bytes for zero extra information**: the
checkpoint is bf16, so every stored f32 carried 16 zero mantissa bits. `lm_head` alone is
311M parameters, read once per generated token, and on a model that fits RAM it is the
single largest per-token read in the engine.

Storing those weights as bf16 and widening back to f32 in-kernel is the *same arithmetic* —
f32 accumulation either way — so the logits are bit-identical, which the token gate confirms
end-to-end across every arm of the A/B. The container goes **7.1 → 5.9 GB**, and forward-only
decode is unchanged, which is the control: the effect is confined to the phase that reads
the weight. The converter verifies the round trip per tensor and falls back to F32 rather
than round a genuinely-f32 source, so exactness is checked rather than assumed.

Measured ABBA on one binary, two containers differing only in IO dtype, corrected timer:

| IO tier | `lm_head`/token | ms/token | end-to-end decode |
|---|---|---|---|
| F32 | 5.1 ms | 20.6 | 48.59 / 48.49 tok/s |
| **BF16** | **2.5 ms** | **18.0** | **55.48 / 55.57 tok/s** |

**`lm_head` 2.04× (the byte ratio is exactly 2.00), end-to-end 1.14×**, arms non-overlapping.
An earlier draft claimed 1.21× from the pre-fix timer; that comparison was not sound, because
the inflation it carried was a one-time weight upload and f32's upload is twice bf16's, so
the slower arm was penalised twice as hard.

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

**Maple is the one model where compression costs nothing at all — and it is not because
the format is clever.** Every expert and attention projection in the released BF16
checkpoint is *already exactly ternary*: `{-s, 0, +s}` with one scale per output row, ~38.7%
exact zeros. So storing them 2 bits at a time is a **re-encoding, not a quantization** —
colibrì's `fmt 3` int2 decodes as `field - 2` ∈ `{-2,-1,0,+1}`, ternary uses three of those
four codes, and `dequant(pack(w)) == w` bit for bit. **40.4 GB → 5.3 GB across 96% of the
parameters with no accuracy question to answer.**

This is checked rather than asserted: the converter verifies the round trip per tensor, and
a tensor that ought to be ternary and is not **aborts the conversion** instead of quietly
falling back to a lossy format. All 18,528 expert and attention tensors passed; the router,
norms, embeddings and `lm_head` are genuinely dense (thousands of distinct values per row,
against three) and stay int8 like every other model here.

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
