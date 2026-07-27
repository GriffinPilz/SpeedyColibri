# Model × lever test matrix

What has actually been measured, per model. **The point of this file is that "have we tried
X on Y?" should be answerable by reading, not by re-deriving it from code and commit logs.**

Legend — `PASS` measured and it helps · `NEG` measured and it does *not* help (do not retry
without new information) · `N/A` not applicable to this architecture · `TODO` never run ·
`BROKEN` known defective.

Rules for this file:
- **A number here must come from a real run**, with the config it was measured under. No
  estimates, no figures carried over from a different model or a different regime.
- **Negative results are first-class.** Most of the value is in the `NEG` rows: they are the
  experiments nobody should repeat.
- When a number is superseded, replace it *and* say what the old one was — a stale figure
  that looks authoritative has cost this project more than a missing one.

Box: gx10-42b2 (DGX Spark, 121.7 GiB). Unless stated, prompt = each model's registry prompt
(512 tokens), token-identity gated. Medians over ≥2 reps for prefill; **≥8 for decode**
(trap 8).

## Coverage

| lever | glm-5.2 | minimax-m3 | minimax-m2.7 | nemotron-3-super |
|---|---|---|---|---|
| end-to-end token validation | PASS | PASS | PASS | PASS |
| prefill bench | PASS | PASS 17.4 tok/s | PASS 18.9 tok/s | PASS 13.3 cold / **27.7 warm** tok/s (20.1 s on `ce53d25`; was 21.47 s quoted, 23.3 s pre-#91) — **~21× behind vLLM**. `COLI_RAM_GB=20` measures 16.8 s / 33.2 tok/s — see #98, unexplained and untested against decode |
| decode bench | PASS | PASS 2.07 | PASS | PASS **9.27 tok/s** — 2.77× behind vLLM |
| serve bench | PASS | PASS 1.38 | PASS | PASS 3.60 tok/s (12/12) |
| batch / genbatch | PASS 1.34× @B64 | **NEG** monotonic loss | TODO | **N/A** hybrid — guarded, PR #9 |
| chat template in serve | PASS | PASS (GLM-style) | PASS (M2 format) | PASS ChatML, PR #8 |
| adaptive expert-cache RAM | PASS | PASS | PASS | PASS near-fit, ~59/121 GB |
| KV reservation accounting | PASS | PASS | PASS | PASS after PR #10 (was 3.7× over) |
| grow-on-demand KV | PASS | PASS | PASS | PASS 3301-tok prompt, no OOM |
| **prefill scratch reservation** | TODO (method too slow) | PASS-ish **1.98×** 533.8 vs 270 | PASS **1.11×** 583.2 vs 527 | **BROKEN** 210 vs 24 KB/tok (8.75×) |
| **MTP / speculative decode** | PASS (break-even) | N/A no head | N/A no head in quant | **NEG (blocked)** head loads + 77% accept, but output DIVERGES and it is 1.18× SLOWER |
| COLI_PREFETCH_AHEAD | PASS 1.58× | PASS 1.26× | PASS **1.43×** | **NEUTRAL 1.00×** (RAM-resident) |
| COLI_RAM_GB sweep | PASS (flat) | TODO | TODO | TODO |
| hot-expert autopin | **NEG** ~10% loss | TODO | TODO | TODO |
| sliding-window attention | **NEG** lose-lose | N/A | N/A | TODO |
| 2-node expert-parallel | PASS 1.38× | TODO | TODO | TODO |
| TP attention (2-node) | PASS | TODO | TODO | TODO |
| **S==1 tile bypass** | PASS **1.152×** | PASS (in 1.081× below) | PASS (in 1.19× below) | PASS **1.53×** (PR #13) |
| **dedicated i8a16_gemv** | **NEG 0.965×** on top of #13 | PASS +4% | PASS (in 1.19× below) | PASS **1.12×** (PR #15) |
| **both S==1 paths (cumulative)** | PASS **1.111×** | PASS **1.081×** | PASS **1.19×** | PASS **1.79×** |
| **grouped nvfp4 relu2 experts** | N/A (fp8+gate) | N/A | N/A | PASS **+2.5%** decode-only (PR #19) |
| long context (>32k) | TODO | TODO | TODO | PASS 1M w/ COLI_ALLOW_LONG_CTX |

## Per-model notes

### nemotron-3-super — hybrid Mamba2 + GQA + latent-MoE
88 layers = 40 Mamba2 + 40 latent-MoE + 8 GQA (NoPE, no qk-norm). 512 experts top-22,
gateless ReLU² in a 1024-wide latent space. NVFP4 routed experts.

- **prefill 38.4 s cold / 23.1 s warm**, **decode 9.27 tok/s**, **serve 3.60 tok/s** (pre-#13).
  Warm prefill is **21.47 s** as of `a95a5b9` (#91); re-measured against a 22.90 s baseline
  in the same session, so the 23.1 above is the historical figure, not a second arm.
- ✅ **RESOLVED 2026-07-25 — the prefill figure that "does not reproduce" was a cache-state
  artifact.** The old note flagged 33.1 s as unreproducible against ~38 s runs and blamed an
  unidentified state. The state is the **in-process expert cache**: a cold `coli gen` pays
  ~9.9 s refilling 53.5 GB (18 131 misses), so one-shot runs land at 38.4 s while a warm
  `coli serve` request lands at 23.1 s. Anything in between — 33.1 included — is a partially
  warm page cache. **Always say which of the two a prefill number is.**
- **Experts are only ~59 GB against 121 GB of RAM (172% coverage) — they FIT.** Profiled:
  `18,179 resident, 0 evictions`; steady-state decode reads NOTHING from disk. It is *not* in
  GLM's disk-streaming regime, so GLM-derived "bytes-bound floor" reasoning does not transfer.
- **Decode profile — RE-MEASURED after PRs #13/#15** (matched differential NGEN=25−NGEN=1).
  Decode 109.5 ms/token; ~8.9 GB/token = **81 GB/s, 30% of ~273 peak** (was 36 GB/s / 13%).

  | component | ms/tok | share |
  |---|---|---|
  | **routed experts (gpu-ffn)** | **47.5** | **43.4%** |
  | mamba in/out-proj | 22.7 | 20.7% |
  | mamba scan | 9.7 | 8.9% |
  | attention (8 layers) | 8.5 | 7.8% |
  | conv+norm | 7.2 | 6.6% |
  | expert-load | 3.3 | 3.0% |
  | shared expert | 1.7 | 1.5% |

  ⚠️ **The pre-#13 ranking is obsolete — do not plan from it.** It had mamba in/out-proj at
  71.0 ms (28.7%) and the **shared expert at 50.9 ms (20.6%)**; those are now 22.7 and 1.7.
  Both fell out incidentally because their matmuls route through the same `coli_cuda_matmul`
  the GEMV work fixed. A planned "fused int8 relu² shared-expert kernel" was dropped after
  this re-profile showed it would chase a 1.5% slice.

  **Grouped expert dispatch: DONE, and it was only +2.5%** (PR #19). The 880 calls/token
  (22 experts × 40 layers) *are* real, but grouping them removed only the memcpy round-trips —
  it still launches 3 kernels per expert, and more importantly **NVFP4 experts are ZERO-COPY
  HOST READS, not VRAM**. ~51 GB/s is plausibly that ceiling, which no amount of grouping
  moves. A prediction of ~4× on this slice (13.8 tok/s) was wrong for exactly that reason.

  ⚠️ **This rules out cheaper expert dispatch as a decode lever.** The slice is bandwidth-bound
  on host reads and device-resident experts are a known 22.7× regression ([[zerocopy-is-load-bearing]]).
  What remains: **fewer forwards (MTP)** or **fewer bytes**, not faster dispatch.

- **MTP / speculative decode — head now converts and loads, but the lever is BLOCKED twice.**
  The head was being dropped outright at convert (`mtp.*` → `None`); it is now mapped into
  `model.layers.{88,89}` and loads (`has_mtp=true`). All 1040 source tensors convert
  (1033 quantized + 7 f32, 1.7 GB, 4 s via `COLI_MTP_ONLY=1` — no 67 GB re-convert).
  Nemotron's head is TWO sublayers (`mtp_hybrid_override_pattern == "*E"`) where GLM's is one.

  Measured on a 32-token prompt, `COLI_NGEN=24`, arms interleaved, 2 reps each. **These are
  NOT comparable to the 8.4 tok/s registry-prompt bench — only the within-run A/B is.**

  | arm | acceptance | tok/forward | ms/forward | ms/token | tokens |
  |---|---|---|---|---|---|
  | DRAFT=0 | — | 1.00 | 292 | **292** | `173c8cc8` |
  | DRAFT=1 | **77%** (10/13) | 1.85 | 636 | **344** (1.18× slower) | `4b201eb9` ✗ |
  | DRAFT=2 | 46% (11/24) | 2.00 | 770 | **385** (1.31× slower) | `3aa6ae18` ✗ |

  1. ⚠️ **CORRECTNESS: speculation changes the output.** Each arm is deterministic across reps
     and they disagree with each other — this is real divergence, not noise. **Mamba recurrent
     state is not rolled back on a rejected draft.** The verify forward runs `[next, drafts…]`
     and `mamba2_mixer` advances conv + SSM state in place for every token in the batch
     (`kv.mamba_conv_row_mut`, `selective_scan` mutating `state.data`); on partial acceptance
     the rejected tokens are permanently baked into the state. Attention KV is fine — stale
     rows get overwritten — which is why **this never bit GLM**. There is no snapshot/rollback
     anywhere in the engine. The sequences share their first 12 tokens then split, the
     signature of accumulating drift rather than a broken head.
     Fix: snapshot conv+SSM (~166 MiB: 40 layers × 128 heads × 64 × 128 f32) before verify,
     cache per-token mixer inputs (~5 MB), restore + replay the *scan only* for the accepted
     prefix. ~1 ms/step always + ~10–20 ms per rejection.
  2. ⚠️ **PERF: the verify forward falls off every decode fast path.** `expert-load` is
     *unchanged* between arms (8077 vs 8060 ms) — extra expert bytes are NOT the cost, it is
     compute. All three of the decode wins are gated to a single row: the tile bypass
     (`backend_cuda.cu:1192`, `S == 1`), `i8a16_gemv` (`:1200`, `S == 1`), and grouped NVFP4
     relu² (`gpu.rs:919`, `rows.len() != 1`). MTP verifies at S=2, so it silently reverts to
     the pre-#13 slow path. Consistent with the arithmetic: those PRs were 1.79× cumulative,
     so a slow-path 1-tok forward ≈ 292 × 1.79 ≈ 523 ms and a 2-tok one somewhat more —
     measured 637. (Diagnosis inferred from the gating + the timing match; NOT yet proven by
     forcing the fast path at S=2.)

  **The 77% acceptance says the head itself is good** — a mis-converted head sits near 0%.
  Both blockers must be fixed for MTP to pay, and #1 makes it slower still. Estimated payoff
  if small-S dispatch lands: 2-tok forward ≈ 356 ms → ~192 ms/token, **~1.5×**.
  Do NOT re-run MTP on Nemotron expecting a win until small-S dispatch exists.

### minimax-m3 / minimax-m2.7 — GQA transformers
Most infra is shared via `arch.is_gqa()`. M2.7 adds per-layer QK-norm.
- M3 batching is a **monotonic loss** (2.47→1.46 tok/s, B1→B32) — the opposite of GLM's win.
  Root cause: M3's byte-efficiency leaves only ~1.05× bytes/token to amortize. Do not build a
  single-box batching scheduler for it.
- Neither has an MTP head in its NVFP4 quant.

### glm-5.2 — MLA + DSA
The disk-streaming extreme: 735 GB of routed experts, ~5.9% RAM coverage. Most of the
I/O-oriented work (prefetch-ahead, eviction policy, autopin, RAM sweeps) was developed against
this model, and **those conclusions are regime-specific — re-measure before assuming they hold
for a model whose experts fit in RAM.**

## Cross-model transfer queue

**Findings established on one model that have NOT been checked on the others.** This is the
highest-value column in this file: most wins here were architecture-neutral, and most wrong
conclusions came from assuming a result transferred when it did not. A row leaves this table
only when someone measures it — not when it seems obvious.

| finding | found on | should transfer to | status |
|---|---|---|---|
| ~~**S==1 fast paths** (PRs #13+#15)~~ | nemotron (1.79×) | — | **DONE 2026-07-25 — glm 1.111×, m3 1.081×, m2.7 1.19×, all token-identical.** The win tracks how *compute*-bound a model is, so it is a nemotron-shaped lever, ~8–19% elsewhere. See below |
| ~~grouped expert dispatch~~ | nemotron | — | **DONE, +2.5% only** — the slice is bandwidth-bound on zero-copy host reads (~51 GB/s), not dispatch-bound. Low value elsewhere |
| **prefill scratch unreserved** (210 vs 24 KB/tok) | nemotron | **glm** still open | **PARTLY DONE 2026-07-25 — m3 1.98×, m2.7 1.11×.** Real but *architectural width*, not a broken formula like nemotron's 8.75×. glm unmeasured: the method needs a 2 GB cache, which made the first of four points take 46 min |
| **MTP head dropped at convert** | nemotron (bug), glm (fixed earlier) | — | FIXED on nemotron; m3/m2.7 have no head in their quants |
| **decode fast paths are gated to S==1** (tile bypass, `i8a16_gemv`, grouped relu²) | nemotron | **glm, m3, m2.7 — and any small-S path anywhere** | UNMEASURED **at S=2..4** — distinct from the S==1 transfer row above, which is closed. Anything running at S=2..4 (MTP verify, short batches, chunked prefill tails) silently gets the SLOW kernel. On nemotron this alone turns a 77%-acceptance MTP into a 1.18× loss |
| **Mamba state is not rolled back on a rejected draft** | nemotron | any future hybrid/recurrent arch | Structural, not a Nemotron quirk: speculation is unsound on ANY in-place recurrent mixer. Pure-attention arches are unaffected (stale KV is overwritten) |
| **chat-template arm required in serve.rs** | nemotron (was serving GLM markers) | any new arch | it is a 4th edit, not the documented 3 |
| ~~**`COLI_PREFETCH_AHEAD`**~~ | glm 1.58×, m3 1.26× | — | **DONE 2026-07-25 — m2.7 1.43×, nemotron 1.00× (neutral), both token-identical.** The win is proportional to how much of prefill is *disk* expert-load, so it orders exactly by regime and is not architectural. See below |
| **hot-expert autopin** (glm: ~10% LOSS) | glm | do NOT assume for others | glm streams from disk; nemotron's experts are RAM-resident — different regime |

**Regime warning.** Most I/O-shaped conclusions (prefetch-ahead, eviction policy, autopin,
RAM sweeps) were developed against GLM, which streams 735 GB of experts from disk at ~5.9% RAM
coverage. Nemotron's experts *fit in RAM* (172% coverage, 0 evictions). **A result measured in
one regime is not evidence in the other** — this is how the "bytes-bound floor" claim was wrongly
carried from GLM to Nemotron and cost a wrong diagnosis.

## The S==1 fast paths, measured on all four models (2026-07-25)

Method: **one binary, three env arms**, so only the dispatch decision varies and no rebuild
sits between the arms. `base` = `COLI_TILE_I8=force COLI_I8_GEMV=0` (reproduces pre-#13
behaviour) · `p13` = `COLI_I8_GEMV=0` (tile bypass only) · `full` = defaults (both).
NGEN=24, arms interleaved, token-identity gated.

| model | base | full | ratio | notes |
|---|---|---|---|---|
| nemotron | 4.65 | 8.3 | **1.79×** | experts RAM-resident (172% coverage) |
| m2.7 | 3.78 | 4.48 | **1.19×** | 8 reps — see trap 8 |
| glm | 0.99 | 1.10 | **1.111×** | `p13` alone is **1.152×**; #15 on top of it is **0.965×**, i.e. negative |
| m3 | 1.86 | 2.01 | **1.081×** | |

**The win scales with how compute-bound the model is.** Nemotron's experts fit in RAM, so
resident matmuls dominate its decode and these kernels sit on the critical path. GLM and M3
are expert-load-I/O bound (m3 ~51% expert-load), so the same kernels touch a small slice.

⚠️ **Corollary:** the S=2 verify cliff that blocks Nemotron MTP is this same number read
backwards, so **small-S dispatch is a nemotron-shaped lever, worth ~8–19% elsewhere.** Do
not build it expecting a *general* win.

> **REVISED 2026-07-25 by the vLLM head-to-head (below).** This corollary originally read
> "kills the planned follow-on work". That was right about generality and wrong as
> guidance. Nemotron is the model we are benchmarked against, MTP is worth a measured
> **1.80×** to vLLM on it, and small-S dispatch is the prerequisite for using our own MTP
> head (which already reaches 77% acceptance). It is not a broad lever; it is the specific
> unlock for the single largest decode win available. Build it — for nemotron.

⚠️ **GLM's #15 result is negative.** It is the one model where the dedicated GEMV loses
ground the tile bypass had already won. If `i8a16_gemv` is ever tuned, re-measure GLM
specifically rather than assuming the fix is universal.

### Prefill scratch (RSS slope)

Regress peak RSS (`/usr/bin/time -v`) against prompt length; the constant — weights plus
cache — cancels. **Pin `COLI_RAM_GB=2`**: at an 8 GB cache the adaptive monitor evicts
experts under pressure and absorbs exactly the growth being measured (observed as RSS
*falling* 3.8 GB from S=512 to S=1024).

| model | measured KB/tok | reserved | ratio |
|---|---|---|---|
| nemotron | 210 | 24 | **8.75×** — a broken formula |
| m3 | 533.8 | 270 | **1.98×** |
| m2.7 | 583.2 | 527 | **1.11×** |

M3 and M2.7 differ by architectural width (hidden 6144 / moe_inter 3072 versus 3072 / 1536),
not by an accounting bug. Use **≥4 points**: a 2-point fit put M2.7 at 676.7 KB/tok (1.28×),
which the 4-point fit corrected to 583.2 (1.11×).

## `COLI_PREFETCH_AHEAD`, measured on all four models (2026-07-25)

Prefill only (self-gated by `PREFETCH_AHEAD_MIN = 64`, so decode is never affected) and
**already default-on** — `COLI_PREFETCH_AHEAD=0` disables it. So this was not a decision
about whether to enable the lever; it was a check that the shipped default is right for
the two models that had never been measured. It is.

Method: one binary per model, arms **interleaved `0 1 0 1`** (trap 10) rather than
`bench.sh`'s all-of-A-then-all-of-B, one discarded warmup per arm, 5 reps, `NGEN=1`,
512-token prompt on both. Run twice — once with `COLI_TIMING` only and once with
`COLI_PROFILE` — so profile overhead was *measured* rather than assumed (trap 13).

| model | OFF | ON | ratio | expert-load OFF→ON |
|---|---|---|---|---|
| glm-5.2 | — | — | **1.58×** | (earlier measurement) |
| **minimax-m2.7** | 40 358 ms | 27 290 ms | **1.479×** profile / 1.429× timing | 18 799 → 12 771 ms |
| minimax-m3 | — | — | **1.26×** | 25 000 → 14 000 ms |
| **nemotron-3-super** | 38 291 ms | 38 120 ms | **1.004×** profile / 1.002× timing | 9 666 → 9 748 ms |

Token-identity PASS on both new models, 24/24 runs each (m2.7 `[44]`, nemotron `[17054]`).
Arms never overlapped on m2.7 (OFF 38.7–40.8 s, ON 26.7–31.6 s), so the result does not
depend on the median.

**The win is proportional to the disk-streaming fraction of prefill, not to architecture.**
Ordering by expert residency — glm (735 GB streamed, ~5.9% coverage) 1.58× → m2.7 1.43× →
m3 1.26× → nemotron (RAM-resident, 172% coverage, 0 evictions) 1.00× — reproduces the
regime axis exactly. There is nothing to hide when the experts are already in RAM.
Nemotron is **neutral, not negative**: its expert-load is unchanged (9.67 vs 9.75 s, well
inside noise), so the default costs it nothing and should stay on everywhere.

### Why prefill falls further than expert-load (resolved 2026-07-25)

The first pass showed prefill dropping 13.1 s while expert-load dropped only 6.0 s, which
pure overlap cannot explain. Measured directly — `/proc/diskstats` deltas per run plus the
full phase breakdown, 5 reps interleaved, token-identical.

**It is not fewer bytes. Prefetch-ahead reads 21% MORE from disk and still finishes
faster.** Median over 5 reps, m2.7 @512:

| | OFF | ON | |
|---|---|---|---|
| disk read | 108.6 GB | **130.9 GB** | +21% — speculative reads include waste |
| read requests | 944 k | 1 136 k | same 112 KB/request |
| throughput (wall) | 1 788 MB/s | **2 946 MB/s** | **1.65×** |
| mean queue depth | 6.2 | **11.5** | 1.85× |
| cache misses / evictions | 13 414 / 0 | 15 353 / 1 641 | prefetch evicts, then re-misses |

So the drive was simply **under-queued**: at QD 6.2 it delivered 1.8 GB/s, and merely
keeping more requests in flight bought 1.65× more bandwidth — enough to absorb 21% extra
traffic *and* finish 13 s sooner. (Ignore the diskstats "busy" fields: they imply
17 GB/s, above what this drive can do, so `io_ms`/`weighted_ms` are unreliable on
multiqueue NVMe. Wall throughput and byte counts are sound.)

**Where the saving actually lands** — leaf deltas sum to 13 815 ms against a 13 776 ms
prefill delta, so the phases account for 99.7% of it:

| phase | OFF | ON | delta | ratio |
|---|---|---|---|---|
| expert-load | 19 016 | 9 983 | −9 033 | 1.90× |
| **attn input proj** | **10 228** | **5 295** | **−4 933** | **1.93×** |
| gpu-ffn | 9 762 | 9 891 | +129 | 0.99× |
| rope+cache / core / o-proj / router | — | — | ±15 | 1.00× |

Only two phases move, and **35% of the win is the attention input projections**, not
expert load at all.

⚠️ **New open question: why do attention projections speed up?** They are matmuls over
*resident* int8 weights and have no dependency on expert prefetch. The obvious guess is
contention — 944 k synchronous preads competing with the projections for memory bandwidth
on GB10 unified memory — but that predicts the opposite sign, since with the lever ON the
reads are issued *during* attention. It is reproducible (5 reps, 1.93× ± tight) and
unexplained. There is no env knob for read-thread count on the `gen` path, so testing it
needs a code change. **Do not cite a mechanism until someone measures it.**

**Headroom remains.** 2.9 GB/s is still far under this drive's ~6.6–10.5 GB/s, and QD 11.5
is under the ~32 buffered reads need. The lever did not exhaust the read path; it only
stopped starving it.

## Head-to-head vs vLLM on nemotron (2026-07-25)

First real external baseline. Same model, same weights, same box — `vllm/vllm-openai:
v0.20.0-aarch64-cu130-ubuntu2404` pointed at the local `Nemotron-3-Super-120B-src` (the
NVFP4 checkpoint we convert from), TP=1 on 42b2, Marlin NVFP4 MoE + FlashInfer attention +
FP8 KV + async scheduling, i.e. the shipped recipe. Client sends the **identical 512 token
ids** we bench with, greedy, **one request at a time** — not concurrent, because our figure
is single-sequence and `max_num_seqs: 10` would otherwise compare aggregate throughput to
latency.

| | prefill tok/s | decode tok/s |
|---|---|---|
| colibrì — `coli gen`, cold per-process cache | 13.3 | **9.27** |
| **colibrì — `coli serve`, warm (the fair number)** | **24.1** → **25.9** after #91 | — |
| vLLM, MTP off | 573 | 14.24 |
| vLLM, MTP on (shipped) | **898** | **25.69** |

The 24.1 figure is the one this section was written against; #91 later moved it to 25.9
(22.1× behind), measured the same way — see the GPU-scan section below.

**Decode: 2.77× behind, and it splits in two.** 1.80× is MTP alone (25.69/14.24); 1.54× is
engine quality with speculation off (14.24/9.27). It is *not* all speculation — that was
the hypothesis this A/B was built to test, and it failed.

**Prefill: 23.8× behind** (573/24.1), against vLLM with speculation off.

⚠️ **Compare warm against warm.** Every `coli gen` invocation refills the in-process expert
cache from scratch — 18 131 misses, 53.5 GB, ~9.9 s — while vLLM is a *server* that paid
that once at startup. Measuring one-shot `coli gen` against a warm vLLM endpoint inflated
the gap to 42×. Under `coli serve` the first request costs 38.4 s and every later one
**23.1 s (24.1 tok/s), reproducible to 0.06%**. Use the serving number for any comparison
against a server.

### Where a warm 23.1 s prefill goes (557 tok, request 3 minus request 2)

| phase | ms | share | |
|---|---|---|---|
| **mamba** | **12 317** | **53%** | **scan 7 951 (CPU, 34% of prefill)** · conv+norm 3 162 · proj 1 203 |
| **moe** | **10 583** | **46%** | **gpu-ffn 9 544** · shared 378 · router 201 · gather 66 · scatter 54 · **expert-load 2** |
| attn | 71 | 0.3% | |

⚠️ **Two corrections to the cold-cache reading, both mine.** First I predicted the CPU
Mamba scan would dominate and the *cold* profile said no (19%) — so I re-prioritized MoE
first. Warm, the scan is **34%** and mamba is **53%**: the original instinct was right and
the cold profile was misleading. Second, the cold profile's headline MoE costs are almost
entirely first-request artifacts — `expert-load` 9 933 → **2 ms**, shared expert 6 618 →
**378 ms**. Only `gpu-ffn` survives at 9.5 s. This is trap 4 (never reuse a number from a
different regime) committed against my own measurement an hour earlier.

For scale: 12 B active params × 557 tokens ≈ 13.4 TFLOP. vLLM's 0.97 s at this length is
~14 TFLOP/s; our 23.1 s is ~0.6 TFLOP/s. Attention is 0.3% — the tensor-core attention and
DSA indexer work, both real wins on GLM, are noise on this model.

For scale: 12 B active params × 512 tokens ≈ 12.3 TFLOP. vLLM's 0.57 s is ~21 TFLOP/s; our
38.5 s is ~0.3 TFLOP/s, about 0.1% of this chip's NVFP4 peak. Attention is 0.2% of the
total — the tensor-core attention and DSA indexer work, both real wins on GLM, are noise
on this model.

### The root cause is one decision, and it explains both gaps

Four fast paths are gated on `S == 1`: the Mamba GPU scan (`forward.rs:348` — its own
comment says "Prefill (S>1) … fall to the CPU `selective_scan`"), the tile bypass
(`backend_cuda.cu:1192`), `i8a16_gemv` (`:1200`), and grouped NVFP4 relu² (`gpu.rs:919`).
The engine is tuned for exactly one token and falls off a cliff for anything else. That
single decision produces **both** results above: prefill at S=512 is catastrophic, *and*
MTP verify at S=2 is a 1.18× regression, which is precisely why we cannot use the
speculation worth 1.80× to vLLM.

Priority by measured value, **on the warm numbers** — with what each turned out to be worth
once measured (2026-07-26):

1. ~~**Chunked GPU Mamba scan**~~ — 34% of warm prefill. **DONE**, scan 7 462 → 794 ms
   (9.4×). The win was not the kernel; see the pinned-memory trap below.
2. **MoE `gpu-ffn`** — 41% of warm prefill, still open and now the whole story. NOT the
   `expert-load`/shared-expert costs the cold profile suggested; those are startup
   artifacts worth 2 ms and 378 ms warm. Its cause is settled — see the next section.
3. ~~**Small-S dispatch**~~ → **DISPROVED.** It does not unblock MTP, which is 1.55×
   *slower*, and lifting the S==1 gates recovers 1.6%. Same section.
4. **CUDA graphs + kernel quality** → the residual 1.54×. Note `cudaGraph` appears **0
   times** in this repo and there are 22 `cudaStreamSynchronize` sites, one per GPU entry.

Together (1)+(2) are 87% of warm prefill, against a 23.8× gap. **(1) is now landed, and it
is worth 1.066× warm — not the 1.54× predicted from its own phase saving. See the next
section: most of the scan's time was hiding other work.**

⚠️ **Also fix the per-process cache refill itself.** 9.9 s and 53.5 GB of re-read on every
`coli gen` is invisible under `serve` but dominates one-shot CLI use, which is how most of
this repo's benchmarking is done. Any prefill figure from `coli gen` carries it.

## RESOLVED (#97): the scan is worth 1.43×, and PR #28 shipped a 1.24× regression (2026-07-26)

⚠️ **This supersedes the section below, which is kept for the reasoning trail.** The "1.066×"
and the queue-drain hypothesis were both wrong, and they were wrong for the same reason:
they came from comparing **two binaries that differ by more than the scan**. Separating the
two effects took three experiments.

**1 — Hold the binary constant, switch the scan with `COLI_MAMBA_CPU`.** One build
(`ce53d25`), mirrored blocks gpu,cpu,cpu,gpu, 6 warm requests each, token-identical:

| arm | warm prefill | scan | shared |
|---|---|---|---|
| GPU scan | **19.98 / 20.20 s** | 620 ms | 5 258 ms |
| CPU scan | 28.66 / 28.68 s | 7 982 ms | 6 129 ms |

**The scan is worth 1.43×**, and nothing moves to the shared expert — it is *cheaper* in the
GPU arm. **Queue drain is dead as an explanation.** It never could have worked: every CUDA
entry point in `backend_cuda.cu` already ends in `cudaStreamSynchronize` on the one
per-device stream, so there is no outstanding queue for a later call to absorb. Reading the
code would have killed the hypothesis before any measurement.

**2 — Hold the scan constant, switch the binary.** Both arms on the CPU scan, so the only
difference is everything else between `63f500c` and `ce53d25`:

| phase | 63f500c | ce53d25 | Δ |
|---|---|---|---|
| **shared** | **380 ms** | **5 698 ms** | **+5 318** |
| attn | 70 ms | 804 ms | +735 |
| mamba | 12 318 ms | 10 938 ms | −1 380 (#96's real win) |
| gpu-ffn | 9 602 ms | 9 595 ms | +7 |
| **wall** | **23.28 s** | **27.96 s** | **+4.68 s** |

The phase deltas sum to the wall delta. So the shared-expert cost is **real time, not
re-attribution and not drain** — a 1.20× prefill regression.

**3 — Bisect it.** One block per commit, `COLI_MAMBA_CPU=1` throughout:

| commit | wall | shared | attn |
|---|---|---|---|
| `63f500c` base | 23.39 s | 1 547 | 72 |
| `9ed41bd` #26 | 23.03 s | 1 478 | 72 |
| `f36f798` #27 (the scan) | 23.05 s | 1 518 | 71 |
| **`a95a5b9` #28** | **28.65 s** | **6 296** | **886** |
| `ce53d25` | 29.00 s | 6 504 | 872 |

**PR #28 is the regression** — flat across #26 and #27, one step at #28. (These `shared`
figures are means over all requests including the warm-up, so they read higher than the
medians above; only the step between commits matters.)

It costs **~120 ms on each of 40 shared-expert calls and ~100 ms on each of 8 attention
calls** — a constant per-GPU-call penalty — while `gpu-ffn`, which dominates, is untouched.
`moe.rs` and `try_expert_ffn_relu2` are **byte-identical** across the range, so the cause is
runtime state, not the shared-expert code. #28 added: the mamba scratch `thread_local`
(#96), the profile split, and the `COLI_EXPERT_SEG`/grouped-relu² scratch and kernels. The
last of those is the only one that allocates new device and pinned-host buffers. Root cause
is **open — see task #98**, and it is worth ~1.30× of warm prefill (20.1 → ~15.5 s), which
makes it a bigger prefill lever than anything left in #90.

**What the numbers actually are, on `ce53d25` today:** warm prefill **20.1 s / 27.7 tok/s**
(the 21.5 s / 25.9 tok/s in the README was measured with a warmer page cache; re-measure
before quoting either). Base `63f500c` is 23.3 s, so main is genuinely ahead — just by
1.16× instead of the 1.43× the scan alone delivers.

⚠️ **Method note.** The first attempt at experiment 1 compared `target/release/coli` against
itself: that binary was a stale pre-#91 build left by an earlier bisect, so `COLI_MAMBA_CPU`
had nothing to switch off. Wall clock looked plausible (0.6% apart, tight spreads, token gate
green) and only the phase table — scan ≈ 7.9 s in *both* arms — gave it away. **Assert on a
phase counter that must move, not on the env var you set.** The A/B script now fails loudly
if the gpu arm's scan is over 3 s. Two further traps in the same session: `pkill -f "coli
serve"` never matches `/tmp/coli_base serve …` (kill by argv signature, not by binary name),
and a server holding 59 GB does not release its port within 3 s.

## Superseded: "the GPU Mamba scan is worth 1.066× warm, not 1.54×" (2026-07-26)

⚠️ **This section corrects a figure I published in PR #28**, which claimed warm prefill had
gone 23.1 → ~15.0 s (1.54×). That number was **derived**, not measured: I subtracted the
scan's phase saving from the warm total. This file's first rule exists precisely to catch
that, and I broke it. Measured, both arms in one session with one script:

| arm | warm prefill (median of 5) | spread | tok/s |
|---|---|---|---|
| baseline `63f500c` (pre-#91) | **22.895 s** | 0.90% | 24.3 |
| merged main `a95a5b9` | **21.469 s** | 2.64% | 25.9 |

**1.066×.** Same box, same 557-token prompt (`passage2.txt`), `coli serve`, request 1
discarded as warm-up, identical output token on all 10 requests. The baseline reproduces
the recorded 23.1 s to within 0.9%, so the protocol is sound and the comparison is fair.
The vLLM prefill gap moves 23.8× → **22.1×**.

### Where the other 5.8 s went

Per-request phase deltas (consecutive cumulative `COLI_PROFILE` counters, warm requests
only; profiling costs nothing here — the profiled baseline ran 22.75 s vs 22.90 unprofiled):

| phase | baseline | merged main | Δ |
|---|---|---|---|
| **mamba scan** | 7 899 ms | **616 ms** | **−7 283** |
| **shared expert** | 377 ms | **6 127 ms** | **+5 750** |
| attn | 69 ms | 886 ms | +817 |
| gpu-ffn | 9 595 ms | 9 567 ms | ~0 |
| **total** | **22 746 ms** | **21 207 ms** | **−1 539** |

The kernel did everything it promised — **the scan is 12.8× faster warm**, better than the
9.4× measured cold. But the shared expert, whose code this branch never touched, grew by
almost exactly what the scan gave back, and the two increases (+6 567 ms) account for 90%
of the missing saving.

**❌ DISPROVED by #97 — the section above has the measured answer.** The mechanism below was
inferred from a two-binary comparison and is wrong twice over: the scan is worth 1.43×, not
1.066×, and the shared-expert cost is a real regression from PR #28, not queue drain. Kept
verbatim because the reasoning error is instructive.

**Inferred mechanism, not yet proven:** the CPU scan was *absorbing* GPU wait. For 7.9 s per
request the CPU sat in `selective_scan` while previously-issued GPU work drained; with the
scan on the GPU the CPU races ahead and blocks at the next sync, which is the shared
expert's matmul. That is trap 18 (sync re-attribution) at full scale rather than the ~700 ms
version already recorded. Supporting it: the shared-expert path is unmodified, gpu-ffn is
unchanged to within 0.3%, and the increases sum to the decrease minus the net win. **To
confirm, put an explicit `cudaStreamSynchronize` plus its own timer immediately after the
scan** — if the wait moves there, it is queue drain and not a real regression.

⚠️ **Consequence for how prefill work is ranked in this file.** A phase's measured time is
an upper bound on what removing it can save, and on this engine it can be a *wildly* loose
one. Every remaining prefill estimate here — including `gpu-ffn` at 41% below — is subject
to the same discount until someone measures end-to-end. Cold `coli gen` does not show this:
the same change is a clean 1.21× there (38.0 → 31.5 s), because cold the CPU scan also
competes with the loader threads. **Two regimes, two different answers, and the warm one is
the one a server delivers.**

### Cross-model gate: the merged prefill work is inert off nemotron (2026-07-26)

Only one merged change reaches a non-nemotron model — `COLI_READ_SUB_KB` modified
`read_raw_shared` in `colibri-safetensors`, which every model loads through. It defaults to
the pre-existing 2 MiB tile, so it *should* be a no-op. Verified rather than assumed, both
binaries built from the two commits and run interleaved against each model's registry
prompt:

| model | token gate | prefill ratio (main/base) | main prefill |
|---|---|---|---|
| minimax-m2.7 | **PASS** `[39341]` ×4 | 0.9986× | 27.16 s (18.8 tok/s) |
| minimax-m3 | **PASS** `[67732]` ×10 | 0.978× fwd / **0.997× reversed** | 30.40 s (16.8 tok/s) |
| glm-5.2 | **PASS** `[374]` ×8 | 0.968× fwd / **1.037× reversed** | ~69 s (7.4 tok/s) |

Every run of both arms produced the identical token. **The change is inert, as designed.**

The two "regressions" are the ordering artifact now recorded in trap 10 — GLM mirrors almost
exactly (0.968 ↔ 1.037) when the within-pair order flips, with the same binaries. Pooled
across both orderings all three models are neutral to within ~1.5%.

Everything else that merged is architecturally nemotron-only: the GPU Mamba scan and
`MambaScratch` (nemotron is the only Mamba arch), `GroupScratch` and
`COLI_EXPERT_GROUP_PREFILL` (gateless-relu² NVFP4 path), and `COLI_EXPERT_SEG` (same, and
off by default).

## MoE prefill `gpu-ffn`: the weight path, and four hypotheses that died (2026-07-26)

`gpu-ffn` is the largest remaining prefill item. Four separate attacks on it produced four
negatives, and the last one — a kernel built specifically to exploit the diagnosis — is what
finally established the real constraint.

**The answer: it is ~90% weight streaming.** Solving the 512- vs 2048-token scaling
(8.916 s vs 12.966 s, where experts rise 1.13× and rows 4×) for its two components:

| | | |
|---|---|---|
| **weight-read** | **7.91 s (89%)** | 47.2 GB at 5.97 GB/s |
| row-compute | 1.01 s (11%) | |

CUDA-event timing agrees from the other direction: H2D 72 ms | D2H+sync 184 | host memcpy
18 | **kernel window 7 575 (84%)**.

### What was ruled out

| hypothesis | result | why it can't work |
|---|---|---|
| **Grouped dispatch at prefill** (`COLI_EXPERT_GROUP_PREFILL=1`) | 1.013× **slower** | The per-expert round-trip it eliminates is under 3% of the phase. It also still launches 3 kernels per expert. |
| **`COLI_FFN_DEVCOPY`** weight staging | slight **loss** (gpu-ffn 8 911 → 8 375 ms with it **off**) | Stages one expert at a time from *pageable* memory. |
| **Small-S dispatch / MTP** | MTP 1.55× **slower**; lifting the S==1 gate = 1.6% | See below. |
| **Occupancy** (segmented GEMM, `COLI_EXPERT_SEG=1`) | token-identical, ~1% **slower** | 86 blocks and ~39 000 blocks perform *identically*. |

### The occupancy disproof, and the reasoning error behind it

The per-expert path launches **86 blocks** at ~25 rows/expert and sits at 2.3% of memory
peak and 0.26% of compute peak — so it looked badly occupancy-starved. Varying rows per
expert seemed to confirm it: 4× the prompt tokens cost only 1.45× the time, i.e. **2.75×
better per token**, with no code change. A segmented GEMM (one grid per layer, row-tile
descriptors, ~453× the blocks per launch) was built to capture that at any prompt length,
and predicted to be worth ~5.7 s.

It is worth nothing: prefill 31 084 → 31 354 ms, moe 26 194 → 26 378, tokens `[17054]`.

⚠️ **The 2.75× was weight amortization, not parallelism.** As rows/expert go 25 → 88, the
weight bytes per output row fall 118 → 33 KB. Longer prompts read the *same* weights for
*more* rows. That is an arithmetic-intensity result, and a segmented GEMM changes launch
structure while leaving that ratio exactly where it was. The experiment was sound; the
inference drawn from it was not.

### Why MTP is blocked on the same thing

Matched-differential (N=13 minus N=1 tokens, so per 12 decode tokens):

| phase | DRAFT=0 | DRAFT=1 | |
|---|---|---|---|
| **moe** | 758 ms | **1 761 ms** | **2.32×** — all of it |
| mamba | 505 ms | 612 ms | 1.21× |
| attn | 181 ms | 105 ms | 0.58× — amortizes, as speculation intends |
| **total** | 1 444 ms | 2 478 ms | 1.72× |

Attention amortizes correctly and mamba is nearly flat. MoE more than doubles for the
**same 264 expert-touches per layer** — so it is not extra weight traffic and not the S==1
gates. The MoE path is simply far less efficient at 1–2 rows per expert than at exactly 1.
The 1.79× S==1 figure that motivated small-S dispatch measures the *resident matmuls*,
which this profile shows are exactly the part that already amortizes.

### What is left

The weight path itself: 47 GB per prefill at ~6 GB/s against a ~51 GB/s zero-copy ceiling.
The untried option is making a layer's experts **device-resident** before its GEMMs — 1.3
GB/layer to VRAM, then ~TB/s reads. This is **not** `COLI_FFN_DEVCOPY`, which stages one
expert at a time from pageable host memory. Before building it, check the arithmetic: 1.3
GB × 40 layers is ~52 GB of H2D per prefill, which is not obviously cheaper than 47 GB of
streaming reads. `COLI_EXPERT_SEG` is kept off-default because a device-resident version
would reuse its tile descriptors and dispatch.

⚠️ **Do #98 before any of this.** The PR that closed those four hypotheses also shipped a
~4.7 s prefill regression (see the #97 section at the top) — a bigger, cheaper win than the
device-resident experts sketched here, and it has to be removed before any new prefill
number from this section means anything. `COLI_EXPERT_SEG`'s device and pinned-host buffers
are the leading suspect for it, which makes them suspect as a foundation too.

## #98 in progress: three eliminations and one large, unexplained cache cliff (2026-07-26)

Chasing the #28 regression. **Root cause not found**; recording so the eliminations are not
repeated. Everything below is nemotron warm prefill through `coli serve`, token-identical.

**Eliminated — it is NOT:**

| hypothesis | test | result |
|---|---|---|
| the shared expert fell back to CPU | `coli gen` prints `gpu::ffn_count()` | **identical** — 297 matmuls, 18 131 fused expert FFNs on both binaries |
| the retained `MambaScratch` (#96) | new `COLI_MAMBA_SCRATCH=0` bypass, one binary, mirrored | **backwards** — reuse is a **3.1 s win** (20.14 s on vs 23.43/23.05 off), so the regression is ~7 s gross, not 4.7 |
| the profile split / instrumentation | cold `coli gen`, `COLI_PROFILE` 0 vs 1, both binaries | no effect (38.6/37.6 unprofiled, 38.7/37.8 profiled) |

**The regression does not exist cold.** Cold `coli gen`: base 38.9 s, current 37.8 s — current
is *faster*. And cold, `shared` is ~7 s on **both** (6992 vs 6668 ms). What differs is that
base collapses to **380 ms** once warm and current stays at ~5700. So nothing got slower;
**a warm-up effect was lost.**

**The handle: expert-cache size, and it is a cliff.** Per-request phases, 5 warm requests:

| cache | warm prefill | shared | attn | expert-load |
|---|---|---|---|---|
| default (fill ~101 GB) | 21.3–21.7 s | 6395–6510 ms | 822–932 ms | 1431 ms |
| `COLI_RAM_GB=80` | 21.65 s | 6487 ms | 943 ms | 1122 ms |
| `COLI_RAM_GB=65` | 20.20 s | 5660 ms | 826 ms | 1557 ms |
| **`COLI_RAM_GB=20`** | **16.76 s (33.2 tok/s)** | **873 ms** | **137 ms** | 3771 ms |

**`COLI_RAM_GB=20` is 1.29× on warm prefill** — shared 7.5× cheaper, attn 6.8× cheaper, for
+1.7 s of expert-load. `gpu-ffn` is identical everywhere (47.9 s cumulative both), as
expected for zero-copy reads.

⚠️ **The obvious explanation is wrong.** "Filling RAM reclaims the pages behind the resident
tier" predicts a gradient, and 65 GB (comfortably above the 59 GB of experts, 56 GB of
headroom) should already be clean. It is not — 65 and 80 GB behave like the default. Only
the small-cache regime collapses, so something else distinguishes it. Do not write the
pressure story into the docs until it predicts the 65 GB point.

**Do not ship `COLI_RAM_GB=20` on this evidence.** Nemotron's experts are 59 GB and fit; a
20 GB cache must cost decode, which was not measured here. The prefill number is real, the
trade is unquantified.

Next: instrument what the shared expert actually waits on at 101 GB versus 20 GB (CUDA
event timing inside `try_expert_ffn_relu2`, plus `/proc/vmstat` compaction and THP counters
across the two arms), and check whether the 380 ms warm base number survives on a box that
has been up as long as this one. Harnesses: `scripts/experiments/`.

## Serving context ceiling per model, and the hybrid's advantage (2026-07-26)

`coli capacity <container>` reads `config.json` alone — no model load — so the KV cost of
every model is cheap to check. Measured on the 121 GiB box:

| model | layers that cache KV | KV / token | fixed per sequence | max served `COLI_CTX` |
|---|---|---|---|---|
| **nemotron-3-super** | **8 of 88** | **24.0 KB** | 166 MB Mamba2 state | **262,144 = its architectural max**, 6.2 GB KV |
| minimax-m3 | 60 of 60 | 270 KB | — | ~402,690 (RAM-clamped, arch max 1M) |
| glm-5.2 | 78 of 78 | 351 KB | — | ~290k (RAM-clamped, arch max 1M) |
| minimax-m2.7 | 62 of 62 | 527 KB | — | ~190k (arch max 196,608) |

The nemotron row is a real startup clamp — `COLI_CTX=1m coli serve` printed
`context length: 262144 tokens (model max 262144; up to 6.2 GB KV)`, i.e. the RAM clamp
never fired. **A hybrid inverts the usual ranking.** Caching KV on 8 layers instead of 88
makes it ~11–22× cheaper per token than the GQA/MLA models, so it is the only one of the
four whose ceiling is the model's rather than the box's. The Mamba2 state is flat per
sequence, so it does not scale with context — but it is ~7× a full 32k-token KV, which is
why folding it into a per-token figure over-charged short prompts by 3.7× before PR #10.

### The deploy path had no nemotron in it (fixed)

Worth logging as a coverage gap of a different kind: every measured nemotron number above
was produced with a locally built binary against a locally converted container, and
nobody checked the *documented* path. `docker/run-dgx.sh -m` knew only `m2.7 | m3 | glm`,
and `scripts/models.toml` carried no `hf_repo` for nemotron **or** m3 — so the registry
that calls itself the single source of truth could not re-materialize half the models it
lists. The fastest model in the matrix was unreachable by the command the README tells
people to run. Verified after the fix: `-m nemotron` resolves to
`nvidia/NVIDIA-Nemotron-3-Super-120B-A12B-NVFP4`, and `coli probe` classifies that
checkpoint `nvfp4`, so the entrypoint's download → convert → serve chain applies to it
unchanged. **When adding a model, run its documented deploy command, not just its bench.**

### The documented command, run from nothing (2026-07-26)

`docker/run-dgx.sh -p 8099 -m nemotron` on gx10-42b2, empty HF cache, **no token**:

| stage | measured |
|---|---|
| image build | ~10 min (`docker build -f docker/Dockerfile`) |
| fetch checkpoint | ~11 min, 36 files, 69 GB |
| convert | **210 s**, 17 shards, 42 267 tensors quantized → 73.3 GB |
| serve | up on :8099, `context length: 32768 (model max 262144)` |
| answer | `"The capital of France is Paris."`, `finish_reason: "stop"`, 7 tokens |

So the whole path is **~25 min from `git clone` to a served answer**, and the README's
"one-time 30–90 min" was a GLM figure applied to every model. Three things this caught
that a bench never would:

- **No token is required.** All four checkpoints report `gated: false` on the Hub; the
  full nemotron download ran with `HF_TOKEN` empty. The README had been telling people
  to get a token before starting.
- **The cache-mount warning fired when the cache *was* mounted.** `run-dgx.sh` always
  mounts `~/.cache/huggingface`, but the entrypoint printed "mount the host HF cache to
  persist it" unconditionally — directly above a 69 GB download, which is the worst
  possible place for a false alarm. Now gated on an actual `mountpoint` check.
- **Docker had never served this model at all.** GPU passthrough, the ChatML template,
  and the stop id all work through the container; none of it had been exercised.

⚠️ **Method note:** the first attempt to wait for the image keyed on `docker image
inspect speedycolibri:latest` succeeding — which returned a *stale tag* while the build
was still compiling. Same shape as trap 8: the gate passed, nothing had been built. Wait
on the build **process**, not on an artifact whose name can pre-exist.

## Recurring traps (cost us real time; check these when adding a model)

1. **An unhandled arch silently inherits a GLM-shaped default.** Four instances on Nemotron
   alone: the bench prompt (synthetic ids hit its stop token, so suites measured *nothing*
   while the token gate still "passed"), the chat template (GLM markers to a ChatML model),
   batched decode (fell to the MLA branch → panic), and KV sizing (`is_gqa()` false while
   `enable_gqa()` is called). Grep every per-arch `match` and `_ =>` fallthrough.
2. **`gpu_eligible` omissions** — a resident weight left off the list in `lib.rs` silently takes
   the single-threaded CPU matmul. Cost 84% of an M3 prefill, 94% of Nemotron's mamba, 62% of
   its MoE phase. Tell: a profile phase whose total far exceeds the sum of its GPU sub-timers.
3. **A passing gate is not proof of a measurement.** Check step/token counts are what you expect.
4. **Never reuse a number from a different regime.** The single most repeated error here.
5. **Re-profile after any change to a shared path, before picking the next target.** PRs #13/#15
   touched `coli_cuda_matmul` and silently re-ranked everything downstream: the shared expert
   went 20.6% → 1.5% and mamba proj 28.7% → 20.7% without either being touched directly. A
   "next lever" chosen off a pre-change profile can be chasing a slice that no longer exists.
6. **A kernel fast path gated on `S == 1` is a trap for every non-decode-shaped caller.**
   Three separate S==1 gates (tile bypass, `i8a16_gemv`, grouped relu²) meant MTP's 2-token
   verify forward reverted to the pre-optimization kernel with no warning — a 1.79× penalty
   applied silently. When gating an optimization on shape, grep for every caller that could
   hit an adjacent shape.
7. **Speculation is unsound on any in-place recurrent mixer.** The accept-longest-prefix
   trick assumes rejected work can be discarded by overwriting; that holds for attention KV
   and NOT for Mamba conv/SSM state. Check this before enabling MTP on a new architecture.
8. **Hash the token line, not the whole stdout.** A first attempt at the MTP A/B md5'd all of
   `coli gen`'s output, which includes a VRAM-summary line that varies run to run — it would
   have manufactured a "divergence" between two identical arms. Gate on `generated (N tok):`.
9. **`tok/s` in the decode timer is per FORWARD, not per token.** With speculation a forward
   emits >1 token, so the printed figure understates throughput. Convert via `tok/forward`
   before comparing to a non-speculative arm — raw, DRAFT=2 looks 2.6× slower than it is.
10. **Interleave A/B arms (0 1 0 1), never all-of-A then all-of-B.** A cold post-rebuild run landed
   entirely on one arm and turned a true 1.12× into an apparent 1.5×. Discard a warmup too.
   ⚠️ **Interleaving is not sufficient if the order *within* each pair is fixed** — the arm that
   runs second is systematically penalized, by ~3% on a streaming-bound model. Demonstrated on
   GLM with identical binaries and a passing token gate: `(base, main)` per pair gives 0.968×,
   `(main, base)` gives **1.037×**, a mirror image. The same shape produced an apparent 2.2%
   m3 regression that vanished on reversal. **Alternate the order too (ABBA, not ABAB)**, or
   report the pooled median of both orderings.
11. **The synthetic bench prompt (`seq 100 611`) is a defect, not a fixture.** That id range
   drives models into degenerate repetition, where argmax near-ties flip on harmless FP
   reassociation — so a **token-identity gate fires on benign noise**. It has produced two
   bogus results: it hit Nemotron's stop token (suites measured *nothing* while the gate
   "passed"), and it manufactured a GLM correctness FAIL that vanished entirely on a real
   natural-language prompt (all four runs then byte-identical). **Use a real NL prompt for
   any correctness gate.** **FIXED 2026-07-25:** `scripts/models.toml` no longer contains
   `100..611`. glm-5.2, minimax-m3 and minimax-m2.7 now carry 512 ids of real English
   prose, tokenized with each model's *own* tokenizer (round-trip verified) and checked to
   generate 6/6 tokens without hitting a stop id. Repeating one short paragraph to reach
   length is the same trap in milder form — the continuation degenerates into a copy task,
   which showed up as m3 and m2.7 emitting byte-identical continuations from a 4×-repeated
   passage — so the committed prompts use non-repeating prose (~283 distinct ids of 512).
   ⚠️ `nemotron-3-super` still carries a repeated passage (87 distinct of 512). It is real
   text and sustains generation, so it is soundly usable, but it was left alone on purpose:
   every nemotron figure in this file was measured against that exact prompt. Regenerate it
   only together with a re-measure.
12. **Median-of-3 is not enough on this box.** Roughly a quarter of decode runs land well
   below the mode, and it hits *both* arms — so P(≥2 of 3 low) ≈ **16% per arm**. This
   manufactured a fake **3.79×** on M2.7 that 8 reps dissolved to the real 1.19×. Use **≥8
   reps**, and never call an arm "unstable" without checking whether the other is equally
   scattered.
13. **Do not diagnose a perf gap with instrumentation heavier than the gap.** `COLI_PROFILE=1`
   dropped both M2.7 arms from ~4.4 to ~0.4–0.9 tok/s and made them *cross*. `COLI_TIMING=1`
   is cheap; reach for it first.
14. **`git fetch` before any claim about what is or isn't merged.** `git log main..HEAD`,
    `git log main`, and `git ls-tree main` all read the *local* ref, which in a worktree-heavy
    repo can sit months behind. A stale `main` produced a confident "these three PRs were
    never merged" (they were all on `origin/main`), two duplicated commits, and a
    reimplementation of #11 in a weaker form that was nearly merged as a regression. Compare
    against `origin/main`, or use `git merge-base --is-ancestor`. Related: because this repo
    **squash-merges**, a branch can show dozens of "unmerged" commits whose content is fully
    upstream — check the *diff*, not the commit count.
15. **In tensor-parallel FFN, gather is byte-exact and all-reduce is not.** Splitting only
    `gate`/`up` by intermediate rows and **gathering** the slices, then running `down` *once*
    on the full intermediate, reproduces single-node output bit-for-bit — each intermediate
    row is an independent dot, and `down` still sees the identical input in the identical
    accumulation order. The tempting all-reduce form (every node runs the full `down` over a
    masked intermediate, partials summed) reorders `down`'s f32 reduction and lands ~1 ULP
    off. That is enough to fail a token-identity gate on an argmax near-tie, so the obvious
    design silently breaks the correctness harness. Measured and asserted on branch
    `multispark` @ `d28bede` (`ffn_intermediate_slice` + `dense_tp_gather_is_byte_identical`),
    **not merged** — the primitive itself is expected to be a wash on a bytes-bound decode,
    so only the finding is recorded here. Cherry-pick that commit if Obj C wants the code.
16. **An async H2D/D2H to a *pageable* host buffer is not async — it is a trickle.** It
    degenerates into a synchronous copy through a small internal bounce buffer. The GPU
    Mamba scan measured **~840 MB of D2H at ~146 MB/s**: kernel 255 ms for all 40 layers,
    download+sync **5 761 ms**. `reserve_pinned` + a plain host memcpy is the whole 9.4×.
    Four attempts were spent optimizing the 255 ms first — shared-memory staging, padding
    away a *genuine* 32-way bank conflict, 128× more threads — each moving nothing.
    **Instrument the transfer before touching the kernel**; CUDA events around the copy
    take minutes and would have skipped all four.
17. **A profile field computed as a remainder is not a measurement.** "conv+norm" was
    `MAMBA_US − scan − proj` and silently absorbed **1 465 ms of allocation churn**, which
    mis-scoped the conv/norm GPU port at 20% when the kernels are ~10%. Any field derived
    by subtraction accumulates everything you forgot to time. Split it into direct timers
    before ranking work against it.
18. **A timer that stops before a device sync attributes its wait to whoever syncs next —
    and a slow CPU phase can be *hiding* GPU work.** After the mamba scan moved to the GPU,
    `attn` appeared to jump 82 → 789 ms with no change to the attention path; it had
    inherited the sync mamba used to absorb, and the old 82 ms was fiction. The full-scale
    version of the same effect cost **79% of a shipped optimization**: the scan fell 7 899 →
    616 ms warm while the untouched shared expert rose 377 → 6 127 ms, turning a predicted
    1.54× into a measured **1.066×**. **A phase's measured time is only an upper bound on
    what removing it saves.** Before ranking work by phase cost, ask what that phase is
    overlapping with; before believing a regression in a phase you did not touch, suspect
    attribution.
19. **Allocation churn is a recurring, invisible tax on this codebase.** Per-layer `Vec`
    allocations cost **8.5% of a warm prefill** in the Mamba mixer (~128 MB zeroed per
    layer × 40), and separately contaminated the grouped-expert A/B enough to move it from
    1.05× to 1.013× slower — i.e. most of a "regression" under test was the measurement
    path's own overhead. Both fixed the same way: a `thread_local` grow-never-shrink
    scratch struct, with every consumer slicing to the *current* length because a stale
    tail is always live. Check for this pattern before benchmarking any per-layer path.
