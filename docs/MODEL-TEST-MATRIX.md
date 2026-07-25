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
| prefill bench | PASS | PASS | PASS | PASS 33.1 s / 15.5 tok/s |
| decode bench | PASS | PASS 2.07 | PASS | PASS **8.3 tok/s** |
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

- **prefill ~38 s**, **decode 7.11 tok/s** (after PR #13), **serve 3.60 tok/s** (measured pre-#13).
- ⚠️ An earlier prefill figure of **33.1 s** does not reproduce — runs both *before* and *after*
  PR #13 give ~38 s on the same box. NOT attributable to #13 (pre-change runs already showed
  38.4 s). Treat 33.1 as measured in an unidentified state; re-run
  `bench.sh nemotron-3-super prefill` for a current number.
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

⚠️ **Corollary that kills the planned follow-on work:** the S=2 verify cliff that blocks
Nemotron MTP is this same number read backwards, so **small-S dispatch is a nemotron-shaped
lever, worth ~8–19% elsewhere.** Do not build it expecting a general win.

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

⚠️ **Unexplained: on m2.7 the prefill saving is 2.2× the expert-load saving.** Prefill fell
13.1 s while expert-load fell only 6.0 s. If the lever merely *overlapped* load with
compute those two should be roughly equal. The untested hypothesis is that batching the
reads deepens the I/O queue and makes the reads genuinely faster rather than merely
hidden — which would fit the separate finding that read QD sits at ~8–10 against a drive
that wants ≥32. **Do not cite a mechanism here until someone measures it**; if reads are
actually getting faster, there is more available than overlap alone.

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
