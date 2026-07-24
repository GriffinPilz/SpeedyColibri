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
(512 tokens), medians over ≥2 reps, token-identity gated.

## Coverage

| lever | glm-5.2 | minimax-m3 | minimax-m2.7 | nemotron-3-super |
|---|---|---|---|---|
| end-to-end token validation | PASS | PASS | PASS | PASS |
| prefill bench | PASS | PASS | PASS | PASS 33.1 s / 15.5 tok/s |
| decode bench | PASS | PASS ~1.9 | PASS | PASS 4.65 tok/s |
| serve bench | PASS | PASS 1.38 | PASS | PASS 3.60 tok/s (12/12) |
| batch / genbatch | PASS 1.34× @B64 | **NEG** monotonic loss | TODO | **N/A** hybrid — guarded, PR #9 |
| chat template in serve | PASS | PASS (GLM-style) | PASS (M2 format) | PASS ChatML, PR #8 |
| adaptive expert-cache RAM | PASS | PASS | PASS | PASS near-fit, ~59/121 GB |
| KV reservation accounting | PASS | PASS | PASS | PASS after PR #10 (was 3.7× over) |
| grow-on-demand KV | PASS | PASS | PASS | PASS 3301-tok prompt, no OOM |
| **prefill scratch reservation** | TODO | TODO | TODO | **BROKEN** 210 vs 24 KB/tok |
| **MTP / speculative decode** | PASS (break-even) | N/A no head | N/A no head in quant | **TODO — head is DROPPED at convert** |
| COLI_PREFETCH_AHEAD | PASS 1.58× | PASS 1.26× | TODO | TODO |
| COLI_RAM_GB sweep | PASS (flat) | TODO | TODO | TODO |
| hot-expert autopin | **NEG** ~10% loss | TODO | TODO | TODO |
| sliding-window attention | **NEG** lose-lose | N/A | N/A | TODO |
| 2-node expert-parallel | PASS 1.38× | TODO | TODO | TODO |
| TP attention (2-node) | PASS | TODO | TODO | TODO |
| long context (>32k) | TODO | TODO | TODO | PASS 1M w/ COLI_ALLOW_LONG_CTX |

## Per-model notes

### nemotron-3-super — hybrid Mamba2 + GQA + latent-MoE
88 layers = 40 Mamba2 + 40 latent-MoE + 8 GQA (NoPE, no qk-norm). 512 experts top-22,
gateless ReLU² in a 1024-wide latent space. NVFP4 routed experts.

- **prefill 33.1 s / 15.5 tok/s**, **decode 4.65 tok/s**, **serve 3.60 tok/s** (2026-07-24).
- **Experts are only ~59 GB against 121 GB of RAM (172% coverage) — they FIT.** This model is
  *not* in the disk-streaming regime GLM is in, so GLM-derived "bytes-bound floor" reasoning
  does not transfer. Any claim that its decode is at a fundamental floor needs its own profile.
- **BROKEN — prefill scratch is unreserved.** Measured peak-RSS slope **210 KB/token**
  (2 reps: 214.5 / 206.2) against the **24 KB/token** serve reserves. The Mamba mixer allocates
  per-layer buffers that scale with prompt length (`proj` 18560 + `gate` 8192 + `hbc` 10240 +
  `h` 8192 + `b`/`c` + `aug`/`conv_aug`), freed per layer so peak ≈ one layer live. A prompt can
  therefore be admitted as "fits" and then OOM in the mixer. ~6.1 GB unaccounted at 32k ctx.
- **MTP head exists but we throw it away.** Source ships `mtp.layers.0` (attention +
  `eh_proj`/`enorm`/`hnorm`) and `mtp.layers.1` (latent-MoE, 512 experts), with
  `num_nextn_predict_layers: 1`. `convert.rs:132` drops every `mtp*`/`nextn` tensor, so the
  container has no head and `DRAFT=N` cannot engage. **This is the same bug already fixed once
  for GLM** (keep-the-head-in-convert). Recoverable — source weights are on disk.
- Context: 262,144 by default, **1,048,576** with `COLI_ALLOW_LONG_CTX=1` (NoPE ⇒ the config's
  `max_position_embeddings` is advisory, not architectural). 24 KB/token makes 1M cost ~24 GiB.
  ⚠️ Interacts with the prefill-scratch bug above: a genuinely long *prompt* is exactly the case
  the reservation under-counts.

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
