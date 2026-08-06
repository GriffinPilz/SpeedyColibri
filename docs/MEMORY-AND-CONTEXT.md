# RAM residency, and context & output length

<!-- Extracted from README.md 2026-08-06 to keep the README to the six things a
reader needs first: speeds, models, how to run them, where they come from, what was
changed, and quality. Nothing here was trimmed in the move. -->

## RAM residency (adaptive by default)

The expert cache **fills RAM and defends it** — no flag, for every model. A background
monitor polls `MemAvailable` every 100 ms and evicts LRU experts the moment free memory
approaches a hard floor (~3 GB), *whatever* consumed it — more experts, a longer KV cache,
the GPU's own working set on GB10's unified pool. A cache that gives memory back under
pressure **cannot OOM**, which is what lets a fill-RAM policy point at a model of any size.
The two figures below are a **2026-07-23 A/B of this policy** — the *ratio* is the result; for
current absolute throughput see [How fast is it?](#how-fast-is-it), where these models now serve
faster than the absolutes quoted here:

- **Near-fit** (experts ≈ RAM — e.g. MiniMax-M2.7, ~122 GB on a 121 GB Spark): fill RAM and
  hold the whole working set resident, dropping the page-cache double-copy (`fadvise`) so
  `MemAvailable` is honest. Measured **1.94×** over the old static default (median 4.83 vs
  2.49 tok/s, diverse-prompt serving) — the working set stays resident instead of streaming.
  **Near-fit went through a period of looking unstable, and that was a different bug.** M2.7's
  throughput swung ~8× run to run, which got blamed on the residency policy, on pack thread
  count and on the buffer-pool cap before the actual cause was found: the *mmap* gate also
  fired at 80% coverage, so spans were served from a page cache the heap had already taken,
  and every non-resident touch became a major fault inside the pack's memcpy. The two
  thresholds are now separate — `NEARFIT_COVERAGE_PCT` 80, `MMAP_MIN_COVERAGE_PCT` 100 — and
  the full measurement record lives beside the constant in
  [`crates/coli/src/main.rs`](../crates/coli/src/main.rs). Two competing memory tiers gated on
  the same number is the shape to watch for.
- **≫ RAM** (experts larger than RAM — e.g. MiniMax-M3 ~193 GB, GLM-5.2 ~338 GB): fill RAM
  with as many experts as fit, keep the OS page cache as a reclaimable second tier, and let
  the monitor evict the LRU tail under pressure. Measured **1.22×** on M3 (2.05 vs 1.68 tok/s)
  — more resident experts, higher hit rate — and, crucially, **no crash**: the same box that
  OOM-died under a fixed 100 GB budget now fills to 121 GB used and holds `avail` at the
  floor for the whole run.

There is **no manual RAM budget knob**. `COLI_RAM_GB` was removed: a hand-set byte budget
cannot know the resident dense tier, the KV for the live context, or the GPU's share of the
unified pool, and it was the one path that skipped the headroom clamp — a 110 GB request on a
121 GiB Spark drove the serve process to 108.7 GiB RSS and into swap, at 0.06–0.24 tok/s.
The budget is derived from those three terms and re-derived under pressure.

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

| model | attention | KV / token (`coli capacity`) | max `COLI_CTX` on a 121 GB Spark |
|---|---|---|---|
| **Nemotron-3-Super** | hybrid — only **8 of 88** layers cache KV | **16 KB** + 166 MB/seq Mamba2 state | **262,144 — its full architectural max** (measured; 4.0 GiB KV) |
| **MiniMax-M3** | GQA, 4 kv-heads × 60 layers | 240 KB | **~450k** (the measured 402,690 clamp, scaled by the corrected per-token figure) |
| **GLM-5.2** | MLA (compressed), **mirrored on the device** | 351 KB | ~290k |
| **DeepSeek-V4-Flash** | latent ring + compressed rows | **13.4 KB** | **1,048,576 — its full architectural max** (measured; 13.4 GB KV) |
| **MiniMax-M2.7** | GQA, 8 kv-heads × 62 layers | 496 KB | ~200k |

**Five of these figures dropped on 2026-08-05, and no model got cheaper — the accounting
did.** V4's fell twice, for two different reasons: the phantom device shadow below, and then
a per-token raw charge that chunked prefill had already made a per-*sequence* one (206.9 →
99.4 → 13.4 KB). The second is why its ceiling reads 1,048,576 rather than a memory limit. `bytes_per_token` charged every architecture for a `qk_rope` row *and* a GB10 device
mirror of it. Only the MLA reader (`sync_device`'s sole caller — GLM-5.2 and Kimi-K3's
gated-MLA layers) actually keeps those, which is why those two are unchanged. The GQA models
rope the key in place and store it in `k_full`/`v_full`; DeepSeek-V4 writes its latent and
nothing else. The buffers behind the phantom terms were allocated but never written, so under
lazy commit they held zero resident bytes the whole time — the charge simply did not match
the allocation. `mirrors_kv_on_device` is the predicate, and
`gqa_writes_no_latent_or_roped_key_rows` poisons the rows and watches the poison survive, so
the claim is executable rather than argued.

**DeepSeek-V4's raw KV is a ring, so generation is nearly free and prompts are not.** V4
keeps a **128-token sliding window**: a query at `p` can only reach back 127 positions, and
everything older is reachable solely through the Compressor. The raw tier is therefore
sized to the widest span a single call reads — `max(window, prompt)` — and generated tokens
add no rows at all. Measured on the box, 43 layers at `kv_lora` 512 + `qk_rope` 64:

| 512-token prompt + 40k generated | raw rows retained | raw KV |
|---|---|---|
| linear (before) | 40,960 | 3.78 GiB |
| ring | 512 | **48.4 MiB** |

The engine prints the sizing once, so a run that is silently *not* ringed is visible rather
than merely indistinguishable: `[kv] raw ring 0 -> 512 rows x 43 layers (48.4 MB)`.

**Long prompts are chunked, but only when the KV actually demands it.** The ring alone was
sized to the prompt, because prefill is the one call that reads every row it wrote. Running
the prompt through in slices bounds it at `window + chunk - 1` rows however long the
context — but chunking is **not free**, and the cost is not the one a byte count suggests:

| 2048-token prompt | raw rows | raw KV | prefill |
|---|---|---|---|
| one call | 2048 | 193.5 MB | 119.6 s |
| chunked at 512 | 639 | 60.4 MB | 168.7 s — **1.41× slower** |

Smaller `S` amortises the routed-expert streaming over less work, the same effect recorded
for MoE pipelining. At 2048 tokens that trade is plainly bad: 133 MB saved out of a 107 GB
process, for 41% of prefill. So the rule is **do not chunk until the retained KV would
exceed `COLI_DSV4_KV_BUDGET_MB` (default 1024), then chunk exactly as coarsely as that
allows.** A 2048-token prompt runs in one call, byte-for-byte as before (measured: 123.9 s,
one ring sizing, 2048 rows). A 1M-token prompt — which would otherwise retain ~95 GiB of raw
rows and simply not fit — chunks at ~10.8k and retains 1 GiB.

**The chunking itself is bit-exact.** On the tiny V4 fixture every chunk of 2 or more
reproduces the single call *to the bit*, on the CPU build and the CUDA build alike — the
Compressor's cross-call carry, the ring's `pos % R` mapping and the causal span arithmetic
all compose exactly. (Chunk 1 differs on CUDA only, because `S == 1` dispatches the decode
kernel family; that arm compares prefill kernels against decode kernels, not two chunkings.)

What is *not* bit-exact on the real model is **kernel selection**: at 43 layers, `S = 128`
and `S = 512` cross tiling thresholds and pick different kernels, whose accumulation orders
differ. `COLI_DEBUG_ACT=1` on the V4 driver measures that rather than leaving it to
argument — at the last prefill position the divergence starts at 8.7e-6 after layer 0 and
**plateaus around 5e-3 by layer 14**, flat for the remaining 28 layers. A plateau is bounded
FP noise in an RMSNorm'd stack; a structural error would keep growing or jump. On a
512-token prompt it changed 1 of 16 generated tokens; at 2048 tokens, 4 of 4 matched. The
tiny fixture cannot see this — 3 layers at hidden 8 never reach the tiled kernels — which is
why the claim is split in two here rather than blurred into one tolerance.

**Together, the ring and the chunking make the per-token cost the compressed tier alone —
13.4 KB.** The raw rows are bounded by the KV budget however long the context runs, so they
are a per-*sequence* cost, not a per-token one. That is what puts V4's ceiling at its
architectural **1,048,576** rather than at a memory limit, and it makes V4 **cheaper per
token than Nemotron-3-Super** (13.4 KB vs 16 KB) despite caching KV on all 43 layers where
Nemotron caches on 8 of 88. Measured — `coli serve` was asked for 2,000,000 and clamped to
the model's own max, with RAM not binding:

```
[serve] context length: 1048576 tokens (model max 1048576; up to 13.4 GB KV)
```

**This was wrong here until 2026-08-05, and the error is worth naming.** The ring shipped
first, and while it was alone the raw rows really were sized to the prompt — so charging them
per token was correct, and this section said so. Chunked prefill removed that, and the
accounting was not revisited. A stale per-token charge is the only thing that had V4 listed at
a fraction of what it can serve.

One caveat remains, and it is a real one: **everything above 2,400 tokens on V4 is still
arithmetic.** The memory now permits 1M and the mechanisms are verified at small scale
(bit-exact chunking, the ring bounded under a forced budget), but the longest context actually
exercised end to end is ~2,400 tokens.

**The hybrid is in a different class here.** Nemotron caches KV on 8 attention layers
instead of all 88, so it costs **~11–22× less per token** than the GQA/MLA models — the
only one of the four whose ceiling is the *model's* limit rather than the box's, and it
reaches it with 6.2 GB of KV to spare. The Mamba2 state is a flat per-sequence cost, not
per-token, so it does not scale with context. Check any model's numbers without loading
it: `coli capacity <container>` reads `config.json` alone.

(architectural maxima are 262,144 / 1M / 1M / 196,608 respectively; only Nemotron and
M2.7 are reachable.) These are ceilings where the experts are nearly all evicted, so throughput there is low;
practical high-throughput context is lower. The default `COLI_CTX` stays **32,768** — a small KV
keeps the most RAM for resident experts and the lowest latency. `max_tokens` defaults to 128 and
is bounded only by the remaining context (no fixed cap).

**The KV grows on demand.** The cache is sized to the window in *address space* but committed
lazily (zero-on-demand pages) — its resident RAM tracks the tokens actually produced, not
`max_tokens`. So a request that sets a large `max_tokens` but stops early never pays for the
tail: only the prompt's KV is reserved up front, and the generation grows a token at a time
while the expert cache evicts against it. A prompt that can't fit is still rejected (507)
rather than OOM'd.

