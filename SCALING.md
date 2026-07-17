# Scaling Plan — Expert-Parallel + GDS (SpeedyColibri)

Coordination doc for the multi-node scaling work. Tracks the phased plan, the
correctness gates, and the parallel worktrees. Started 2026-07-17.

## Thesis

A model whose **active** params fit but whose **total** doesn't (GLM-5.2, and
K3-class) is a streaming problem. Scale it two ways, over ConnectX-7 (200 GbE RoCE):

1. **Pool RAM across nodes** — shard experts; each node keeps its slice resident, so
   the aggregate resident pool grows with node count.
2. **Scale aggregate SSD read bandwidth** — each node streams *its own* shard from
   *its own* NVMe, so read bandwidth scales ~linearly with nodes (the single-drive
   ~10.5 GB/s ceiling stops being the cap).

GDS (cuFile) is the per-node refinement: DMA storage→memory to **free the CPU cores**
the read path currently pegs, so they're available for the RDMA transport.

Data-flow rule (non-negotiable): **move activations to experts, not weights.** A
remote expert gets the tiny activation rows (~tens of KB); the owner computes the FFN
locally (from its own SSD/RAM) and returns the result. The 18 MB weights never cross
the wire — that's what makes the wire traffic cheap and the SSD scaling real.

## Baseline (single node, record before changing anything)

| Metric | Value |
|---|---|
| Bulk preload (chunked reads) | ~8.75 GB/s (83% of the ~10.5 GB/s drive ceiling) |
| NVMe saturation | ~10 threads (QD10); more threads / 2-per-core do NOT help |
| Decode | load-bound; expert loading dominates the step |
| Zero-copy experts (unified mem) | GPU reads RAM in place; no VRAM bounce |
| Correctness oracle | `coli gen <snap> 100 200 300` → `[198, 82, 198, 82, 200, 82, 198, 82]` |

## Already built

- ✅ Single-node streaming: zero-copy experts, coalesced + chunked parallel reads.
- ✅ `colibri-cluster`: `ExpertSharding` (owner/is_local), `Transport` trait,
  `ExpertRequest`/`ExpertResponse`, `LocalTransport`.
- ✅ Peer discovery (branch `multispark`, not merged): UDP beacon :48757 + ConnectX
  MAC-OUI classification; `coli cluster` prints peers/ranks/ports. Verified 42b2↔5a4f.
- ❌ RDMA/TCP transport (stub only). ❌ GDS.

## THE universal gate

> Every distributed or transport change MUST generate tokens **identical to
> single-node**: `[198, 82, 198, 82, 200, 82, 198, 82]`. If tokens differ, it's wrong —
> stop and fix before measuring performance. Machine is shared/noisy: trust only
> same-session relative numbers.

## Execution plan (waves — worktrees run in parallel)

### Wave 1 — start both now

- [ ] **RDMA-A · Expert-parallel over TCP** — branch off `multispark`. Critical path,
      highest value/risk. (Prompt: "Expert-parallel data-flow over TCP".)
      - Gate: 2-node run (42b2+5a4f, experts `eid%2`) == single-node tokens.
      - Record: per-node cache misses (~½ each → ~2× aggregate SSD BW), tok/s vs 1-node,
        wire bytes/token.
- [ ] **GDS-1 · Feasibility probe** — separate worktree, ~1 h spike, no engine changes.
      (Prompt: "GDS Feasibility probe".)
      - Gate/output: is real GDS available (not cuFile compat mode)? Does it offload CPU
        vs chunked pread? Clear **GO / NO-GO**. NO-GO cancels Wave-2/3 GDS tasks.

### Wave 2 — after Wave-1 gates are green

- [ ] **RDMA-B · RDMA verbs transport** — after RDMA-A is correct. Swaps the wire behind
      the `Transport` trait. (Prompt: "RDMA/RoCE verbs transport".)
      - Sub-gate FIRST: `coli cluster --rdma-ping` round-trips 42b2↔5a4f before wiring
        the engine (RoCE bring-up: GID index, RoCEv2, PFC/MTU — budget time here).
      - Gate: RDMA tokens == TCP tokens == single-node. A/B vs TCP: round-trip latency,
        tok/s, driver CPU%.
- [ ] **GDS-2 · cuFile FFI + registered read** — only if GDS-1 = GO. Independent crate,
      overlaps RDMA-B. (Prompt: "cuFile FFI + registered-buffer read".)
      - Gate: GDS read byte-identical to `read_raw_shared`; `--features gds` off = build
        + 87 tests unchanged.

### Wave 3 — last (measured in the multi-node context on purpose)

- [ ] **GDS-3 · Wire GDS into expert load + benchmark** — after RDMA-A exists so the
      CPU-offload value is measurable where it matters. (Prompt: "Wire GDS + benchmark".)
      - Gate: tokens match (unconstrained + `COLI_RAM_GB=25`). A/B `COLI_GDS=1` vs `0`:
        preload GB/s, decode load time, **driver CPU% during load** (the real win).

## Ordering rationale (don't reshuffle without reading this)

- **Scaling foundation before its optimizations.** RDMA-A alone delivers the thesis
  (RAM pool + SSD BW scaling); TCP-over-RoCE is enough because only tiny activations
  cross the wire. GDS and verbs are refinements on top.
- **Cheap gates first.** GDS-1 is a 1-hour probe that can cancel a whole track.
- **TCP before verbs.** 90% of the risk is the all-to-all/overlap/correctness logic,
  not the wire; a known-good TCP baseline de-risks the verbs footguns.
- **GDS measured last.** Single-node it's drive-capped (misleading "marginal"); its
  value is CPU offload, visible only once multi-node.

## Coordination

- One worktree + branch per task; feature-gate new deps (`gds`, `rdma`) so default
  builds are unchanged.
- Keep tracks in disjoint crates to avoid collisions: RDMA in `colibri-cluster`
  (+ `moe.rs` integration), GDS in `colibri-safetensors` (+ `colibri-backend` FFI).
- Parallel efforts in flight: `main` (fp8), `multispark` (discovery). Rebase before
  merging; land `multispark` → `main` before/with RDMA-A.

## Hardware / access

- Node A: `ssh dgx1@gx10-42b2` (rank 0 / driver). ConnectX IP `192.168.100.11`.
  CUDA 13.0 at `/usr/local/cuda`; rustup installed; source at `~/SpeedyColibri`.
- Node B: `5a4f` — **docker-only, no rustup**: copy the aarch64 `coli` binary from 42b2.
- Model: `~/.cache/huggingface/hub/models--mateogrgic--GLM-5.2-colibri-int4-with-int8-mtp/snapshots/*/`
- Both: GB10 Grace-Blackwell, ~121 GB unified LPDDR5X. Cap `COLI_RAM_GB`/`COLI_VRAM_GB`
  to avoid systemd-oomd (SIGTERM 143).

## Decision log / open questions

- [ ] GDS GO/NO-GO (from GDS-1) — record the CPU + GB/s numbers and the decision.
- [ ] Transport lib for RDMA-B: raw libibverbs+rdma_cm FFI (dependency-light) vs UCX
      (simpler, adds dep). Default to raw verbs unless bring-up proves too costly.
- [ ] EP topology: rank-0-drives + peers-as-expert-servers (v1) — revisit symmetric
      driving later if it helps.
- [ ] GDS buffer ownership: copy-out to `Arc<[u8]>` vs a `Bytes::Registered` pooled
      variant (decide in GDS-2 from the probe's registration-cost numbers).

## Backlog — orthogonal speedups (not in the wave plan)

Single-node optimizations that attack the same long-context wall as multi-node, but
independently. Kept here so they aren't lost; not gated on the RDMA/GDS waves.

- [ ] **Sliding-window attention (SWA)** — bound each token's attention to the last W
      positions so prefill attention drops from O(n²) to O(n·W). Goal: **more speed at
      long context with minimal-to-no quality loss.**
      - *Motivation (measured 2026-07-17, single node, 8/4):* prefill dominates long
        context — 512 in → 202 s, 2048 in → 618 s — and decode degrades with context
        (0.58 → 0.45 tok/s from 512 → 2048) because each step attends over the whole
        KV. Both are the attention span growing; SWA caps it.
      - *Quality risk is real:* GLM-5.2's checkpoint config has **no** window param —
        it was trained with **full** attention, so SWA is an added approximation, not a
        native mode. It may hurt quality; this is why it's a *test*, not a change.
      - *Quality gate (hard):* `coli ppl` must stay at the 8/4 baseline — perplexity
        ≈ 6.189, top-1 ≈ 57.9%. Sweep W (e.g. 1k/2k/4k/8k) and take the smallest W that
        does not regress ppl. If no W both speeds up and holds quality → negative
        result, record it and drop.
      - *Measure:* prefill (TTFT) and decode tok/s at 2k/8k/32k input, SWA on vs off,
        plus ppl on vs off. Same harness as the long-context record
        (`scripts` + `coli ppl`). Report speedup **and** ppl delta together — a speedup
        that moved ppl is not a win.
      - *Also worth a look:* attention-sink / StreamingLLM variants (keep the first few
        tokens + a sliding window) often hold quality far better than a plain window on
        a full-attention-trained model — cheap to try once the window path exists.
