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

GDS (cuFile) was to be the per-node refinement: DMA storage→memory to free the CPU
cores the read path pegs, for the RDMA transport. **GDS-1 came back NO-GO** — GB10's
unified memory makes it inapplicable (see Wave 1). The *goal* — offload read-path CPU
— stays; pursue it via async I/O (io_uring), not GDS.

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

- [x] **RDMA-A · Expert-parallel over TCP** — CORRECTNESS GATE PASSED (2026-07-17).
      The whole path was already coded (TcpTransport, moe_sharded, serve wiring); the
      only gap was that `set_cluster` lived only in `serve`, so `coli gen` — the token
      oracle — couldn't run multi-node. Fixed in `aec9c52`.
      - ✅ Gate: 42b2 (rank 0, experts 0-127) + 5a4f (rank 1, experts 128-255),
        contiguous-block sharding (fingerprint `0xa8ec43a8dc6fb35c` agreed on both),
        `coli gen <int4-snap> 100 200 300` == single-node, all 16 tokens:
        `[198,82,198,82,200,82,198,82,198,82,198,82,198,82,198,82]`. Real wire
        (TcpTransport over the 192.168.100.0/24 RoCE fabric), not LocalTransport.
      - [ ] **Still to record** (performance, not correctness): tok/s 2-node vs 1-node,
        per-node cache misses (expect ~½ each → ~2× aggregate SSD BW), wire bytes/token.
      - Note: ran on the int4 `mateogrgic` model (both nodes have it; 5a4f can't fit the
        356 GB 8/4 container in its 114 GB free). Correctness is model-independent.
- [x] **GDS-1 · Feasibility probe** — **NO-GO** (2026-07-17). Cancels GDS-2 + GDS-3.
      `gdscheck -p` on the GB10 Spark (GDS 1.15.1.6, libcufile 2.12, aarch64):
      every storage backend `Unsupported` (NVMe, NVMe P2PDMA, NVMeOF, all FS),
      `use_compat_mode : true`, nvidia-fs module not loaded (loading it needs root we
      don't have on the shared box). So cuFile would run in compat mode = POSIX read +
      CPU bounce = zero DMA offload, the exact thing this gate rejects.
      - **Root cause is architectural, not a missing module:** GB10 is unified coherent
        memory. The engine already reads NVMe→host RAM and the GPU reads that RAM
        *zero-copy* (no cudaMemcpy). GDS's value is DMA NVMe→**VRAM** to skip the host
        bounce on a *discrete* GPU; there is no separate VRAM here to bounce to, so
        `NVMe P2PDMA: Unsupported` is the platform correctly saying GDS's model doesn't
        apply. Even with root + nvidia-fs it would not help the way it does on PCIe.
      - **The underlying goal survives GDS** — free the CPU cores the read path pegs so
        they can drive the RDMA transport. Pursue via async I/O (io_uring) instead;
        tracked in the backlog, NOT as a GDS task.

### Wave 2 — after Wave-1 gates are green

- [ ] **RDMA-B · RDMA verbs transport** — after RDMA-A is correct. Swaps the wire behind
      the `Transport` trait. (Prompt: "RDMA/RoCE verbs transport".)
      - Sub-gate FIRST: `coli cluster --rdma-ping` round-trips 42b2↔5a4f before wiring
        the engine (RoCE bring-up: GID index, RoCEv2, PFC/MTU — budget time here).
      - Gate: RDMA tokens == TCP tokens == single-node. A/B vs TCP: round-trip latency,
        tok/s, driver CPU%.
- [~] **GDS-2 · cuFile FFI + registered read** — CANCELLED (GDS-1 = NO-GO). Independent crate,
      overlaps RDMA-B. (Prompt: "cuFile FFI + registered-buffer read".)
      - Gate: GDS read byte-identical to `read_raw_shared`; `--features gds` off = build
        + 87 tests unchanged.

### Wave 3 — last (measured in the multi-node context on purpose)

- [~] **GDS-3 · Wire GDS into expert load + benchmark** — CANCELLED (GDS-1 = NO-GO). Kept for context. Was: after RDMA-A exists so the
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

- [x] GDS GO/NO-GO (from GDS-1) — **NO-GO** (2026-07-17). No CPU/GB-s numbers taken:
      gdscheck reports compat-mode-only (every backend Unsupported, nvidia-fs unloaded,
      no root), so there was no real-GDS path to benchmark. Root cause is GB10's unified
      memory — the GPU already reads host RAM zero-copy, so GDS's NVMe->VRAM DMA has no
      VRAM bounce to eliminate. GDS-2/3 cancelled. Read-path CPU offload -> io_uring backlog.
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
