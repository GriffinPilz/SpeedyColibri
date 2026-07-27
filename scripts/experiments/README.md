# One-off measurement harnesses

Scripts written to answer a specific question, kept because the *question* recurs and
because each one encodes a trap that already cost time once. They are not part of the
model harness (`scripts/bench.sh`) and are not run by CI. Paths assume gx10-42b2.

| script | question it answered |
|---|---|
| `warm_prefill.py` | What is warm prefill through `coli serve`? Discards request 1 (the in-process expert cache costs ~9.9 s on a one-shot `coli gen`), reports a median and the generated token so callers can gate on identity. |
| `scan_ab.sh` | Is the GPU Mamba scan worth anything? Same binary, `COLI_MAMBA_CPU` switches the arm. **1.43×** (#97). |
| `bin_ab.sh` | Is a phase difference between two binaries real time or re-attribution? Holds the scan constant on the CPU path so only the rest of the diff moves. |
| `bisect_shared.sh` | Which commit introduced it? Builds and measures each commit in a range on one phase counter. Found PR #28 (#97 → #98). |
| `phase_diff.py` | Per-request phase costs from a server log (two-arm delta). `COLI_PROFILE` counters are cumulative, so a request costs the difference between consecutive prints. |
| `phases.py` | Same cumulative-diff arithmetic, one row per log — for the #98 four-arm `COLI_RAM_GB`×`COLI_FFN_DEVCOPY` matrix where a two-arm delta doesn't fit. |
| `relu2_evt.sh` | Is the shared-expert cost GPU or CPU? Runs `COLI_RELU2_EVT=1`, which splits the fused NVFP4 relu² kernel's device time into stage-H2D vs kernel, bucketed by input width `D`. Proved the routed (`D=1024`) device time is **identical** across budgets and the shared expert (`D=4096`) never reaches that kernel at all — so the #98 regression is entirely CPU-side. |
| `fix_ab.sh` | Does pooling the shared-expert scratch remove the regression? `COLI_SHARED_SCRATCH` switches the arm on one binary; ABBA-mirrored at 65 GB (the pressure regime) plus a 20 GB no-pressure control. **1.32×**, token-identical, neutral without pressure (#98). |
| `decode_ab.sh` | Does the pooled scratch (which also runs at S=1) stay token-identical and no slower in decode? Greedy 64-token gens, `COLI_SHARED_SCRATCH` on vs off, byte-compares the generated text. |

## What these encode

- **Assert on a phase that must move, not on the env var you set.** The first `scan_ab.sh`
  run compared `target/release/coli` with itself — it was a stale pre-#91 build — so the arm
  flag switched nothing. Wall clock looked fine and the token gate passed; only `scan ≈ 8 s`
  in *both* arms exposed it. The script now fails loudly on that.
- **Kill by argv signature, not by binary name.** `pkill -f "coli serve"` does not match
  `/tmp/coli_base serve …`, so the old server kept the port and three of four blocks died on
  bind while the script reported success.
- **A server holding ~59 GB does not exit in 3 s.** Wait for the port, then escalate.
- **Mirror the block order** (A,B,B,A). A fixed within-pair order penalises whichever arm
  runs second by ~3% on this box — enough to invent or hide a small effect. See trap 10 in
  [`docs/MODEL-TEST-MATRIX.md`](../../docs/MODEL-TEST-MATRIX.md).
- **A phase's Rust timer and its GPU timer can measure different things.** `SHARED_US`
  wraps the shared expert's buffer allocation *and* its sync wait; `GPUFFN_US` times only
  the kernel call, so the routed experts' `xg`/`hh` allocations fall in untimed gaps. That
  asymmetry made the shared expert look like a GPU problem when the GPU was flat (RELU2_EVT).
  When a phase balloons, check whether its timer spans an allocation the "flat" phase's
  timer skips — before blaming the kernel.
