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
| `phase_diff.py` | Per-request phase costs from a server log. `COLI_PROFILE` counters are cumulative, so a request costs the difference between consecutive prints. |

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
