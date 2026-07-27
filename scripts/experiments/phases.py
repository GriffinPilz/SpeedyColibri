#!/usr/bin/env python3
"""Per-request phase medians for every `coli serve` log given on the command line.

Same arithmetic as scripts/experiments/phase_diff.py — [profile] counters are CUMULATIVE
for the life of the process, so one request costs the difference between consecutive
prints — but prints one row per log instead of a two-arm delta, because #98's devcopy
matrix has four arms.
"""
import re
import statistics
import sys
from pathlib import Path

PATS = {
    "prefill_ms":  r"prefill ([\d.]+) ms",
    "attn":        r"attn ([\d.]+) ms",
    "mamba":       r"mamba ([\d.]+) ms",
    "moe":         r"moe ([\d.]+) ms",
    "expert-load": r"expert-load ([\d.]+) ms",
    "gather":      r"gather ([\d.]+) ms",
    "gpu-ffn":     r"gpu-ffn\(\+sync\) ([\d.]+) ms",
    "scatter":     r"scatter ([\d.]+) ms",
    "shared":      r"shared ([\d.]+) ms",
    "scan":        r"scan ([\d.]+) ms",
}


def snapshots(path):
    cur, out = {}, []
    for line in Path(path).read_text(errors="replace").splitlines():
        if "[profile]" not in line:
            continue
        for name, pat in PATS.items():
            m = re.search(pat, line)
            if m:
                cur[name] = float(m.group(1))
        if "mamba breakdown" in line:
            out.append(cur)
            cur = dict(cur)
    return out


def block(path):
    snaps = snapshots(path)
    rows = [{k: b.get(k, 0) - a.get(k, 0) for k in PATS} for a, b in zip(snaps, snaps[1:])]
    rows = rows[1:]  # drop the warm-up's own diff
    if not rows:
        return None, 0
    return {k: statistics.median(r[k] for r in rows) for k in PATS}, len(rows)


keys = [k for k in PATS if k != "prefill_ms"]
print(f"{'log':<20}{'n':>3}" + "".join(f"{k:>13}" for k in keys))
for p in sys.argv[1:]:
    med, n = block(p)
    tag = Path(p).stem.replace("server-", "")
    if not med:
        print(f"{tag:<20}{n:>3}  (no warm requests)")
        continue
    print(f"{tag:<20}{n:>3}" + "".join(f"{med[k]:13.0f}" for k in keys))
