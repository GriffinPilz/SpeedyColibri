#!/usr/bin/env python3
"""Per-request phase costs from a `coli serve` log, for the #97 scan A/B.

The [profile] counters are CUMULATIVE for the life of the process, so one request's cost
is the difference between consecutive prints. Request 1 is the warm-up and is dropped
along with its own diff (the first diff is req2-req1, which is the first warm request).

Prints the median warm-request cost per phase for each block, then the gpu-vs-cpu delta.
"""
import re
import statistics
import sys
from pathlib import Path

# phase name -> regex capturing its ms value, across the four [profile] lines
PATS = {
    "attn":        r"attn ([\d.]+) ms",
    "mamba":       r"mamba ([\d.]+) ms",
    "moe":         r"moe ([\d.]+) ms",
    "expert-load": r"expert-load ([\d.]+) ms",
    "router":      r"router ([\d.]+) ms",
    "gather":      r"gather ([\d.]+) ms",
    "gpu-ffn":     r"gpu-ffn\(\+sync\) ([\d.]+) ms",
    "scatter":     r"scatter ([\d.]+) ms",
    "shared":      r"shared ([\d.]+) ms",
    "scan":        r"scan ([\d.]+) ms",
    "in/out-proj": r"in/out-proj ([\d.]+) ms",
    "conv":        r"conv ([\d.]+) ms",
    "gated-norm":  r"gated-norm ([\d.]+) ms",
}


def snapshots(path):
    """Each [profile] block ends with the mamba line; emit one dict per completed block."""
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
            cur = dict(cur)  # counters are cumulative; carry forward
    return out


def per_request(path):
    snaps = snapshots(path)
    rows = []
    for a, b in zip(snaps, snaps[1:]):
        rows.append({k: b.get(k, 0) - a.get(k, 0) for k in PATS})
    return rows[1:]  # drop the warm-up's own diff


def block(path):
    rows = per_request(path)
    return {k: statistics.median(r[k] for r in rows) for k in PATS}, len(rows)


logs = sys.argv[1:]
res = {}
for p in logs:
    tag = Path(p).stem  # server-<idx>-<arm>
    med, n = block(p)
    res[tag] = med
    print(f"{tag}: {n} warm requests")

arms = sorted({t.rsplit("-", 1)[1] for t in res})
a, b = arms[0], arms[1] if len(arms) > 1 else arms[0]
ga = [t for t in res if t.endswith(a)]
gb = [t for t in res if t.endswith(b)]
print(f"\n{'phase':<14}" + "".join(f"{t.replace('server-',''):>12}" for t in res)
      + f"{a+' mean':>11}{b+' mean':>11}{'delta':>10}")
for k in PATS:
    ma = statistics.mean(res[t][k] for t in ga)
    mb = statistics.mean(res[t][k] for t in gb)
    print(f"{k:<14}" + "".join(f"{res[t][k]:12.0f}" for t in res)
          + f"{ma:11.0f}{mb:11.0f}{ma - mb:+10.0f}")
