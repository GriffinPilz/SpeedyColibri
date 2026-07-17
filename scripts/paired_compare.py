#!/usr/bin/env python3
"""Paired comparison of `bench_serve.py` runs across two configs.

Why paired: every config runs the *same* prompt list, and prompts differ wildly in
intrinsic cost (a code prompt routes to different experts than prose). Comparing
medians throws that structure away and buries a real effect under prompt-to-prompt
spread — which is exactly how a first pass at this concluded "no significant
difference". Pairing by prompt cancels prompt difficulty and asks the only question
that matters: *for the same prompt*, is config B faster than config A?

Reports the per-prompt deltas, a bootstrap CI on the mean delta, and a sign test
(how many prompts moved which way) — the sign test makes no distributional
assumption at all, which is worth having when n=12 and the metric is a ratio.

Input: a log with `########## GB=<n> REP=<r> ##########` section headers followed by
bench_serve rows `<idx> <tok/s> <tok> <sec> '<prompt>'`.

Usage: paired_compare.py <log> [baseline_gb] [other_gb]
"""

import collections
import math
import re
import statistics
import sys

HDR = re.compile(r"#+\s*GB=(\d+)\s+REP=(\d+)\s*#+")
ROW = re.compile(r"^\s*(\d+)\s+([\d.]+)\s+(\d+)\s+([\d.]+)\s+'(.*)'\s*$")


def parse(path):
    """-> {gb: {prompt: [tok/s per rep]}}"""
    out = collections.defaultdict(lambda: collections.defaultdict(list))
    gb = None
    for line in open(path):
        m = HDR.search(line)
        if m:
            gb = int(m.group(1))
            continue
        m = ROW.match(line)
        if m and gb is not None:
            out[gb][m.group(5)].append(float(m.group(2)))
    return out


# Two-sided 95% t critical values by degrees of freedom. A percentile bootstrap was
# tried first and measured MISCALIBRATED here: on synthetic null data (no real effect)
# it claimed one 12.7% of the time, not the advertised 5% — the percentile interval is
# too narrow at n~12. The t interval is calibrated for exactly this small-n case.
_T95 = {1: 12.706, 2: 4.303, 3: 3.182, 4: 2.776, 5: 2.571, 6: 2.447, 7: 2.365,
        8: 2.306, 9: 2.262, 10: 2.228, 11: 2.201, 12: 2.179, 13: 2.160, 14: 2.145,
        15: 2.131, 16: 2.120, 17: 2.110, 18: 2.101, 19: 2.093, 20: 2.086, 21: 2.080,
        22: 2.074, 23: 2.069, 24: 2.064, 25: 2.060, 26: 2.056, 27: 2.052, 28: 2.048,
        29: 2.045, 30: 2.042}


def t_ci(deltas):
    """Two-sided 95% CI on the mean paired delta (Student t, small-n correct)."""
    n = len(deltas)
    if n < 2:
        return float("-inf"), float("inf")
    m = statistics.fmean(deltas)
    se = statistics.stdev(deltas) / math.sqrt(n)
    t = _T95.get(n - 1, 1.96)
    return m - t * se, m + t * se


def sign_p(wins, n):
    """Exact two-sided binomial p for `wins` of `n` under p=0.5 — distribution-free,
    so it doesn't care that tok/s is a ratio with a skewed tail."""
    if n == 0:
        return 1.0
    c = lambda k: math.comb(n, k)
    tail = sum(c(k) for k in range(0, min(wins, n - wins) + 1))
    return min(1.0, 2 * tail / 2 ** n)


def main():
    if len(sys.argv) < 2:
        print(__doc__)
        sys.exit(2)
    data = parse(sys.argv[1])
    if not data:
        print("no bench rows parsed — did the log keep per-prompt lines?")
        sys.exit(1)
    gbs = sorted(data)
    a = int(sys.argv[2]) if len(sys.argv) > 2 else gbs[0]
    b = int(sys.argv[3]) if len(sys.argv) > 3 else gbs[-1]

    for gb in gbs:
        reps = {len(v) for v in data[gb].values()}
        allv = [x for v in data[gb].values() for x in v]
        print(f"  GB={gb}: {len(data[gb])} prompts x {reps} reps, "
              f"median {statistics.median(allv):.3f} tok/s over {len(allv)} requests")
    print()

    shared = sorted(set(data[a]) & set(data[b]))
    if not shared:
        print("no prompts in common")
        sys.exit(1)

    print(f"PAIRED: GB={b} vs GB={a}  (per-prompt mean across reps)")
    print(f"{'tok/s '+str(a):>12} {'tok/s '+str(b):>12} {'delta':>8} {'%':>7}  prompt")
    deltas, pct = [], []
    for p in shared:
        va, vb = statistics.fmean(data[a][p]), statistics.fmean(data[b][p])
        d = vb - va
        deltas.append(d)
        pct.append(d / va * 100)
        print(f"{va:>12.3f} {vb:>12.3f} {d:>+8.3f} {d/va*100:>+6.1f}%  {p[:34]!r}")

    md = statistics.fmean(deltas)
    lo, hi = t_ci(deltas)
    wins = sum(1 for d in deltas if d > 0)
    n = len(deltas)
    print()
    print(f"mean delta      : {md:+.3f} tok/s ({statistics.fmean(pct):+.1f}%)")
    print(f"95% CI (t)      : [{lo:+.3f}, {hi:+.3f}]")
    print(f"sign test       : GB={b} faster on {wins}/{n} prompts (p={sign_p(wins, n):.3f})")
    crosses = lo <= 0 <= hi
    print()
    if crosses:
        print(f"VERDICT: not resolved — the CI crosses zero, so GB={b} and GB={a} are")
        print("         indistinguishable on this workload. Prefer the safer setting.")
    else:
        better = b if md > 0 else a
        print(f"VERDICT: GB={better} is genuinely faster — the CI excludes zero.")


if __name__ == "__main__":
    main()
