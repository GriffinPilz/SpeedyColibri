#!/usr/bin/env python3
"""Measure whether a *static pinned hot-set* of routed experts is worth it.

Input: a `COLI_EXPERT_LOG` file — one line per routed position,
`step layer pos e0 e1 ... ek` (top-K routed experts, best-first).

The swappable cache holds only the *routed* experts (the 1 shared expert/layer is a
resident layer weight, never swapped). With a byte budget far smaller than the full
expert set, the question is: is routing skewed enough that pinning the per-layer
*most-frequent* experts beats letting a dynamic LFRU policy churn them?

For each MoE layer we build the expert-frequency distribution and report the
**coverage curve**: sorting experts by frequency, what fraction of all routing
selections do the top-N experts cover? Compared against the uniform baseline
(top-N covers N/E). Excess over uniform == the exploitable skew a static pin buys.

We also report per-layer **Gini** (0 = uniform, 1 = one expert takes everything) and
a **held-out** coverage: split the log in half by generation order, pick the hot-set
on the first half, score coverage on the second — so the number reflects what a pin
chosen from history actually delivers on future tokens (no train/test leakage).

Usage: expert_hotset_analysis.py <log> [pin_per_layer,...]
  pin_per_layer default 32,64,128 (out of E routed experts, typically 256).
"""

import sys
from collections import defaultdict, Counter


def parse(path):
    """Return (rows_in_order, n_experts). rows: (step, layer, pos, [experts])."""
    rows = []
    emax = 0
    with open(path) as f:
        for line in f:
            if line.startswith("#") or not line.strip():
                continue
            p = line.split()
            step, layer, pos = int(p[0]), int(p[1]), int(p[2])
            experts = [int(x) for x in p[3:]]
            if experts:
                emax = max(emax, max(experts))
            rows.append((step, layer, pos, experts))
    rows.sort(key=lambda r: (r[0], r[2], r[1]))  # generation order
    return rows, emax + 1


def gini(counts):
    xs = sorted(counts)
    n = len(xs)
    if n == 0 or sum(xs) == 0:
        return 0.0
    cum = 0
    for i, x in enumerate(xs, 1):
        cum += i * x
    return (2 * cum) / (n * sum(xs)) - (n + 1) / n


def coverage(freq, hotset, total):
    """Fraction of `total` selections that land in `hotset` under `freq`."""
    if total == 0:
        return 0.0
    return sum(freq.get(e, 0) for e in hotset) / total


def main():
    if len(sys.argv) < 2:
        print(__doc__)
        sys.exit(2)
    path = sys.argv[1]
    pins = [int(x) for x in sys.argv[2].split(",")] if len(sys.argv) > 2 else [32, 64, 128]

    rows, E = parse(path)
    if not rows:
        print("empty log")
        sys.exit(1)

    # Per-layer frequency over the whole log (for the in-sample coverage curve),
    # and a first-half / second-half split for the held-out number.
    by_layer = defaultdict(list)
    for step, layer, pos, experts in rows:
        by_layer[layer].append(experts)

    layers = sorted(by_layer)
    print(f"log rows={len(rows)}  MoE layers={len(layers)}  routed experts E={E}")
    print(f"selections/layer ~= {len(rows)//max(len(layers),1)}  "
          f"(top-{len(rows and rows[0][3]) or '?'} per position)")
    print()

    # ---- aggregate coverage curve (mean over layers), in-sample ----
    curve = {n: [] for n in pins}
    ginis = []
    held = {n: [] for n in pins}
    for layer in layers:
        seqs = by_layer[layer]
        freq = Counter()
        for experts in seqs:
            freq.update(experts)
        total = sum(freq.values())
        ginis.append(gini([freq.get(e, 0) for e in range(E)]))
        ranked = [e for e, _ in freq.most_common()]
        for n in pins:
            curve[n].append(coverage(freq, set(ranked[:n]), total))

        # held-out: hot-set from first half, coverage on second half
        half = len(seqs) // 2
        if half >= 4:
            f1 = Counter()
            for experts in seqs[:half]:
                f1.update(experts)
            hot = set(e for e, _ in f1.most_common(max(pins)))
            f2 = Counter()
            for experts in seqs[half:]:
                f2.update(experts)
            t2 = sum(f2.values())
            for n in pins:
                held[n].append(coverage(f2, set(list(sorted(f1, key=lambda x: -f1[x]))[:n]), t2))

    def mean(xs):
        return sum(xs) / len(xs) if xs else 0.0

    print(f"mean per-layer Gini = {mean(ginis):.3f}   "
          f"(0=uniform/no hot-set, ~0.3+ = worth pinning)")
    print(f"  range [{min(ginis):.3f}, {max(ginis):.3f}] across layers")
    print()
    print(f"{'pin/layer':>10}  {'uniform':>8}  {'in-sample':>10}  {'held-out':>9}  {'excess':>7}")
    print("-" * 52)
    for n in pins:
        uni = n / E
        ins = mean(curve[n])
        ho = mean(held[n]) if held[n] else float("nan")
        print(f"{n:>10}  {uni:>7.0%}  {ins:>9.0%}  {ho:>8.0%}  {ins-uni:>+6.0%}")
    print()
    print("Read: 'held-out' is the hit rate a static pin chosen from history delivers")
    print("on future tokens. If it's ~= 'uniform', routing is balanced and there is no")
    print("hot-set to pin — stream on demand. If held-out >> uniform, pin the top-N.")


if __name__ == "__main__":
    main()
