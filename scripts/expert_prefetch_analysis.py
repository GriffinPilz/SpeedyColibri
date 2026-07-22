#!/usr/bin/env python3
"""Measure whether a speculative expert prefetcher is worth building.

Input: a `COLI_EXPERT_LOG` file — one line per routed position,
`step layer pos e0 e1 ... ek` (top-K experts, best-first).

We replay the tokens in generation order and, for each MoE layer, ask: *one layer
ahead* (i.e. while computing layer L-1, so its experts are known but L's router has
not run), how well can we predict layer L's experts and — with an LRU cache
simulated alongside — what fraction of the actual disk **loads (misses)** would a
prefetch of the top-N predictions have hidden?

Everything is ONLINE: each prediction uses only statistics accumulated from
*earlier* tokens, so the numbers reflect what a live prefetcher could actually do
(no peeking at the future, no train/test leakage).

Predictors compared:
  freq         per-layer global frequency (a static hot-set — ~ what AUTOPIN pins)
  persist      the previous token's experts at the same layer (temporal locality)
  markov       from THIS token's previous-layer experts, the learned L-1 -> L
               co-occurrence (context-aware, the interesting one)
  markov+freq  markov, backfilled with freq

Usage: expert_prefetch_analysis.py <log> [cache_experts] [topN,topN,...]
  cache_experts default 4600 (~85 GB / 18.9 MB per int4 expert).
"""

import sys
from collections import defaultdict, Counter, OrderedDict


def parse(path):
    """Return tokens in generation order: list of {layer: [experts]}."""
    seqs = {}
    order = []
    with open(path) as f:
        for line in f:
            if line.startswith("#") or not line.strip():
                continue
            p = line.split()
            step, layer, pos = int(p[0]), int(p[1]), int(p[2])
            key = (step, pos)
            if key not in seqs:
                seqs[key] = {}
                order.append(key)
            seqs[key][layer] = [int(x) for x in p[3:]]
    order.sort()  # (step, pos) is generation order
    return [seqs[k] for k in order]


def evaluate(tokens, cache_experts, topns):
    preds = ["freq", "persist", "markov", "markov+freq"]
    recall = {p: {n: [] for n in topns} for p in preds}   # of the 8 actual
    misscov = {p: {n: [] for n in topns} for p in preds}  # of the actual misses

    freq = defaultdict(Counter)                                   # layer -> Counter(expert)
    trans = defaultdict(lambda: defaultdict(Counter))            # layer -> prevExpert -> Counter(expert)
    prev_token = None

    cache = OrderedDict()  # (layer, expert) -> True, LRU

    def resident(item):
        return item in cache

    def touch(item):
        if item in cache:
            cache.move_to_end(item)
        else:
            cache[item] = True
            if len(cache) > cache_experts:
                cache.popitem(last=False)

    n_actual = n_miss = n_access = 0

    for layer_experts in tokens:
        sorted_l = sorted(layer_experts)
        for i, l in enumerate(sorted_l):
            actual = layer_experts[l]
            actual_set = set(actual)
            misses = {e for e in actual_set if not resident((l, e))}
            prevL = sorted_l[i - 1] if i > 0 else None

            freq_pred = [e for e, _ in freq[l].most_common(max(topns))]
            persist_pred = list(prev_token.get(l, [])) if prev_token else []
            mk = Counter()
            if prevL is not None:
                for pe in layer_experts[prevL]:
                    mk.update(trans[l][pe])
            markov_pred = [e for e, _ in mk.most_common(max(topns))]
            mf = list(dict.fromkeys(markov_pred + freq_pred))
            plists = {"freq": freq_pred, "persist": persist_pred,
                      "markov": markov_pred, "markov+freq": mf}

            for p in preds:
                pl = plists[p]
                for n in topns:
                    top = set(pl[:n])
                    if actual_set:
                        recall[p][n].append(len(top & actual_set) / len(actual_set))
                    if misses:
                        misscov[p][n].append(len(top & misses) / len(misses))

            n_actual += len(actual_set)
            n_miss += len(misses)
            n_access += 1

            # commit the access: update cache + online stats
            for e in actual:
                touch((l, e))
            freq[l].update(actual)
            if prevL is not None:
                for pe in layer_experts[prevL]:
                    trans[l][pe].update(actual)
        prev_token = {l: es for l, es in layer_experts.items()}

    return preds, recall, misscov, n_actual, n_miss, n_access


def mean(xs):
    return sum(xs) / len(xs) if xs else 0.0


def main():
    if len(sys.argv) < 2:
        print(__doc__)
        sys.exit(2)
    path = sys.argv[1]
    cache_experts = int(sys.argv[2]) if len(sys.argv) > 2 else 4600
    topns = [int(x) for x in sys.argv[3].split(",")] if len(sys.argv) > 3 else [8, 16, 24]

    tokens = parse(path)
    preds, recall, misscov, n_actual, n_miss, n_access = evaluate(tokens, cache_experts, topns)

    print(f"tokens={len(tokens)}  accesses(token*layer)={n_access}  "
          f"experts/access={n_actual/max(n_access,1):.1f}")
    print(f"cache={cache_experts} experts  ->  miss rate {n_miss/max(n_actual,1)*100:.0f}% "
          f"({n_miss} loads of {n_actual})")
    print()
    print("Predictor accuracy (one layer ahead, online). recall = of the 8 actual;")
    print("miss-cov = of the actual cache MISSES (the loads a prefetch would hide).")
    print()
    hdr = "predictor      " + "".join(f"  recall@{n:<3} misscov@{n:<3}" for n in topns)
    print(hdr)
    print("-" * len(hdr))
    for p in preds:
        row = f"{p:<14}"
        for n in topns:
            row += f"  {mean(recall[p][n])*100:8.0f}% {mean(misscov[p][n])*100:9.0f}%"
        print(row)
    print()
    best = max(preds, key=lambda p: mean(misscov[p][topns[-1]]))
    print(f"best miss-coverage @top{topns[-1]}: '{best}' hides "
          f"{mean(misscov[best][topns[-1]])*100:.0f}% of loads "
          f"(prefetching {topns[-1]} experts/layer, waste "
          f"{(1-mean(recall[best][topns[-1]])):.0%} of prefetches unused).")


if __name__ == "__main__":
    main()
