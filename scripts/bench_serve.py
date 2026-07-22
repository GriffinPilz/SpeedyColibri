#!/usr/bin/env python3
"""Realistic throughput benchmark against a running `coli serve`.

Why this exists: the easy benchmarks lie, in two ways we hit for real.

  1. `coli gen <ids...>` greedy-decodes from arbitrary token ids and **degenerates
     into a loop** after ~30 tokens (it emits a short cycle forever). A looping
     generation re-routes to the *same* experts every step, so the cache hit rate is
     nothing like a real workload's and throughput is flattered ~2x.
  2. Replaying **one** prompt repeatedly warms the cache on that prompt's tiny
     working set. Also flattering: a real server answers varied requests, and the
     union of their experts is far larger than any one request's.

So: many *different* natural-language prompts, through the real HTTP path, each
generation short enough to stay coherent (greedy degenerates if you let it run).
Prompts span distinct domains deliberately — routing is content-dependent, so
diverse topics are what actually exercises the expert working set.

Reports end-to-end tok/s (what a caller experiences, prefill included) per request
plus the distribution. Compare the *median*: the first request pays cold-cache costs
and would drag a mean around.

Usage: bench_serve.py [host:port] [tokens_per_request] [--repeat N]
  bench_serve.py 127.0.0.1:8080 32
"""

import json
import statistics
import sys
import time
import urllib.request

# Deliberately spread across domains: routing is content-dependent, so this is what
# makes the working set realistic rather than a single hot cluster.
PROMPTS = [
    "The capital of France is",
    "def quicksort(arr):\n    # sort a list in place\n",
    "The mitochondria in a eukaryotic cell are responsible for",
    "To make a classic risotto, first you",
    "In 1215, King John of England signed",
    "The derivative of x squared with respect to x is",
    "Once upon a time there was a",
    "The patient presented with a persistent cough and",
    "Under contract law, an offer becomes binding when",
    "The offside rule in association football states that",
    "Photosynthesis converts carbon dioxide and water into",
    "SELECT name, COUNT(*) FROM orders GROUP BY",
]


def one(url, prompt, n_tokens):
    """POST one completion; return (tok/s, tokens, seconds) or None on failure."""
    body = json.dumps(
        {"model": "glm", "prompt": prompt, "max_tokens": n_tokens, "stream": False}
    ).encode()
    req = urllib.request.Request(
        url, data=body, headers={"Content-Type": "application/json"}
    )
    t0 = time.time()
    try:
        with urllib.request.urlopen(req, timeout=1800) as r:
            out = json.loads(r.read())
    except Exception as e:  # noqa: BLE001 — report and keep going
        print(f"  ! request failed: {e}", file=sys.stderr)
        return None
    dt = time.time() - t0
    tokens = out.get("usage", {}).get("completion_tokens", 0)
    if not tokens or dt <= 0:
        return None
    return tokens / dt, tokens, dt


def main():
    hostport = sys.argv[1] if len(sys.argv) > 1 else "127.0.0.1:8080"
    n_tokens = int(sys.argv[2]) if len(sys.argv) > 2 else 32
    repeat = 1
    if "--repeat" in sys.argv:
        repeat = int(sys.argv[sys.argv.index("--repeat") + 1])
    url = f"http://{hostport}/v1/completions"

    prompts = PROMPTS * repeat
    print(f"benchmark: {len(prompts)} distinct-prompt requests x {n_tokens} tok -> {url}")
    print(f"{'#':>3}  {'tok/s':>7}  {'tok':>4}  {'sec':>7}  prompt")
    rates = []
    for i, p in enumerate(prompts, 1):
        r = one(url, p, n_tokens)
        if r is None:
            continue
        rate, tokens, dt = r
        rates.append(rate)
        print(f"{i:>3}  {rate:>7.2f}  {tokens:>4}  {dt:>7.1f}  {p[:38]!r}")

    if not rates:
        print("no successful requests")
        sys.exit(1)
    rates_sorted = sorted(rates)
    print()
    print(f"requests   : {len(rates)}")
    print(f"median     : {statistics.median(rates):.2f} tok/s   <- compare this")
    print(f"mean       : {statistics.fmean(rates):.2f} tok/s")
    print(f"min / max  : {rates_sorted[0]:.2f} / {rates_sorted[-1]:.2f} tok/s")
    if len(rates) > 1:
        print(f"stdev      : {statistics.stdev(rates):.2f}")
    print(f"first req  : {rates[0]:.2f} tok/s (cold cache; excluded from nothing, just noted)")


if __name__ == "__main__":
    main()
