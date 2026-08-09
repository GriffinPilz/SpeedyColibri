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
import os
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
    # The served model is whichever container `coli serve` loaded; the API `model`
    # field is just a label. The harness passes COLI_SERVE_MODEL so logs name the
    # right model (default keeps the historical "glm").
    model = os.environ.get("COLI_SERVE_MODEL", "glm")
    body = json.dumps(
        {"model": model, "prompt": prompt, "max_tokens": n_tokens, "stream": False}
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


def measure_fixed_cost(url, reps=5):
    """Seconds a request costs before the per-token rate applies.

    A 1-token request is one decode step plus everything that is paid once: the accept,
    the HTTP parse, tokenization, prefill of a short prompt, detokenization. Subtracting
    one token's marginal cost would need the rate we are trying to derive, so this returns
    the whole 1-token time and the caller absorbs that single token — the error is one
    token's worth, which is under 1% of a 32-token request.

    This exists because the serve column read as a mystery without it. maple measured
    70.5 tok/s serving against 112.8 tok/s decoding, and the whole difference was a fixed
    cost divided by only 32 tokens — not a slower engine. Reporting one number invited the
    wrong conclusion twice, so now both are printed.
    """
    ts = []
    for _ in range(reps + 1):  # first is a warmup, discarded
        r = one(url, PROMPTS[0], 1)
        if r is not None:
            ts.append(r[2])
    if len(ts) < 2:
        return None
    return min(ts[1:])  # min, not median: this is a floor, and noise only adds


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
    rates, toks = [], []
    for i, p in enumerate(prompts, 1):
        r = one(url, p, n_tokens)
        if r is None:
            continue
        rate, tokens, dt = r
        rates.append(rate)
        toks.append(tokens)
        print(f"{i:>3}  {rate:>7.2f}  {tokens:>4}  {dt:>7.1f}  {p[:38]!r}")

    # TOKEN GATE. `rate` is tokens/second, so a request returning FEWER tokens than asked
    # for still produces a perfectly plausible number — the rate just drops, and the run
    # reads as "slow model" rather than "broken model". Not hypothetical: a Kimi-K3 serve
    # run reported a clean 0.28 tok/s on the same day that model was being SIGTERMed at
    # 512-token prefill, and nothing in the summary could tell those apart.
    short = [(i, t) for i, t in enumerate(toks, 1) if t < n_tokens]
    if not rates:
        print("no successful requests")
        sys.exit(1)
    rates_sorted = sorted(rates)
    print()
    if short:
        print(f"SHORT COMPLETIONS: {len(short)}/{len(toks)} returned < {n_tokens} tokens "
              f"-> {short[:5]}", file=sys.stderr)
    print(f"requests   : {len(rates)}  (tokens {min(toks)}-{max(toks)} of {n_tokens} asked)")
    print(f"median     : {statistics.median(rates):.2f} tok/s   <- compare this")
    print(f"mean       : {statistics.fmean(rates):.2f} tok/s")
    print(f"min / max  : {rates_sorted[0]:.2f} / {rates_sorted[-1]:.2f} tok/s")
    if len(rates) > 1:
        print(f"stdev      : {statistics.stdev(rates):.2f}")
    print(f"first req  : {rates[0]:.2f} tok/s (cold cache; excluded from nothing, just noted)")

    fixed = measure_fixed_cost(url)
    if fixed is not None:
        med_dt = statistics.median(
            [t / r for t, r in zip(toks, rates)]  # seconds per request
        )
        med_tok = statistics.median(toks)
        gen_s = med_dt - fixed
        print()
        print(f"fixed cost : {fixed * 1000:.0f} ms per request, before the first token")
        if gen_s > 0:
            print(f"marginal   : {med_tok / gen_s:.2f} tok/s   <- compare THIS to the decode column")
            print(f"             ({fixed * 1000:.0f} ms + {gen_s / med_tok * 1000:.2f} ms x {med_tok:.0f} tok "
                  f"= {med_dt * 1000:.0f} ms)")
        else:
            print("marginal   : n/a (fixed cost exceeds the median request — raise the token count)")


if __name__ == "__main__":
    main()
