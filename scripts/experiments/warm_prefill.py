#!/usr/bin/env python3
"""Measure WARM prefill through `coli serve`, the only fair number to compare to a server.

Protocol, matching how the recorded 23.1 s figure was taken:
  - one server, loaded once, so the in-process expert cache is paid at startup and NOT
    per request (a one-shot `coli gen` refills 53.5 GB / ~9.9 s every invocation);
  - the IDENTICAL prompt every request, decoded from the nemotron registry ids so the
    prompt matches every other nemotron figure in the matrix doc;
  - max_tokens=1, so the wall clock is prefill plus a single decode step;
  - request 1 is DISCARDED as the warm-up; the reported figure is the median of the rest.

Emits the generated token id too, so the caller can gate on token identity across arms.
"""
import json
import statistics
import sys
import time
import urllib.request

HOSTPORT = sys.argv[1] if len(sys.argv) > 1 else "127.0.0.1:8080"
NREQ = int(sys.argv[2]) if len(sys.argv) > 2 else 5
PROMPT_FILE = sys.argv[3] if len(sys.argv) > 3 else "/tmp/passage2.txt"

prompt = open(PROMPT_FILE).read()
url = f"http://{HOSTPORT}/v1/completions"


def one():
    body = json.dumps(
        {"model": "nemotron", "prompt": prompt, "max_tokens": 1, "stream": False,
         "temperature": 0}
    ).encode()
    req = urllib.request.Request(url, data=body,
                                 headers={"Content-Type": "application/json"})
    t0 = time.perf_counter()
    with urllib.request.urlopen(req, timeout=600) as r:
        out = json.loads(r.read())
    dt = time.perf_counter() - t0
    text = out["choices"][0]["text"]
    usage = out.get("usage", {})
    return dt, text, usage.get("prompt_tokens"), usage.get("completion_tokens")


rows = []
for i in range(NREQ):
    dt, text, ptok, ctok = one()
    tag = "warmup(discarded)" if i == 0 else ""
    print(f"  req {i+1}: {dt:7.3f}s  prompt_tokens={ptok} completion={ctok} "
          f"text={text!r} {tag}", flush=True)
    rows.append((dt, text, ptok, ctok))

kept = rows[1:]
if not kept:
    sys.exit("need >=2 requests")
times = [r[0] for r in kept]
texts = {r[1] for r in kept}
ptoks = {r[2] for r in kept}
med = statistics.median(times)
ptok = kept[0][2] or 0
print()
print(f"  kept {len(kept)} requests: {['%.3f' % t for t in times]}")
print(f"  MEDIAN {med:.3f} s   prompt_tokens={ptok}   "
      f"prefill {ptok/med:.1f} tok/s" if ptok else f"  MEDIAN {med:.3f} s")
print(f"  spread {(max(times)-min(times))/med*100:.2f}% of median")
print(f"  token identity: {'OK' if len(texts) == 1 else 'DIVERGED ' + repr(texts)}"
      f"   prompt_tokens stable: {'OK' if len(ptoks) == 1 else 'NO ' + repr(ptoks)}")
