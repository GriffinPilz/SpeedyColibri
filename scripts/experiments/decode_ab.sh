#!/usr/bin/env bash
# #98 FIX decode safety: the pooled shared-expert scratch runs in decode too (S=1, where
# `sh`/`uu` are a few KB). It must be (a) token-identical and (b) no slower than the
# pre-#98 fresh-alloc path. Default budget (all experts resident — the shipping config),
# 64-token generations, greedy so the token stream is deterministic across arms.
set -u
REPO=/home/dgx1/SpeedyColibri-nemotron
CONTAINER=/home/dgx1/models/Nemotron-3-Super-120B-container
PORT=8098
OUT=/tmp/decode_ab
BIN=$REPO/target/release/coli
SIG="serve $CONTAINER $PORT"
GEN=64; NREQ=6
rm -rf "$OUT"; mkdir -p "$OUT"

stop_server() {
  pkill -f "$SIG" 2>/dev/null
  for _ in $(seq 1 60); do ss -ltn 2>/dev/null | grep -q ":$PORT " || return 0; sleep 2; done
  pkill -9 -f "$SIG" 2>/dev/null
  for _ in $(seq 1 30); do ss -ltn 2>/dev/null | grep -q ":$PORT " || return 0; sleep 2; done
}

run_arm() {
  local tag=$1 scratch=$2
  echo "########## $tag  SHARED_SCRATCH=$scratch ##########"
  stop_server
  setsid nohup env COLI_SHARED_SCRATCH="$scratch" \
    "$BIN" serve "$CONTAINER" "$PORT" </dev/null >"$OUT/server-$tag.log" 2>&1 &
  for _ in $(seq 1 450); do
    grep -q "OpenAI-compatible server" "$OUT/server-$tag.log" && break; sleep 2
  done
  grep -q "OpenAI-compatible server" "$OUT/server-$tag.log" || { echo "  SERVER FAILED"; return; }
  python3 - "$PORT" "$GEN" "$NREQ" "$OUT/tok-$tag.txt" <<"PY" | tee "$OUT/client-$tag.log"
import json,sys,time,urllib.request
port,gen,nreq,tokf=sys.argv[1],int(sys.argv[2]),int(sys.argv[3]),sys.argv[4]
prompt=open("/tmp/passage2.txt").read()
url=f"http://127.0.0.1:{port}/v1/completions"
def one():
    body=json.dumps({"model":"nemotron","prompt":prompt,"max_tokens":gen,
                     "stream":False,"temperature":0}).encode()
    r=urllib.request.Request(url,data=body,headers={"Content-Type":"application/json"})
    t0=time.perf_counter()
    with urllib.request.urlopen(r,timeout=600) as resp: out=json.loads(resp.read())
    dt=time.perf_counter()-t0
    ch=out["choices"][0]; u=out.get("usage",{})
    return dt,ch["text"],u.get("completion_tokens",gen)
rates=[];txt0=None
for i in range(nreq):
    dt,txt,ct=one()
    if i==0: txt0=txt; continue           # warmup
    rates.append(ct/dt)
    if txt!=txt0: print(f"  req{i}: TOKEN MISMATCH vs warmup")
rates.sort()
med=rates[len(rates)//2]
open(tokf,"w").write(txt0)
print(f"MEDIAN decode {med:.2f} tok/s over {len(rates)} reps (gen={gen})")
print(f"first 40 chars: {txt0[:40]!r}")
PY
  stop_server
  echo
}

run_arm on  1
run_arm off 0
echo "== token identity across arms =="
if diff -q "$OUT/tok-on.txt" "$OUT/tok-off.txt" >/dev/null 2>&1; then
  echo "IDENTICAL (on == off)"
else
  echo "DIFFER — investigate"; diff "$OUT/tok-on.txt" "$OUT/tok-off.txt" | head
fi
echo "DECODE_AB_DONE"
