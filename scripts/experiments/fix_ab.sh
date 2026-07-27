#!/usr/bin/env bash
# #98 FIX A/B: does pooling the shared-expert scratch (COLI_SHARED_SCRATCH, default on)
# remove the memory-pressure prefill tax at a full expert-cache budget?
#
# One binary, COLI_SHARED_SCRATCH switches the arm (1=fix/pool, 0=pre-#98 fresh alloc).
# Two budgets: 65 GB (the pressure regime where the tax appears) and 20 GB (no pressure —
# pooling must be at worst neutral there). ABBA-mirrored within each budget so the second
# arm is not systematically penalised. Token-identity gated: pooling only changes WHICH
# buffer is written, never the math, so the generated token must be byte-identical.
#
# ARM ASSERTION: with pooling ON the `shared` phase at 65 GB must drop well below its OFF
# value; the script prints the phase table so that is visible, and asserts the env reached
# the process.
set -u
REPO=/home/dgx1/SpeedyColibri-nemotron
CONTAINER=/home/dgx1/models/Nemotron-3-Super-120B-container
PORT=8098
OUT=/tmp/fix_ab
BIN=$REPO/target/release/coli
SIG="serve $CONTAINER $PORT"
NREQ=7
rm -rf "$OUT"; mkdir -p "$OUT"

stop_server() {
  pkill -f "$SIG" 2>/dev/null
  for _ in $(seq 1 60); do ss -ltn 2>/dev/null | grep -q ":$PORT " || return 0; sleep 2; done
  pkill -9 -f "$SIG" 2>/dev/null
  for _ in $(seq 1 30); do ss -ltn 2>/dev/null | grep -q ":$PORT " || return 0; sleep 2; done
  echo "  WARNING: port $PORT still bound"
}

run_arm() {
  local tag=$1 ram=$2 scratch=$3
  echo "########## $tag  RAM_GB=$ram SHARED_SCRATCH=$scratch ##########"
  stop_server
  setsid nohup env COLI_PROFILE=1 COLI_RAM_GB="$ram" COLI_SHARED_SCRATCH="$scratch" \
    "$BIN" serve "$CONTAINER" "$PORT" </dev/null >"$OUT/server-$tag.log" 2>&1 &
  for _ in $(seq 1 450); do
    grep -q "OpenAI-compatible server" "$OUT/server-$tag.log" && break; sleep 2
  done
  if ! grep -q "OpenAI-compatible server" "$OUT/server-$tag.log"; then
    echo "  SERVER FAILED"; tail -5 "$OUT/server-$tag.log"; return
  fi
  python3 "$REPO/scripts/experiments/warm_prefill.py" "127.0.0.1:$PORT" "$NREQ" /tmp/passage2.txt \
    2>&1 | tee "$OUT/client-$tag.log" | grep -Ei "median|token identity"
  tr '\0' '\n' < "/proc/$(pgrep -f "$SIG" | head -1)/environ" 2>/dev/null \
    | grep -E "COLI_(RAM_GB|SHARED_SCRATCH)=" | sed 's/^/  env: /'
  stop_server
  echo
}

# 65 GB pressure regime: A B B A  (on off off on)
run_arm "r65-b1-on"  65 1
run_arm "r65-b2-off" 65 0
run_arm "r65-b3-off" 65 0
run_arm "r65-b4-on"  65 1
# 20 GB no-pressure control: on vs off (pooling must be neutral here)
run_arm "r20-on"  20 1
run_arm "r20-off" 20 0
echo "FIX_AB_DONE"
