#!/usr/bin/env bash
# #97 step 2: is the shared-expert "+5.75 s" real time, or a re-attribution?
#
# Both arms run the CPU scan, so the GPU scan (#27) is held constant and the only
# difference is everything else between 63f500c and ce53d25 (#26 read_sub_kb, #27's
# non-scan changes, #28 scratch reuse + profile split + the closed-hypothesis knobs):
#
#   base = /tmp/coli_base            (63f500c, CPU scan is its only path)
#   cur  = target/release/coli       (ce53d25) + COLI_MAMBA_CPU=1
#
# If wall clock is flat and only the `shared` counter moves, #28 re-attributed existing
# work into that timer. If wall clock moves with it, there is a real regression.
#
# Mirrored blocks (base,cur,cur,base) for the same reason as scan_ab.sh: whichever arm
# runs second is penalised ~3% on this box.
set -u
REPO=/home/dgx1/SpeedyColibri-nemotron
CONTAINER=/home/dgx1/models/Nemotron-3-Super-120B-container
PORT=8098
NREQ=6
OUT=/tmp/bin_ab
rm -rf "$OUT"; mkdir -p "$OUT"

# A server holding ~59 GB of expert cache does not exit in 3 s. The first attempt at
# this script used `pkill; sleep 3` and blocks 2-4 all died on "Address already in use",
# so three of four arms produced nothing while the script reported success. Wait for the
# port to actually be free, and escalate to SIGKILL if it is not.
# Match on the argv signature, NOT on "coli serve": the base arm's binary is
# /tmp/coli_base, whose command line reads "coli_base serve ...", so `pkill -f
# "coli serve"` silently matches nothing and the old server keeps the port. That is what
# killed blocks 2-4 on the first attempt -- and the same pattern in my `ps` check made
# the box look idle while a 59 GB server was still up.
SIG="serve $CONTAINER $PORT"
stop_server() {
  pkill -f "$SIG" 2>/dev/null
  for _ in $(seq 1 60); do
    ss -ltn 2>/dev/null | grep -q ":$PORT " || return 0
    sleep 2
  done
  pkill -9 -f "$SIG" 2>/dev/null
  for _ in $(seq 1 30); do
    ss -ltn 2>/dev/null | grep -q ":$PORT " || return 0
    sleep 2
  done
  echo "  WARNING: port $PORT still bound after SIGKILL"
}

run_block() {   # $1 = arm (base|cur)  $2 = index
  local arm=$1 idx=$2 bin env=()
  if [[ "$arm" == base ]]; then bin=/tmp/coli_base
  else bin=$REPO/target/release/coli; env=(COLI_MAMBA_CPU=1); fi
  echo "########## block $idx: arm=$arm  bin=$(md5sum "$bin" | cut -c1-8) ##########"

  stop_server
  setsid nohup env COLI_PROFILE=1 COLI_TIMING=1 "${env[@]}" \
    "$bin" serve "$CONTAINER" "$PORT" </dev/null >"$OUT/server-$idx-$arm.log" 2>&1 &
  for _ in $(seq 1 240); do
    grep -q "OpenAI-compatible server" "$OUT/server-$idx-$arm.log" && break
    sleep 2
  done
  grep -q "OpenAI-compatible server" "$OUT/server-$idx-$arm.log" || {
    echo "  SERVER FAILED"; tail -5 "$OUT/server-$idx-$arm.log"; return 1; }

  python3 /tmp/warm_prefill.py "127.0.0.1:$PORT" "$NREQ" /tmp/passage2.txt 2>&1 \
    | tee "$OUT/client-$idx-$arm.log"

  # both arms must be on the CPU scan for this comparison to isolate anything
  local scan_ms n
  scan_ms=$(grep -o "scan [0-9.]* ms" "$OUT/server-$idx-$arm.log" | tail -1 | tr -dc '0-9.')
  n=$(grep -c "mamba breakdown" "$OUT/server-$idx-$arm.log")
  awk -v s="${scan_ms:-0}" -v n="${n:-1}" 'BEGIN{printf "  SCAN CHECK: %.0f ms/request (must be ~8000 for BOTH arms)\n", (n>0)? s/n : 0}'

  stop_server
  echo
}

run_block base 1
run_block cur  2
run_block cur  3
run_block base 4
echo "BIN_AB_DONE"
