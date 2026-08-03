#!/usr/bin/env bash
# #56 — does the gridDim.y = 65535 limit send prefill back to the CPU above S=65535?
#
# The point of this probe is that it does NOT need the run to finish. #54 established the
# signature: GPU ~0% with process CPU pinned at exactly ~100% means the single-threaded CPU
# core; a busy GPU with CPU in the hundreds means the GPU path. That is readable within a few
# minutes, while an actual 73728-token prefill would take roughly an hour on the GPU path
# (core is O(n^2): 695 s at 32768 -> ~3500 s at 73728) and effectively forever on the CPU one.
#
# So: start it, sample, classify, kill. 73728 = 512 x 144, comfortably above 65535, and below
# M2.7's ~100k context limit.
set -u
export CUDA_HOME=/usr/local/cuda
export PATH=$HOME/.cargo/bin:/usr/local/cuda/bin:$PATH
export LD_LIBRARY_PATH=/usr/local/cuda/lib64
cd "$HOME/SpeedyColibri-k3" || exit 1
source scripts/lib.sh

others=$(ps -eo pid,args --no-headers | grep -E "release/coli (gen|serv[e])" | grep -v grep || true)
[ -n "$others" ] && { echo "REFUSING — another coli is running:" >&2; echo "$others" | cut -c1-80 >&2; exit 1; }
bench_lock_acquire

MULT="${MULT:-144}"                 # x512 tokens
WATCH="${WATCH:-420}"               # seconds to observe before classifying
NTOK=$((512 * MULT))
OUT=/tmp/probe65k; mkdir -p "$OUT"
load_model minimax-m2.7

PROMPT=""
for _ in $(seq 1 "$MULT"); do PROMPT="$PROMPT $PROMPT_TOKENS"; done
echo "probing S=$NTOK (gridDim.y limit is 65535) — observing ${WATCH}s"

mem_reset >/dev/null 2>&1
( while true; do
    g=$(nvidia-smi --query-gpu=utilization.gpu --format=csv,noheader,nounits 2>/dev/null | head -1)
    c=$(ps -eo pcpu,comm --no-headers 2>/dev/null | awk '$2=="coli"{s+=$1} END{printf "%.0f", s}')
    echo "$(date +%s) ${g:-NA} ${c:-0}"
    sleep 2
  done ) > "$OUT/samp.txt" 2>/dev/null &
SAMPPID=$!

COLI_PROFILE=1 COLI_TIMING=1 COLI_NGEN=1 \
  ./target/release/coli gen "$CONTAINER" $PROMPT > "$OUT/run.log" 2>&1 &
GENPID=$!

sleep "$WATCH"
kill "$SAMPPID" 2>/dev/null || true

# Classify on the SECOND half of the window, so model load (GPU idle, CPU busy unpacking)
# does not get mistaken for the fallback it is meant to detect.
n=$(wc -l < "$OUT/samp.txt")
echo "--- samples: $n (classifying on the last half) ---"
awk -v n="$n" 'NR>n/2 {if($2!="NA"){g+=$2; gn++}; c+=$3; cn++}
     END{ printf "GPU mean %.1f%%   CPU mean %.0f%%\n", (gn?g/gn:-1), (cn?c/cn:0) }' "$OUT/samp.txt"
echo "--- GPU sample distribution (last half) ---"
awk -v n="$n" 'NR>n/2 && $2!="NA" {print $2}' "$OUT/samp.txt" | sort -n | uniq -c | tail -8
echo "--- did it produce a prefill line yet? ---"
grep -E 'prefill [0-9]+ tok' "$OUT/run.log" || echo "(still in prefill — expected)"
grep -F '[cache] NOTE' "$OUT/run.log" || true
tail -2 "$OUT/run.log"

echo "--- stopping the probe ---"
kill -KILL "$GENPID" 2>/dev/null || true
sleep 2
ps -eo pid,etime,args --no-headers | grep -E 'release/col[i] gen' | cut -c1-60 || echo "clean"
echo PROBE_COMPLETE
