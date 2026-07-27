#!/usr/bin/env bash
set -u
REPO=/home/dgx1/SpeedyColibri-nemotron
C=/home/dgx1/models/Nemotron-3-Super-120B-container
PORT=8098; SIG="serve $C $PORT"; OUT=/tmp/wsmm_ab; rm -rf $OUT; mkdir -p $OUT
stop(){ pkill -f "$SIG" 2>/dev/null; for _ in $(seq 1 40); do ss -ltn 2>/dev/null|grep -q ":$PORT "||return; sleep 2; done; pkill -9 -f "$SIG" 2>/dev/null; sleep 3; }
arm(){ local tag=$1 ws=$2
  echo "########## $tag  NVFP4_WSMM=$ws ##########"; stop
  setsid nohup env COLI_PROFILE=1 COLI_RELU2_EVT=1 COLI_NVFP4_WSMM=$ws "$REPO/target/release/coli" serve "$C" "$PORT" </dev/null >$OUT/s-$tag.log 2>&1 &
  for _ in $(seq 1 450); do grep -q "OpenAI-compatible server" $OUT/s-$tag.log && break; sleep 2; done
  grep -q "OpenAI-compatible server" $OUT/s-$tag.log || { echo SERVER_FAILED; tail -5 $OUT/s-$tag.log; return; }
  python3 "$REPO/scripts/experiments/warm_prefill.py" 127.0.0.1:$PORT 6 /tmp/passage2.txt 2>&1 | tee $OUT/c-$tag.log | grep -Ei "median|token id"
  grep "relu2-evt" $OUT/s-$tag.log | tail -2 | sed "s/^/  /"
  stop; echo; }
# ABBA on off off on
arm b1-on 1; arm b2-off 0; arm b3-off 0; arm b4-on 1
echo "=== per-arm generated token (cross-arm identity) ==="
for t in b1-on b2-off b3-off b4-on; do
  tok=$(grep -oE "completion=[0-9]+ text=.[^\x27]*" $OUT/c-$t.log | head -1)
  echo "$t: $(grep -oE \"text=.[^,]*\" $OUT/c-$t.log | head -1)"
done
echo WSMM_AB_DONE
