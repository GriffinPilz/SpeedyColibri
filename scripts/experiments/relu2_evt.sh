#!/usr/bin/env bash
# #98 E2: is `shared`'s 6189 ms at 65 GB inside the GPU kernel, or in the CPU wall time
# AROUND it (the per-call vec![0f32;...] allocation faulting under memory pressure)?
#
# COLI_RELU2_EVT=1 splits coli_cuda_expert_mlp_nvfp4_relu2's GPU timeline into
# stage(H2D) vs kernel, bucketed by input width D. The shared expert is D=4096, the
# routed latent experts D=1024, so the two buckets separate them. If the D=4096 kernel
# time is ~flat between 20 and 65 GB while the Rust-side SHARED_US grows 16x, the 6189 ms
# is CPU-side allocation, not GPU — which points the fix at scratch pooling (like #96),
# not the kernel or the zero-copy read path.
set -u

# HISTORICAL: this script swept COLI_RAM_GB, which no longer exists — the expert-cache
# budget is adaptive only. Every arm would now get the SAME budget and the sweep would
# report a difference of zero as though it were a finding. Die instead of misleading.
die_removed() {
  echo "ERROR: this experiment swept COLI_RAM_GB, which was removed (adaptive budget only)." >&2
  echo "       Re-express the arms in terms of something that still exists before running." >&2
  exit 2
}
die_removed

REPO=/home/dgx1/SpeedyColibri-nemotron
CONTAINER=/home/dgx1/models/Nemotron-3-Super-120B-container
PORT=8098
OUT=/tmp/relu2_evt
BIN=$REPO/target/release/coli
SIG="serve $CONTAINER $PORT"
NREQ=6
rm -rf "$OUT"; mkdir -p "$OUT"

stop_server() {
  pkill -f "$SIG" 2>/dev/null
  for _ in $(seq 1 60); do ss -ltn 2>/dev/null | grep -q ":$PORT " || return 0; sleep 2; done
  pkill -9 -f "$SIG" 2>/dev/null
  for _ in $(seq 1 30); do ss -ltn 2>/dev/null | grep -q ":$PORT " || return 0; sleep 2; done
  echo "  WARNING: port $PORT still bound"
}

run_arm() {
  local tag=$1 ram=$2
  echo "########## $tag  RAM_GB=$ram ##########"
  stop_server
  setsid nohup env COLI_PROFILE=1 COLI_RELU2_EVT=1 COLI_RAM_GB="$ram" \
    "$BIN" serve "$CONTAINER" "$PORT" </dev/null >"$OUT/server-$tag.log" 2>&1 &
  for _ in $(seq 1 450); do
    grep -q "OpenAI-compatible server" "$OUT/server-$tag.log" && break; sleep 2
  done
  if ! grep -q "OpenAI-compatible server" "$OUT/server-$tag.log"; then
    echo "  SERVER FAILED"; tail -5 "$OUT/server-$tag.log"; return
  fi
  python3 "$REPO/scripts/experiments/warm_prefill.py" "127.0.0.1:$PORT" "$NREQ" /tmp/passage2.txt \
    2>&1 | tee "$OUT/client-$tag.log" | grep -Ei "median|token identity"
  echo "  --- last relu2-evt lines (GPU-side stage/kernel by D) ---"
  grep "relu2-evt" "$OUT/server-$tag.log" | tail -4 | sed 's/^/  /'
  stop_server
  echo
}

run_arm r20 20
run_arm r65 65
echo "RELU2_EVT_DONE"
