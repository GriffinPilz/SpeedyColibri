#!/usr/bin/env bash
# #97 step 3: which commit made the shared expert 15x slower?
#
# base 63f500c  shared 380 ms   |   ce53d25  shared 5698 ms, both on the CPU scan.
# moe.rs is byte-identical between them and so is try_expert_ffn_relu2, so the cause is
# runtime state, not the shared-expert code. Three code commits sit in between:
#
#   9ed41bd  #26  vLLM head-to-head + COLI_READ_SUB_KB (safetensors read path)
#   f36f798  #27  GPU Mamba2 seq scan (new CUDA device + pinned host buffers)
#   a95a5b9  #28  mamba scratch reuse + profile split + closed-hypothesis knobs
#
# COLI_MAMBA_CPU=1 everywhere so the scan is constant across all builds and the only
# thing moving is the rest of each commit. One block of 6 requests each; we are looking
# for a 15x step in one counter, not a few percent, so reps are cheap here.
set -u
REPO=/home/dgx1/SpeedyColibri-nemotron
CONTAINER=/home/dgx1/models/Nemotron-3-Super-120B-container
PORT=8098
OUT=/tmp/bisect_shared
SIG="serve $CONTAINER $PORT"
rm -rf "$OUT"; mkdir -p "$OUT"

stop_server() {
  pkill -f "$SIG" 2>/dev/null
  for _ in $(seq 1 60); do ss -ltn 2>/dev/null | grep -q ":$PORT " || return 0; sleep 2; done
  pkill -9 -f "$SIG" 2>/dev/null
  for _ in $(seq 1 30); do ss -ltn 2>/dev/null | grep -q ":$PORT " || return 0; sleep 2; done
  echo "  WARNING: port still bound"
}

cd "$REPO" || exit 1
for sha in 63f500c 9ed41bd f36f798 a95a5b9 ce53d25; do
  echo "########## $sha ##########"
  git checkout -q --detach "$sha" || { echo "  checkout FAILED"; continue; }
  if ! scripts/build.sh >"$OUT/build-$sha.log" 2>&1; then
    echo "  BUILD FAILED"; tail -3 "$OUT/build-$sha.log"; continue
  fi
  stop_server
  setsid nohup env COLI_PROFILE=1 COLI_MAMBA_CPU=1 \
    ./target/release/coli serve "$CONTAINER" "$PORT" </dev/null \
    >"$OUT/server-$sha.log" 2>&1 &
  for _ in $(seq 1 240); do
    grep -q "OpenAI-compatible server" "$OUT/server-$sha.log" && break; sleep 2
  done
  grep -q "OpenAI-compatible server" "$OUT/server-$sha.log" || { echo "  SERVER FAILED"; continue; }

  python3 /tmp/warm_prefill.py "127.0.0.1:$PORT" 6 /tmp/passage2.txt 2>&1 \
    | tee "$OUT/client-$sha.log" | grep -E "MEDIAN|token identity"

  # per-request means straight off the cumulative counters
  for phase in "shared" "scan" "gpu-ffn\(\+sync\)" "attn"; do
    v=$(grep -o "$phase [0-9.]* ms" "$OUT/server-$sha.log" | tail -1 | grep -o "[0-9.]*")
    n=$(grep -c "mamba breakdown" "$OUT/server-$sha.log")
    awk -v p="$phase" -v s="${v:-0}" -v n="${n:-1}" \
      'BEGIN{printf "  %-16s %8.0f ms/request\n", p, (n>0)? s/n : 0}'
  done
  stop_server
  echo
done
git checkout -q --detach ce53d25
echo "BISECT_DONE"
