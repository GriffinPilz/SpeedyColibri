#!/usr/bin/env bash
# Task #97 step 1: does the "+5.75 s on the shared expert" reproduce at all?
#
# The original observation compared TWO BINARIES (63f500c vs a95a5b9), which differ by
# more than the GPU scan -- #96's scratch reuse and the profile split rode along. Here
# both arms are the SAME binary and the only difference is COLI_MAMBA_CPU:
#
#   gpu = GPU seq scan (default)      cpu = COLI_MAMBA_CPU=1, the pre-#91 CPU scan
#
# `mamba_scan_gpu_enabled()` is a OnceLock, so the arm is fixed per process -- hence a
# server restart per arm rather than interleaved requests. Blocks run gpu,cpu,cpu,gpu
# (mirrored) because a fixed within-pair order penalises whichever arm runs second by
# ~3% on this box (trap 10). Each block discards request 1 and keeps 5.
#
# Profile counters are CUMULATIVE per process, so per-request phase cost is the
# difference between consecutive prints; we keep the raw lines and difference later.
set -u
REPO=/home/dgx1/SpeedyColibri-nemotron
BIN=$REPO/target/release/coli
CONTAINER=/home/dgx1/models/Nemotron-3-Super-120B-container
PORT=8098
NREQ=6
OUT=/tmp/scan_ab
rm -rf "$OUT"; mkdir -p "$OUT"

run_block() {   # $1 = arm (gpu|cpu)   $2 = block index
  local arm=$1 idx=$2 env=()
  [[ "$arm" == cpu ]] && env=(COLI_MAMBA_CPU=1)
  echo "########## block $idx: arm=$arm ##########"

  pkill -f "coli serve" 2>/dev/null; sleep 3
  setsid nohup env COLI_PROFILE=1 COLI_TIMING=1 "${env[@]}" \
    "$BIN" serve "$CONTAINER" "$PORT" </dev/null >"$OUT/server-$idx-$arm.log" 2>&1 &
  # wait for the listener, not for a fixed sleep
  for _ in $(seq 1 240); do
    grep -q "OpenAI-compatible server" "$OUT/server-$idx-$arm.log" && break
    sleep 2
  done
  grep -q "OpenAI-compatible server" "$OUT/server-$idx-$arm.log" || {
    echo "  SERVER FAILED TO START"; tail -5 "$OUT/server-$idx-$arm.log"; return 1; }
  # Confirm the arm actually took effect rather than trusting the env var. The FIRST run
  # of this A/B compared `target/release/coli` against itself: that binary was a stale
  # pre-#91 build left behind by an earlier bisect, so COLI_MAMBA_CPU had nothing to
  # switch off and both arms measured the CPU scan. Wall clock looked fine (0.6% apart);
  # only the phase table gave it away. So assert on the phase, not on the env var:
  # a live GPU scan is sub-second, the CPU scan is ~8 s.
  echo "  arm requested: COLI_MAMBA_CPU=${env[*]:-unset}"

  python3 /tmp/warm_prefill.py "127.0.0.1:$PORT" "$NREQ" /tmp/passage2.txt \
    2>&1 | tee "$OUT/client-$idx-$arm.log"

  # phase assertion: last cumulative scan total / number of prints ~= per-request scan
  local scan_ms n_prof
  scan_ms=$(grep -o "scan [0-9.]* ms" "$OUT/server-$idx-$arm.log" | tail -1 | tr -dc '0-9.')
  n_prof=$(grep -c "mamba breakdown" "$OUT/server-$idx-$arm.log")
  local per_req
  per_req=$(awk -v s="${scan_ms:-0}" -v n="${n_prof:-1}" 'BEGIN{printf "%.0f", (n>0)? s/n : 0}')
  echo "  ARM CHECK: scan ~${per_req} ms/request over $n_prof requests"
  if [[ "$arm" == gpu && "$per_req" -gt 3000 ]]; then
    echo "  *** ARM DID NOT TAKE EFFECT: gpu arm is running the CPU scan ***"
  elif [[ "$arm" == cpu && "$per_req" -lt 3000 ]]; then
    echo "  *** ARM DID NOT TAKE EFFECT: cpu arm is running the GPU scan ***"
  fi

  pkill -f "coli serve" 2>/dev/null; sleep 3
  echo
}

run_block gpu 1
run_block cpu 2
run_block cpu 3
run_block gpu 4
echo "AB_DONE"
