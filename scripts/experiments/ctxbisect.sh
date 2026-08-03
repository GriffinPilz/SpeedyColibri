#!/usr/bin/env bash
# #54 — locate the context length at which prefill leaves the GPU path.
#
# MEASURED 2026-08-03: a 113851-token prefill on m2.7 ran ONE thread at 100% CPU for 202
# minutes with the GPU at 0%, zero disk I/O across 93 minutes, and flat memory. It never
# finished. At 512 tokens the same model is healthy (18.5 tok/s, GPU busy).
#
# Somewhere between those two the work moves off the GPU. Bisecting is minutes per point
# instead of hours, and the profile breakdown says WHICH phase moved.
#
# ANSWERED (#54): the knee is exactly at 8192. `coli_cuda_gqa_attn` had one `T > 8192` guard
# covering both of its kernels, and it is a real shared-memory bound for only ONE of them —
# so the flash kernel, which tiles over keys and exists precisely for long context, was
# refused along with the scalar one. `return 0` reads as "no GPU", so the caller dropped to
# the single-threaded CPU core. 8192 ran GPU-busy in 163 s; 16384 sat at GPU 0% / CPU 100%.
# Keep this script: the same shape recurs (see #56 — the next wall is gridDim.y at S=65535),
# and the cheapest way to find a silent size gate is still to sweep the size.
#
# What each captured column is FOR — a sweep that only records wall time can tell you the
# knee exists but not what it is:
#   attn vs moe        which phase grows superlinearly
#   core, proj         attn's internals: an O(n^2) core is expected to grow, proj is not
#   attn - (proj+core) the gpu-eligible-trap tell. A phase total that greatly exceeds the sum
#                      of its GPU sub-timers means the work ran somewhere those timers do not
#                      cover — i.e. a silent CPU fallback. This is the single most diagnostic
#                      column here, and it is why sub-timers are captured at all.
#   gpu_mean           idle GPU + busy CPU is the fallback signature
#   cpu_max            ONE thread at 100% (~100) vs many (~2000 on 20 cores) separates a
#                      serial fallback from merely slow parallel work. `matmul_qt`'s CPU path
#                      is single-threaded (see memory dsa-indexer), so ~100 points straight at it
#
# Prompts are the registry's real NL token ids repeated (see memory
# synthetic-bench-prompt-is-a-defect — the 100..611 synthetic range makes models degenerate;
# this is a PERF probe, not a correctness one, but reuse the real ids anyway).
set -u
export CUDA_HOME=/usr/local/cuda
export PATH=$HOME/.cargo/bin:/usr/local/cuda/bin:$PATH
export LD_LIBRARY_PATH=/usr/local/cuda/lib64
cd "$HOME/SpeedyColibri-k3" || exit 1
source scripts/lib.sh

others=$(ps -eo pid,args | grep -E "release/coli (gen|serv[e])" | grep -v grep || true)
[ -n "$others" ] && { echo "REFUSING — another coli is running:" >&2; echo "$others" >&2; exit 1; }
bench_lock_acquire

MODEL="${1:-minimax-m2.7}"
OUT="${CTX_OUT:-/tmp/ctxbisect}"; mkdir -p "$OUT"; R="$OUT/results.tsv"
[ -f "$R" ] || printf "ntok\twall_s\trc\tprefill_ms\tattn_ms\tcore_ms\tproj_ms\tmoe_ms\txload_ms\tunaccounted_ms\tgpu_mean\tcpu_max\n" > "$R"
load_model "$MODEL"

CAP="${CTX_CAP_SECS:-1500}"
# CTX_MULTS lets a follow-up run re-measure only the points that matter (each
# multiplier is x512 tokens). CTX_OUT keeps its results in a separate file: mixing
# pre-fix and post-fix rows in one table is a trap for whoever reads it next.
for mult in ${CTX_MULTS:-1 4 16 32 64}; do
  ntok=$((512 * mult))
  log="$OUT/n${ntok}.log"
  if [ -s "$log" ] && grep -q RUNDONE "$log"; then echo "  skip $ntok (already done)"; continue; fi
  PROMPT=""
  for _ in $(seq 1 "$mult"); do PROMPT="$PROMPT $PROMPT_TOKENS"; done

  mem_reset >/dev/null 2>&1
  # Sample GPU utilisation AND the coli process's total CPU% for the life of the run.
  # `ps -o %cpu` on the PROCESS sums its threads, so ~100 means one core saturated and
  # ~2000 means all 20 busy — the distinction the whole ticket turns on.
  ( while true; do
      g=$(nvidia-smi --query-gpu=utilization.gpu --format=csv,noheader,nounits 2>/dev/null | head -1)
      c=$(ps -eo pcpu,comm --no-headers 2>/dev/null | awk '$2=="coli"{s+=$1} END{printf "%.0f", s}')
      echo "${g:-NA} ${c:-0}"
      sleep 2
    done ) > "$OUT/samp_$ntok.txt" 2>/dev/null &
  SAMPPID=$!

  t0=$(date +%s)
  timeout "$CAP" env COLI_PROFILE=1 COLI_TIMING=1 COLI_NGEN=1 \
    ./target/release/coli gen "$CONTAINER" $PROMPT >"$log" 2>&1
  rc=$?; t1=$(date +%s)
  kill "$SAMPPID" 2>/dev/null || true
  echo "RUNDONE rc=$rc" >>"$log"

  pf=$(grep -oP 'prefill \d+ tok: \K[0-9.]+' "$log" | head -1)
  at=$(grep -oP '\battn \K[0-9]+' "$log" | head -1)
  mo=$(grep -oP '\bmoe \K[0-9]+' "$log" | head -1)
  xl=$(grep -oP 'of which expert-load \K[0-9]+' "$log" | head -1)
  # `proj` also matches inside `o-proj` (- is a non-word char), so take the FIRST match on
  # the attn-breakdown line, which is the standalone one.
  bd=$(grep -F '[profile] attn breakdown' "$log" | head -1)
  pj=$(printf '%s' "$bd" | grep -oP '\bproj \K[0-9]+' | head -1)
  co=$(printf '%s' "$bd" | grep -oP '\bcore \K[0-9]+' | head -1)
  ro=$(printf '%s' "$bd" | grep -oP 'rope\+cache \K[0-9]+' | head -1)
  ds=$(printf '%s' "$bd" | grep -oP 'dsa-indexer \K[0-9]+' | head -1)
  op=$(printf '%s' "$bd" | grep -oP 'o-proj \K[0-9]+' | head -1)
  # Residual against ALL FIVE sub-timers, not just two. Subtracting a subset would leave a
  # large positive residual on a perfectly healthy run and read as a fallback that isn't there.
  un=$(awk -v a="${at:-}" -v p="${pj:-}" -v c="${co:-}" -v r="${ro:-}" -v d="${ds:-}" -v o="${op:-}" \
        'BEGIN{ if(a==""||p==""||c==""||r==""||d==""||o=="") print "NA";
                else printf "%d", a-p-c-r-d-o }')
  gm=$(awk '{if($1!="NA"){s+=$1; n++}} END{if(n) printf "%.0f", s/n; else print "NA"}' "$OUT/samp_$ntok.txt" 2>/dev/null)
  cm=$(awk '{if($2+0>m) m=$2+0} END{if(NR) printf "%.0f", m; else print "NA"}' "$OUT/samp_$ntok.txt" 2>/dev/null)

  printf "%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\n" \
    "$ntok" "$((t1-t0))" "$rc" "${pf:-NA}" "${at:-NA}" "${co:-NA}" "${pj:-NA}" \
    "${mo:-NA}" "${xl:-NA}" "${un:-NA}" "${gm:-NA}" "${cm:-NA}" >>"$R"
  echo "  ntok=$ntok $((t1-t0))s rc=$rc | prefill=${pf:-NA}ms attn=${at:-NA} core=${co:-NA} proj=${pj:-NA} moe=${mo:-NA} | unacct=${un:-NA} GPU=${gm:-NA}% CPUmax=${cm:-NA}%"

  if [ "$rc" = 124 ]; then
    echo "  ntok=$ntok exceeded ${CAP}s — the knee is at or below here; stopping the sweep"
    break
  fi
done

echo "== TABLE =="; column -t "$R" 2>/dev/null || cat "$R"
echo CTXBISECT_COMPLETE
