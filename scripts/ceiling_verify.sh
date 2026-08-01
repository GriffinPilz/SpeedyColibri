#!/usr/bin/env bash
# Does every model actually stay inside the RAM ceiling under a workload that REACHES
# its fill target?
#
#   Usage: scripts/ceiling_verify.sh [model ...]        (default: every registered model)
#
# This lived in /tmp on one box and covered a hardcoded list of four models. Kimi-K3 was
# not on that list, so K3 violated the ceiling for weeks while the check reported four
# PASSes — and when K3 was finally seen dying, the omission is what let it be misattributed
# to an unrelated buffer-pool change. The model list now comes from the registry, so a
# newly added model is covered the day it is added rather than the day someone remembers.
#
# The workload matters as much as the list. An earlier pass used a 12-token prompt and 6
# generated tokens and passed everything with swap 0 — while GLM-5.2 was configured to fill
# 123 GB on a 121 GiB box. It passed because the run ended long before the cache grew
# anywhere near its target. A short run only shows the accounting is not wrong at the
# START. So: the full 512-token prompt, and generation, per model.
#
# What is checked, and why each one is here:
#   - the process SURVIVES. earlyoom sends SIGTERM, which surfaces as rc=143 with no
#     output at all — a killed run and a quiet run look identical unless you check rc.
#   - it EMITTED TOKENS. rc=0 with no tokens has happened (a harness bug swallowed them).
#   - peak RSS stays under the ceiling.
#   - swap does not move. The kernel begins paging while MemAvailable still reads healthy,
#     so swap is the only unambiguous over-commit signal.
#   - the announced PLAN fits, not merely this run's peak: a model can stay under the
#     ceiling purely because it never got far enough to reach its own fill target.
set -u
source "$(dirname "$0")/lib.sh"

# 96% of MemTotal, matching TARGET_RAM_PCT in crates/coli/src/main.rs. Read from the box
# rather than hardcoded: a 128 GB Spark and a 121 GiB Spark are not the same number, and
# the previous hardcoded 108.0 GiB was silently wrong on any other host.
CEIL_GIB=$(awk '/MemTotal/{printf "%.1f", $2*0.96/1048576}' /proc/meminfo)
TOTAL_GB=$(awk '/MemTotal/{printf "%.1f", $2*1024/1e9}' /proc/meminfo)

# earlyoom's own trigger, so a FAIL can say whether we were killed by our own guard failing
# to brake or by something else entirely.
EO_PCT=$(ps -eo args 2>/dev/null | sed -n 's/.*earlyoom .*-m \([0-9]*\).*/\1/p' | head -1)
EO_GB=$(awk -v p="${EO_PCT:-0}" '/MemTotal/{printf "%.2f", $2*p/100*1024/1e9}' /proc/meminfo)

MODELS=("$@")
if [[ ${#MODELS[@]} -eq 0 ]]; then
  IFS=',' read -ra MODELS <<< "$(model_names)"
fi

echo "== RAM ceiling: <= 96% of ${TOTAL_GB} GB = ${CEIL_GIB} GiB, swap must not move =="
[[ -n "${EO_PCT:-}" ]] && echo "   earlyoom is armed at -m ${EO_PCT} = ${EO_GB} GB available"
echo "   models: ${MODELS[*]}"
echo

sw() { awk '/SwapTotal/{t=$2} /SwapFree/{f=$2} END{printf "%d",(t-f)/1024}' /proc/meminfo; }

bench_lock_acquire
fail=0
for m in "${MODELS[@]}"; do
  if ! load_model "$m" 2>/dev/null; then
    printf "  %-18s SKIP (not in registry)\n" "$m"; continue
  fi
  if [[ ! -d "$CONTAINER" ]]; then
    printf "  %-18s SKIP (container not materialized on this host)\n" "$m"; continue
  fi
  mem_reset >/dev/null 2>&1
  s0=$(sw); t0=$(date +%s)
  COLI_NGEN="${CEIL_NGEN:-8}" timeout "${CEIL_TIMEOUT:-2400}" /usr/bin/time -f 'MAXRSS_KB=%M' \
    "$COLI_BIN" gen "$CONTAINER" $PROMPT_TOKENS > "/tmp/ceil_$m.log" 2>&1
  rc=$?; t1=$(date +%s); s1=$(sw)
  R=$(grep -oE 'MAXRSS_KB=[0-9]+' "/tmp/ceil_$m.log" | grep -oE '[0-9]+' | tail -1)
  rss=$(awk -v k="${R:-0}" 'BEGIN{printf "%.1f", k/1048576}')
  dense=$(grep -oE 'dense [0-9]+ GB' "/tmp/ceil_$m.log" | grep -oE '[0-9]+' | head -1)
  fill=$(grep -oE 'fill to ~[0-9]+ GB' "/tmp/ceil_$m.log" | grep -oE '[0-9]+' | head -1)
  tok=$(grep -oE 'generated \([0-9]+ tok\): \[[^]]*\]' "/tmp/ceil_$m.log" | head -1)
  st="PASS"
  [[ "$rc" == 0 ]] || { st="FAIL-rc$rc"; fail=1; }
  [[ -n "$tok" ]]  || { st="FAIL-NO-TOKENS"; fail=1; }
  [[ "$s1" -le "$((s0+64))" ]] || { st="FAIL-SWAP"; fail=1; }
  awk -v r="$rss" -v c="$CEIL_GIB" 'BEGIN{exit !(r < c)}' || { st="FAIL-RSS"; fail=1; }
  # The plan itself must fit, not just this run's peak. These are printed with `>> 30`,
  # i.e. GiB despite the "GB" label, so compare them against the GiB ceiling.
  if [[ -n "$dense" && -n "$fill" ]]; then
    awk -v d="$dense" -v f="$fill" -v c="$CEIL_GIB" 'BEGIN{exit !(d+f <= c)}' \
      || { st="FAIL-PLAN"; fail=1; }
  fi
  printf "  %-18s %-14s wall=%5ss rss=%6s GiB  dense=%sGiB fill=%sGiB (sum=%s/%s)  swap %sM->%sM\n" \
         "$m" "$st" "$((t1-t0))" "$rss" "${dense:-?}" "${fill:-?}" \
         "$(( ${dense:-0} + ${fill:-0} ))" "$CEIL_GIB" "$s0" "$s1"
  # rc=143 is earlyoom, and it is silent. Say so, because "Terminated" with no output is
  # otherwise indistinguishable from a run that simply produced nothing.
  [[ "$rc" == 143 ]] && echo "      ^ SIGTERM: earlyoom killed it. The adaptive guard did not brake in time."
done
echo
echo "CEILINGVERIFY rc=$fail"
exit "$fail"
