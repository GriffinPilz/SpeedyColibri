#!/usr/bin/env bash
# Model-parameterized benchmark battery — the same measurements for every model.
#   Usage: scripts/bench.sh <model> [suite]
#   suites: prefill | decode | batch | serve | all      (default: all)
#
# Every gen-based suite runs a token-identity gate: repeated runs (or the batched
# path) must emit the SAME tokens, so a "faster" number that changed the output
# fails loudly instead of being reported as a win. Cold expert-load cache variance
# is ~±40% run-to-run on these boxes, so suites report a MEDIAN over reps (plus a
# discarded warmup) and lean on the cache-independent phase timers where possible.
#
# Knobs: BENCH_REPS (default 3), DECODE_NGEN (24), BATCH_SIZES ("1 4 8 16 32"),
#        BATCH_NGEN (10), SERVE_PORT (8080), SERVE_NTOK (32), COLI_BIN, COLI_MODELS_ROOT.
set -euo pipefail
source "$(dirname "$0")/lib.sh"
[[ $# -ge 1 ]] || die "usage: scripts/bench.sh <model> [prefill|decode|batch|serve|all]   (models: $(model_names))"

MODEL="$1"; SUITE="${2:-all}"
load_model "$MODEL"; need_coli; need_container
cd "$REPO_ROOT"

REPS="${BENCH_REPS:-3}"
NTOK_PROMPT="$(wc -w <<<"$PROMPT_TOKENS")"

echo "======================================================================"
echo " bench: $COLI_MODEL ($ARCH)   suite=$SUITE   reps=$REPS"
echo " container: $CONTAINER"
echo " prompt:    $PROMPT_SPEC  ($NTOK_PROMPT tokens)"
echo "======================================================================"

# field "<text>" "<pcre with \K>" — first match, or empty. MUST stay tolerant of a
# no-match: `set -e` + `pipefail` would otherwise turn a single missing field into a
# silent whole-suite abort (grep exits 1 on no match → the $(...) assignment trips -e).
field() { grep -oP "$2" <<<"$1" | head -1 || true; }

# ---- prefill: NGEN=1, profile, median phase breakdown, token-identity ----------
suite_prefill() {
  echo; echo "── prefill (NGEN=1, profile; $REPS reps + warmup) ──"
  local prefills=() eloads=() toks=()
  COLI_NGEN=1 COLI_PROFILE=1 COLI_TIMING=1 "$COLI_BIN" gen "$CONTAINER" $PROMPT_TOKENS >/dev/null 2>&1 || true
  local i out pf el ff pj at mo tok
  for i in $(seq 1 "$REPS"); do
    out=$(COLI_NGEN=1 COLI_PROFILE=1 COLI_TIMING=1 "$COLI_BIN" gen "$CONTAINER" $PROMPT_TOKENS 2>&1)
    pf=$(field "$out" 'prefill \d+ tok: \K[0-9.]+')
    el=$(field "$out" 'expert-load \K[0-9]+')
    ff=$(field "$out" 'gpu-ffn\(\+sync\) \K[0-9]+')
    pj=$(field "$out" 'proj \K[0-9]+')
    at=$(field "$out" 'attn \K[0-9]+')
    mo=$(field "$out" 'moe \K[0-9]+')
    tok=$(field "$out" 'generated \(1 tok\): \K\[[0-9]+\]')
    prefills+=("$pf"); eloads+=("$el"); toks+=("$tok")
    printf "  run %d: prefill=%8sms  attn=%6sms  moe=%6sms (expert-load=%6sms gpu-ffn=%6sms)  proj=%6sms  tok=%s\n" \
      "$i" "$pf" "$at" "$mo" "$el" "$ff" "$pj" "$tok"
  done
  printf "  MEDIAN: prefill=%sms  expert-load=%sms  (%.1f tok/s)\n" \
    "$(median "${prefills[@]}")" "$(median "${eloads[@]}")" \
    "$(awk -v n="$NTOK_PROMPT" -v ms="$(median "${prefills[@]}")" 'BEGIN{printf (ms>0)? n/(ms/1000) : 0}')"
  gate_tokens "${toks[@]}"
}

# ---- decode: warm steady-state tok/s + decode phase breakdown ------------------
suite_decode() {
  local ngen="${DECODE_NGEN:-24}"
  echo; echo "── decode (NGEN=$ngen warm steady-state; $REPS reps) ──"
  # coli emits one `[timing] decode tok N: X ms (Y tok/s)` line per generated token.
  # Steady-state = MEDIAN of the per-token rates (robust to the cold first tokens and
  # the ±40% expert-load cache spikes); `best` = the compute-floor token.
  local meds=() toks=()
  COLI_NGEN="$ngen" COLI_TIMING=1 "$COLI_BIN" gen "$CONTAINER" $PROMPT_TOKENS >/dev/null 2>&1 || true
  local i out med best tok
  for i in $(seq 1 "$REPS"); do
    out=$(COLI_NGEN="$ngen" COLI_TIMING=1 "$COLI_BIN" gen "$CONTAINER" $PROMPT_TOKENS 2>&1)
    local rates_all=()
    mapfile -t rates_all < <(grep -oP 'decode tok \d+: [0-9.]+ ms \(\K[0-9.]+' <<<"$out")
    med=$(median "${rates_all[@]}")
    best=$(printf "%s\n" "${rates_all[@]}" | sort -n | tail -1)
    tok=$(field "$out" 'generated \(\d+ tok\): \K\[[0-9, ]+\]')
    meds+=("$med"); toks+=("$tok")
    printf "  run %d: decode median=%s tok/s  best=%s tok/s  (%d steps)\n" "$i" "$med" "${best:-NA}" "${#rates_all[@]}"
  done
  printf "  MEDIAN across reps: %s tok/s\n" "$(median "${meds[@]}")"
  gate_tokens "${toks[@]}"
}

# ---- batch: aggregate tok/s vs batch size + a token-identity verify -------------
suite_batch() {
  local sizes="${BATCH_SIZES:-1 4 8 16 32}" ngen="${BATCH_NGEN:-10}"
  echo; echo "── batch (genbatch aggregate tok/s vs B; ngen=$ngen) ──"
  echo "  verify (B=8, token-identity vs single-sequence):"
  local vout
  vout=$(COLI_BATCH_VERIFY=1 "$COLI_BIN" genbatch "$CONTAINER" 8 8 $PROMPT_TOKENS 2>&1 | grep -i "VERIFY" || true)
  echo "    ${vout:-<no VERIFY line — check genbatch output>}"
  local b out rate step
  for b in $sizes; do
    out=$("$COLI_BIN" genbatch "$CONTAINER" "$b" "$ngen" $PROMPT_TOKENS 2>&1)
    rate=$(field "$out" 'aggregate \K[0-9.]+')
    step=$(field "$out" 'steady-state \K[0-9.]+')
    printf "  B=%-3s  %8s ms/step   aggregate %s tok/s\n" "$b" "$step" "$rate"
  done
}

# ---- serve: real HTTP throughput over diverse prompts (bench_serve.py) ----------
suite_serve() {
  local port="${SERVE_PORT:-8080}" ntok="${SERVE_NTOK:-32}"
  echo; echo "── serve (bench_serve.py, diverse NL prompts, HTTP :$port) ──"
  local log="/tmp/coli_serve_${MODEL}_$$.log"
  COLI_SERVE_MODEL="$MODEL" COLI_PORT="$port" "$COLI_BIN" serve "$CONTAINER" "$port" >"$log" 2>&1 &
  local pid=$!
  # shellcheck disable=SC2064
  trap "kill $pid 2>/dev/null || true" RETURN
  echo "  waiting for serve (pid $pid) to load + listen…"
  # coli serve loads the ENTIRE model before it binds, so the port is refused until
  # loading finishes. Poll for a real HTTP code; "000" = not-listening-yet, keep waiting.
  # (curl -w already prints 000 on connection-refused — do NOT also `echo 000`, or the
  # code becomes "000000" != "000" and the loop falsely declares serve up on poll 1.)
  local i code up=0
  for i in $(seq 1 "${SERVE_WAIT:-180}"); do
    kill -0 "$pid" 2>/dev/null || { echo "  serve died during load — see $log" >&2; tail -5 "$log" >&2; return 1; }
    code=$(curl -s -o /dev/null -w '%{http_code}' --max-time 2 "http://127.0.0.1:$port/" 2>/dev/null || true)
    [[ -n "$code" && "$code" != "000" ]] && { up=1; break; }
    sleep 1
  done
  [[ "$up" == 1 ]] || { echo "  serve never answered on :$port — see $log" >&2; return 1; }
  COLI_SERVE_MODEL="$MODEL" "$PY" "$HARNESS_DIR/bench_serve.py" "127.0.0.1:$port" "$ntok"
}

case "$SUITE" in
  prefill) suite_prefill ;;
  decode)  suite_decode ;;
  batch)   suite_batch ;;
  serve)   suite_serve ;;
  all)     suite_prefill; suite_decode; suite_batch ;;   # serve is opt-in (spins a server)
  *) die "unknown suite '$SUITE' (prefill|decode|batch|serve|all)" ;;
esac
echo; echo "==== bench $COLI_MODEL/$SUITE done ===="
