#!/usr/bin/env bash
# Shared helpers for the model-parameterized harness (convert/build/bench/pipeline).
# Source this, then `load_model <name>` to pull a registry entry into the environment.
HARNESS_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$HARNESS_DIR/.." && pwd)"
PY="${PYTHON:-python3}"
COLI_BIN="${COLI_BIN:-$REPO_ROOT/target/release/coli}"

die() { echo "harness: $*" >&2; exit 1; }

# Resolve a model through the registry and export its fields:
#   COLI_MODEL ARCH CONTAINER SOURCE PROMPT_TOKENS PROMPT_SPEC CONVERT_ENV NOTES
load_model() {
  local out
  out="$("$PY" "$HARNESS_DIR/model.py" env "$1")" || exit 1
  eval "$out"
}

model_names() { "$PY" "$HARNESS_DIR/model.py" list | awk '{print $1}' | paste -sd, -; }

need_coli()      { [[ -x "$COLI_BIN" ]] || die "coli not found at $COLI_BIN — run scripts/build.sh (or set COLI_BIN)"; }
need_container() { [[ -d "$CONTAINER" ]] || die "container missing: $CONTAINER (model '$COLI_MODEL' not materialized on this host)"; }
need_source()    { [[ -d "$SOURCE" ]]    || die "source missing: $SOURCE (model '$COLI_MODEL')"; }

# Reset memory state between benchmark arms.
#
# Page-cache carry-over is the biggest source of contamination in this repo's numbers:
# the SAME configuration measured 2.27 tok/s early in a sequence and 0.23 tok/s late,
# purely from what the previous arm left warm. An A/B that does not reset is partly
# measuring the order of its arms.
#
# `coli dropcache` uses posix_fadvise(DONTNEED) — no root, and only this model's pages.
# It also reports swap, which fadvise CANNOT reclaim: if a run has driven the box into
# swap, every later measurement is degraded until someone with root runs
# `swapoff -a && swapon -a`. Better to see that in the log than to spend a day
# re-measuring a poisoned box (which is exactly what happened before this existed).
#
# Call it BEFORE each arm (start from a known state) and AFTER (leave the box clean for
# whatever runs next).
mem_reset() {
  [[ -x "$COLI_BIN" && -d "$CONTAINER" ]] || return 0
  "$COLI_BIN" dropcache "$CONTAINER" 2>/dev/null | sed 's/^/    /'
  # Let the kernel actually complete the reclaim before the next arm starts timing.
  sleep "${MEM_RESET_SETTLE:-3}"
}

# Median of numeric args (integers or floats).
median() {
  printf "%s\n" "$@" | sort -n | awk '
    {a[NR]=$1}
    END{ if(NR==0){print "NA"} else if(NR%2){print a[(NR+1)/2]} else {printf "%.1f",(a[NR/2]+a[NR/2+1])/2} }'
}

# Token-identity gate: all args must be equal and non-empty. Prints PASS/FAIL, returns nonzero on FAIL.
gate_tokens() {
  local first="$1" t
  [[ -n "$first" ]] || { echo "  token-gate: FAIL (no tokens captured)"; return 1; }
  for t in "$@"; do
    [[ "$t" == "$first" ]] || { echo "  token-gate: FAIL (got $t vs $first — outputs diverged!)"; return 1; }
  done
  echo "  token-gate: PASS (all runs → $first)"
}
