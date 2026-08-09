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
# Strict: measurement paths must never turn into a download (see fetch_container below).
# The message names the fix when the registry knows one, so "not materialized" is not a
# dead end.
need_container() {
  [[ -d "$CONTAINER" ]] && return 0
  local hint=""
  [[ -n "${HF_REPO:-}" ]] && hint=" — fetch it with: scripts/fetch.sh $COLI_MODEL"
  die "container missing: $CONTAINER (model '$COLI_MODEL' not materialized on this host)$hint"
}
need_source()    { [[ -d "$SOURCE" ]]    || die "source missing: $SOURCE (model '$COLI_MODEL')"; }

# ---- materializing a container from the Hub -----------------------------------------
#
# `hf_repo` in the registry is the CONTAINER, not the upstream checkpoint, so this is a
# download and never a conversion. The docker path has done this since the entrypoint was
# written; the bare-metal scripts did not, so `scripts/serve.sh <model>` on a fresh host hit
# `need_container` and stopped at "container missing" with no route forward — while the
# README told the reader to convert a model that was already published ready-to-run.
#
# The CLI is looked up in $HOME/.local/bin too. `hf` installs there by default and that
# directory is NOT on PATH under a non-interactive ssh (login shells source the profile that
# adds it; `ssh host 'cmd'` does not). `command -v hf` therefore fails on exactly the hosts
# this is most likely to run on.
hf_cli() {
  local c
  c=$(command -v hf || command -v huggingface-cli) && { echo "$c"; return 0; }
  for c in "$HOME/.local/bin/hf" "$HOME/.local/bin/huggingface-cli"; do
    [[ -x "$c" ]] && { echo "$c"; return 0; }
  done
  return 1
}

# Download this model's container into $CONTAINER. Idempotent: `hf download` fetches only
# what is missing or the wrong size, so re-running it verifies an existing directory instead
# of re-transferring it.
#
# That holds for a container this host BUILT as well as one it downloaded, which is the case
# that matters — running this against a 145 GB convert-built container would be expensive if
# it did not. Measured 2026-08-08: a 55.7 MB shard placed by hand, with no
# `.cache/huggingface` metadata beside it, re-downloaded in 0 s and 0 bytes off the wire.
# Size match alone is enough for `hf` to skip it.
fetch_container() {
  local cli
  [[ -n "${HF_REPO:-}" ]] || die "no hf_repo for '$COLI_MODEL' in scripts/models.toml — nothing to fetch"
  cli=$(hf_cli) || die "no hf CLI found (tried PATH and ~/.local/bin) — pip install huggingface_hub"
  # A container download is hundreds of GB on most of these models. Running one against the
  # same NVMe a benchmark is reading does not merely add noise to that benchmark, it changes
  # the quantity being measured — see the note on bench_lock_acquire above.
  if bench_lock_held; then
    die "a measurement is running (lock $BENCH_LOCK) — refusing to start a bulk download beside it"
  fi
  echo "[fetch] $COLI_MODEL: $HF_REPO -> $CONTAINER"
  "$cli" download "$HF_REPO" --local-dir "$CONTAINER" >&2 || die "hf download failed for $HF_REPO"
  [[ -f "$CONTAINER/config.json" ]] || die "downloaded $HF_REPO but $CONTAINER has no config.json"
}

# Container if present, download if not. Callers that SERVE use this; callers that MEASURE
# use `need_container`, so a benchmark can never silently turn into a download.
ensure_container() {
  [[ -d "$CONTAINER" && -f "$CONTAINER/config.json" ]] && return 0
  [[ -n "${HF_REPO:-}" ]] || die "container missing: $CONTAINER, and '$COLI_MODEL' has no hf_repo to fetch from"
  fetch_container
}

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
#
# It also drops the PREVIOUSLY-RESET model's pages, tracked in `$LAST_CONTAINER_FILE`.
# `fadvise` only touches the containers named, so dropping just the incoming model leaves
# the outgoing one resident and the new model must evict it while streaming. Measured
# 2026-08-02 on K3: 95 MB/s at the device and still unfinished at 69 min, against 456 s for
# the very next run once those pages were gone — ~30x, from nothing but whose pages were
# resident.
#
# Doing it HERE rather than at each call site is the point: `scripts/bench.sh` worked around
# this with a discarded warmup run (a whole extra model-load per model), and every ad-hoc
# script has to remember. Anything that calls `mem_reset` now gets it for free — including
# the one-liner I ran an hour after writing the warning, and walked straight into.
LAST_CONTAINER_FILE="${LAST_CONTAINER_FILE:-/tmp/colibri-last-container}"

mem_reset() {
  [[ -x "$COLI_BIN" && -d "$CONTAINER" ]] || return 0
  local prev=""
  [[ -f "$LAST_CONTAINER_FILE" ]] && prev=$(cat "$LAST_CONTAINER_FILE" 2>/dev/null)
  # `|| true` is load-bearing: callers run under `set -euo pipefail`, so a non-zero
  # dropcache takes the whole suite down between printing its header and its first result.
  # That is exactly what happened — an argv indexing bug made dropcache exit 1, and every
  # bench.sh run produced no output while looking like it had merely been slow.
  if [[ -n "$prev" && "$prev" != "$CONTAINER" && -d "$prev" ]]; then
    "$COLI_BIN" dropcache "$CONTAINER" "$prev" 2>/dev/null | sed 's/^/    /' || true
  else
    "$COLI_BIN" dropcache "$CONTAINER" 2>/dev/null | sed 's/^/    /' || true
  fi
  printf '%s' "$CONTAINER" >"$LAST_CONTAINER_FILE" 2>/dev/null || true
  # Let the kernel actually complete the reclaim before the next arm starts timing.
  sleep "${MEM_RESET_SETTLE:-3}"
}

# Declare that a measurement is in progress, so anything that would compete for the drive
# or the network refuses to start.
#
# Everything this harness measures is sensitive to the NVMe: expert-load is the bulk of a
# decode step, and prefill reads hundreds of GB. A concurrent bulk transfer — a Hugging Face
# container upload is ~1.85 TB across the fleet — does not merely add noise, it changes the
# quantity being measured. Two independent stale benchmark processes contending on this box
# already produced an hour of numbers that were silently measuring each other.
#
# The lock is advisory and self-cleaning: the trap fires on normal exit and on INT/TERM, and
# a stale lock from a killed run is detected by checking whether the recorded PID is alive.
BENCH_LOCK="${BENCH_LOCK:-/tmp/colibri-bench.lock}"

bench_lock_acquire() {
  if [[ -f "$BENCH_LOCK" ]]; then
    local pid; pid=$(cat "$BENCH_LOCK" 2>/dev/null)
    if [[ -n "$pid" ]] && kill -0 "$pid" 2>/dev/null; then
      echo "harness: a measurement is already running (pid $pid) — refusing to start a second" >&2
      exit 1
    fi
    echo "harness: clearing stale lock from dead pid ${pid:-?}" >&2
  fi
  echo $$ > "$BENCH_LOCK"
  # EXIT does the cleanup; INT and TERM must additionally STOP.
  #
  # These were one handler — `trap 'rm -f "$BENCH_LOCK"' EXIT INT TERM` — which removed the
  # lock and then RETURNED, so the script ignored the signal and kept measuring with its own
  # guard gone. A TERM'd sweep silently continued to its next point, and the deleted lock
  # file read as proof it had died. The next run's "is anything already measuring?" check
  # then passed, which is precisely how two measurements end up contending (1.65x) while
  # both look clean.
  #
  # `exit` from the INT/TERM handler re-triggers EXIT, which does the rm — so the cleanup is
  # still in exactly one place. Note bash defers a trap until the current foreground command
  # finishes, so a signal lands after the in-flight point completes, not mid-point.
  trap 'rm -f "$BENCH_LOCK"' EXIT
  trap 'echo "harness: interrupted (SIGINT) — stopping" >&2; exit 130' INT
  trap 'echo "harness: terminated (SIGTERM) — stopping" >&2; exit 143' TERM
}

# True while a measurement holds the lock. For non-bench tooling (uploads, fetches) to test.
bench_lock_held() {
  local pid
  [[ -f "$BENCH_LOCK" ]] || return 1
  pid=$(cat "$BENCH_LOCK" 2>/dev/null)
  [[ -n "$pid" ]] && kill -0 "$pid" 2>/dev/null
}

# Median of numeric args (integers or floats).
median() {
  printf "%s\n" "$@" | sort -n | awk '
    {a[NR]=$1}
    END{ if(NR==0){print "NA"} else if(NR%2){print a[(NR+1)/2]}
         # EVEN count averages the middle pair, and %.1f throws away real precision below
         # 1 tok/s: two K3 decode reps of 0.35 printed as "0.3", a 14% misreport that reads
         # as a regression against the recorded ~0.4. GLM (~0.9) is in the same range. The
         # ODD path prints verbatim, which is why per-run medians (23 rates) looked right
         # and only the 2-rep median was wrong. Two decimals below 1, unchanged above.
         else {v=(a[NR/2]+a[NR/2+1])/2; printf (v<1)?"%.2f":"%.1f", v} }'
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
