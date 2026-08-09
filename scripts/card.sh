#!/usr/bin/env bash
# Regenerate a model card's measurement rows from a real benchmark run.
#
#   Usage: scripts/card.sh <model> [--upload]
#          scripts/card.sh <model> --from <bench-log> [--upload]
#
# Why this exists: nothing connected the published cards to the numbers, so every perf merge
# silently rotted them. maple's card went stale TWICE in one day — once when the accept loop
# stopped napping (serve 70.5 -> 91.1) and again hours later when the expert path learned to
# group a prefill (prefill 160.2 -> 230.0, serve -> 109.0). Both times the fix was hand-
# editing a published page, which is exactly the kind of thing that gets forgotten once.
#
# It rewrites ONLY the three measurement rows, and it does not publish unless you say so:
# without `--upload` it edits the container's local README.md and prints the diff, so the
# change can be read before it becomes a public page.
#
# It also DETECTS the harder half of the problem. The rows are easy; the trap is prose that
# quotes the same numbers — maple's card explained its serve figure as `52 ms + 9.36 ms x 32
# tok`, which stayed wrong after the table was right. So every number this script replaces
# is grepped for in the rest of the card, and any surviving mention is reported. It will not
# rewrite that prose (it is argument, not data) but it will refuse to let it pass unnoticed.
set -euo pipefail
source "$(dirname "$0")/lib.sh"

usage() { die "usage: scripts/card.sh <model> [--from <bench-log>] [--upload]"; }
[[ $# -ge 1 ]] || usage
MODEL="$1"; shift
UPLOAD=0; FROM=""
while [[ $# -gt 0 ]]; do
  case "$1" in
    --upload) UPLOAD=1; shift ;;
    --from)   FROM="${2:?--from needs a file}"; shift 2 ;;
    *) usage ;;
  esac
done

load_model "$MODEL"
CARD="$CONTAINER/README.md"
[[ -f "$CARD" ]] || die "no card at $CARD (the container has no README.md to update)"

# ---- gather the numbers ------------------------------------------------------------
if [[ -n "$FROM" ]]; then
  [[ -f "$FROM" ]] || die "no such bench log: $FROM"
  log=$(cat "$FROM")
else
  need_coli; need_container
  echo "[card] benchmarking $MODEL (prefill, decode, serve) — this is the slow part"
  log=$(for s in prefill decode serve; do "$HARNESS_DIR/bench.sh" "$MODEL" "$s" 2>&1; done)
fi

# One pattern per figure, each anchored on the line bench.sh actually prints. A missing
# field is fatal rather than blank: a card row reading "** tok/s" would publish cleanly.
grab() { grep -oP "$1" <<<"$log" | tail -1 || true; }
PF_MS=$(grab 'MEDIAN: prefill=\K[0-9.]+')
PF_TPS=$(grab 'MEDIAN: prefill=[0-9.]+ms\s+expert-load=[0-9]+ms\s+\(\K[0-9.]+')
DEC_FWD=$(grab 'MEDIAN across reps: \K[0-9.]+')
DEC_E2E=$(grab 'END-TO-END:\s+\K[0-9.]+')
SRV=$(grab 'median\s+: \K[0-9.]+')
for v in PF_MS PF_TPS DEC_FWD DEC_E2E SRV; do
  [[ -n "${!v}" ]] || die "could not extract $v from the bench output — did a suite fail?"
done
PF_S=$(awk "BEGIN{printf \"%.1f\", $PF_MS/1000}")
# One decimal everywhere, matching the cards' existing style. bench.sh reports two on some
# figures and one on others; publishing that inconsistency would make diffs between runs
# look like precision changes rather than measurement changes.
r1() { awk "BEGIN{printf \"%.1f\", $1}"; }
PF_TPS=$(r1 "$PF_TPS"); DEC_FWD=$(r1 "$DEC_FWD"); DEC_E2E=$(r1 "$DEC_E2E"); SRV=$(r1 "$SRV")

# ---- rewrite the three rows --------------------------------------------------------
#
# Canonical decode row for EVERY model, not just the ones that had it. `coli gen` and
# bench.sh print the pair fleet-wide now, and a lone forward-only figure is the omission
# that made maple look 39% faster than a caller experiences.
#
# THE BASELINE IS THE PUBLISHED CARD, NOT THE LOCAL ONE, and that is load-bearing. Taking it
# locally makes the staleness check below erode itself: the first run rewrites the rows, so a
# second run sees nothing changed and publishes prose that is still wrong — which is the exact
# failure this script exists to prevent. Caught by testing the refusal path, where `--upload`
# sailed through on the second invocation.
old_rows=""
if [[ -n "${HF_REPO:-}" ]]; then
  old_rows=$(curl -fsSL --max-time 30 \
    "https://huggingface.co/$HF_REPO/raw/main/README.md" 2>/dev/null |
    grep -E '^\| (prefill|decode|serving) \|' || true)
fi
if [[ -z "$old_rows" ]]; then
  echo "[card] note: could not read the published card; falling back to the local one as the" >&2
  echo "[card] baseline, so re-running before fixing prose will under-report staleness." >&2
  old_rows=$(grep -E '^\| (prefill|decode|serving) \|' "$CARD" || true)
fi
[[ -n "$old_rows" ]] || die "no measurement rows in $CARD — expected '| prefill | …' etc."

tmp=$(mktemp)
awk -v pf="$PF_TPS" -v pfs="$PF_S" -v dfwd="$DEC_FWD" -v de2e="$DEC_E2E" -v srv="$SRV" '
  /^\| prefill \|/ { printf "| prefill | **%s tok/s** (%s s) |\n", pf, pfs; next }
  /^\| decode \|/  { printf "| decode | **%s tok/s** end-to-end (%s forward-only) |\n", de2e, dfwd; next }
  /^\| serving \|/ { printf "| serving | **%s tok/s** median, 12 diverse prompts over HTTP |\n", srv; next }
  { print }
' "$CARD" > "$tmp"
mv "$tmp" "$CARD"

echo "[card] $MODEL rows now:"
grep -E '^\| (prefill|decode|serving) \|' "$CARD" | sed 's/^/    /'

# ---- the hard half: prose that quotes the old numbers -------------------------------
stale=0
while read -r line; do
  n=$(grep -oP '\*\*\K[0-9.]+(?= tok/s)' <<<"$line" || true)
  [[ -n "$n" ]] || continue
  # Skip a figure that is still current — only the superseded ones matter.
  case "$n" in "$PF_TPS"|"$DEC_E2E"|"$DEC_FWD"|"$SRV") continue ;; esac
  # `<!--hist-->` marks a line that cites an old number ON PURPOSE. This repo records
  # superseded figures constantly — "it read X until Y" is how a correction is made legible —
  # and without an opt-out the check fires on exactly the prose that is doing the right
  # thing. Found by using it: the first real correction it gated was blocked by its own
  # explanation of the correction.
  hits=$(grep -nF "$n" "$CARD" |
         grep -vE '^\s*[0-9]+:\| (prefill|decode|serving) \|' |
         grep -vF '<!--hist-->' || true)
  if [[ -n "$hits" ]]; then
    [[ "$stale" == 0 ]] && echo "[card] STALE PROSE — these superseded numbers still appear outside the table:"
    stale=1
    echo "$hits" | sed "s/^/    ($n) /"
  fi
done <<<"$old_rows"
[[ "$stale" == 0 ]] && echo "[card] no superseded figures found elsewhere in the card"

# ---- publish, only on request -------------------------------------------------------
if [[ "$UPLOAD" == 1 ]]; then
  [[ -n "${HF_REPO:-}" ]] || die "no hf_repo for '$MODEL' — nothing to upload to"
  [[ "$stale" == 0 ]] || die "refusing to publish: superseded numbers are still in the prose above. Fix them, then re-run with --upload."
  cli=$(hf_cli) || die "no hf CLI found (tried PATH and ~/.local/bin)"
  ( cd "$CONTAINER" && "$cli" upload "$HF_REPO" README.md README.md \
      --commit-message "card: measurements refreshed from a bench run" )
else
  echo "[card] local only. Review it, then re-run with --upload to publish to ${HF_REPO:-<no hf_repo>}."
fi
