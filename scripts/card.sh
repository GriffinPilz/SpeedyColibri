#!/usr/bin/env bash
# Regenerate a model card's measurement rows from a real benchmark run.
#
#   Usage: scripts/card.sh <model> [--upload]
#          scripts/card.sh <model> --from <bench-log> [--upload]
#          scripts/card.sh <model> --suites "prefill decode" --allow-partial [--upload]
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

usage() { die "usage: scripts/card.sh <model> [--from <bench-log>] [--suites <list>] [--allow-partial] [--upload]"; }
[[ $# -ge 1 ]] || usage
MODEL="$1"; shift
UPLOAD=0; FROM=""; SUITES="prefill decode serve"; ALLOW_PARTIAL=0
while [[ $# -gt 0 ]]; do
  case "$1" in
    --upload)        UPLOAD=1; shift ;;
    --from)          FROM="${2:?--from needs a file}"; shift 2 ;;
    --suites)        SUITES="${2:?--suites needs a list, e.g. \"prefill decode\"}"; shift 2 ;;
    --allow-partial) ALLOW_PARTIAL=1; shift ;;
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
  echo "[card] benchmarking $MODEL ($SUITES) — this is the slow part"
  log=$(for s in $SUITES; do "$HARNESS_DIR/bench.sh" "$MODEL" "$s" 2>&1; done)
fi

# One pattern per figure, each anchored on the line bench.sh actually prints.
grab() { grep -oP "$1" <<<"$log" | tail -1 || true; }
PF_MS=$(grab 'MEDIAN: prefill=\K[0-9.]+')
# `expert-load` is a MEDIAN, so it is frequently fractional — nemotron prints `2.0ms`, m3
# `13803.0ms`. The original `[0-9]+ms` matched only an integer, so this figure silently
# failed to parse on most of the fleet and the script died claiming the suite had failed.
# It went unnoticed because the one model it was first used on, maple, prints `0ms`.
PF_TPS=$(grab 'MEDIAN: prefill=[0-9.]+ms\s+expert-load=[0-9.]+ms\s+\(\K[0-9.]+')
DEC_FWD=$(grab 'MEDIAN across reps: \K[0-9.]+')
DEC_E2E=$(grab 'END-TO-END:\s+\K[0-9.]+')
SRV=$(grab 'median\s+: \K[0-9.]+')

# A row is rewritten only when EVERY figure it needs parsed; otherwise it is left exactly
# as published and reported. Blanking a row was always the thing to avoid — "** tok/s"
# publishes cleanly and says nothing — but the original spelled that as "die on any missing
# field", which made two legitimate cases impossible:
#
#   * kimi-k3 has never had a serving figure ("not run — the suite takes hours"). Its card
#     could not be regenerated AT ALL, so the one tool that keeps cards honest was unusable
#     on the model whose numbers are hardest to re-measure.
#   * a single crashed suite threw away the other two suites' fresh numbers.
#
# Leaving a row untouched is safe; publishing a HALF-refreshed card silently is not. So a
# partial run still refuses `--upload` unless `--allow-partial` says the omission is
# deliberate. That keeps the failure loud while letting k3 through on purpose.
rows_done=(); rows_left=()
have() { for v in "$@"; do [[ -n "${!v}" ]] || return 1; done; }
have PF_MS PF_TPS  && rows_done+=(prefill) || rows_left+=(prefill)
have DEC_FWD DEC_E2E && rows_done+=(decode)  || rows_left+=(decode)
have SRV           && rows_done+=(serving) || rows_left+=(serving)
[[ ${#rows_done[@]} -gt 0 ]] || die "no figures parsed from the bench output — did every suite fail?"
PF_S=""; [[ -n "$PF_MS" ]] && PF_S=$(awk "BEGIN{printf \"%.1f\", $PF_MS/1000}")
# One decimal everywhere, matching the cards' existing style. bench.sh reports two on some
# figures and one on others; publishing that inconsistency would make diffs between runs
# look like precision changes rather than measurement changes.
# An `if` rather than `[[ … ]] && awk`: the && form returns 1 on an empty value, and under
# `set -e` that kills the script mid-assignment with no message at all.
r1() { if [[ -n "$1" ]]; then awk "BEGIN{printf \"%.1f\", $1}"; fi; }
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
  /^\| prefill \|/ && pf  != "" { printf "| prefill | **%s tok/s** (%s s) |\n", pf, pfs; next }
  /^\| decode \|/  && de2e != "" { printf "| decode | **%s tok/s** end-to-end (%s forward-only) |\n", de2e, dfwd; next }
  /^\| serving \|/ && srv != "" { printf "| serving | **%s tok/s** median, 12 diverse prompts over HTTP |\n", srv; next }
  { print }
' "$CARD" > "$tmp"
mv "$tmp" "$CARD"

echo "[card] $MODEL rows now (rewrote: ${rows_done[*]}):"
grep -E '^\| (prefill|decode|serving) \|' "$CARD" | sed 's/^/    /'
if [[ ${#rows_left[@]} -gt 0 ]]; then
  echo "[card] LEFT AS PUBLISHED — no figure parsed for: ${rows_left[*]}"
  echo "[card]   (deliberate? pass --allow-partial. otherwise a suite failed — check the log.)"
fi

# ---- the hard half: prose that quotes the old numbers -------------------------------
stale=0
while read -r line; do
  n=$(grep -oP '\*\*\K[0-9.]+(?= tok/s)' <<<"$line" || true)
  [[ -n "$n" ]] || continue
  # Only a row we REWROTE can have superseded its own number. A row left as published still
  # holds the current figure, so prose quoting it is correct — flagging it would send you
  # hunting for staleness that is not there, and (with --allow-partial) block the upload.
  row=$(grep -oP '^\| \K(prefill|decode|serving)' <<<"$line" || true)
  [[ " ${rows_done[*]} " == *" $row "* ]] || continue
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
  [[ ${#rows_left[@]} -eq 0 || "$ALLOW_PARTIAL" == 1 ]] || \
    die "refusing to publish: no figure for ${rows_left[*]}, so that row would keep an unverified number while the others move. Re-run the missing suite, or pass --allow-partial if leaving it is deliberate."
  cli=$(hf_cli) || die "no hf CLI found (tried PATH and ~/.local/bin)"
  # `hf upload` CREATES the repo when it does not exist. This script's job is to refresh a
  # card on a repo that is already published, so a typo'd or stale `hf_repo` should fail —
  # not quietly stand up a new public model page. Found the hard way: a test pointed at a
  # deliberately nonexistent repo to prove the upload path was reached, and it published one.
  curl -fsS -o /dev/null "https://huggingface.co/api/models/$HF_REPO" 2>/dev/null ||
    die "no such published repo '$HF_REPO' — refusing to CREATE it. card.sh only updates an existing card; check hf_repo in scripts/models.toml."
  ( cd "$CONTAINER" && "$cli" upload "$HF_REPO" README.md README.md \
      --commit-message "card: measurements refreshed from a bench run" )
else
  echo "[card] local only. Review it, then re-run with --upload to publish to ${HF_REPO:-<no hf_repo>}."
fi
