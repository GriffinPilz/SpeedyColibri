#!/usr/bin/env bash
# One entry point covering a model end to end: convert → build → bench.
#   Usage: scripts/pipeline.sh <model> [phases]
#   phases: comma list of convert,build,bench   (default: build,bench)
#           convert is opt-in — it's slow and overwrites the container.
#   BENCH_SUITE selects the bench battery (default: all).
set -euo pipefail
source "$(dirname "$0")/lib.sh"
[[ $# -ge 1 ]] || die "usage: scripts/pipeline.sh <model> [convert,build,bench]   (models: $(model_names))"

MODEL="$1"; PHASES="${2:-build,bench}"
load_model "$MODEL"
echo "#### pipeline: $COLI_MODEL ($ARCH)   phases=$PHASES ####"

IFS=',' read -ra PH <<<"$PHASES"
for p in "${PH[@]}"; do
  case "$p" in
    convert) "$HARNESS_DIR/convert.sh" "$MODEL" ;;
    build)   "$HARNESS_DIR/build.sh" ;;
    bench)   "$HARNESS_DIR/bench.sh" "$MODEL" "${BENCH_SUITE:-all}" ;;
    *)       die "unknown phase '$p' (convert|build|bench)" ;;
  esac
done
echo "#### pipeline $COLI_MODEL done ####"
