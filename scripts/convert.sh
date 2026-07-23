#!/usr/bin/env bash
# Convert a model's source checkpoint into a coli container, per the registry.
#   Usage: scripts/convert.sh <model>
# Reads source/container paths + the model's COLI_* convert flags from models.toml,
# so `coli convert` is invoked identically every time without hand-typed paths.
set -euo pipefail
source "$(dirname "$0")/lib.sh"
[[ $# -ge 1 ]] || die "usage: scripts/convert.sh <model>   (models: $(model_names))"

load_model "$1"
need_coli
need_source

echo "[convert] $COLI_MODEL ($ARCH)"
echo "[convert] source:    $SOURCE"
echo "[convert] container: $CONTAINER"
echo "[convert] flags:     ${CONVERT_ENV:-<defaults>}"
if [[ -d "$CONTAINER" ]]; then
  echo "[convert] NOTE: container already exists and will be overwritten by coli convert." >&2
fi

# shellcheck disable=SC2086  # CONVERT_ENV is intentionally word-split into env assignments
env $CONVERT_ENV "$COLI_BIN" convert "$SOURCE" "$CONTAINER"
echo "[convert] done → $CONTAINER"
