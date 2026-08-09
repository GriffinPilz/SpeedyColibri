#!/usr/bin/env bash
# Download a model's ready-to-run CONTAINER from the Hub — no conversion step.
#
#   Usage: scripts/fetch.sh <model>          # e.g. maple-preview, deepseek-v4-flash
#          scripts/fetch.sh --all            # every registered model with an hf_repo
#
# `hf_repo` in scripts/models.toml is the CONTAINER, so this downloads something the engine
# can load directly. Converting from the upstream checkpoint is the other route and is only
# worth it if you want to change how the model is packed — see docs/DEVELOPMENT.md.
#
# Idempotent: `hf download` transfers only files that are missing or the wrong size, so
# re-running this over a complete directory verifies it in seconds, and over an interrupted
# one COMPLETES it rather than leaving a half-downloaded container to fail at load.
#
# These containers are large — 5.9 GB (maple) to 1.4 TB (kimi-k3). `scripts/fetch.sh --all`
# is ~2.5 TB and is offered because it is occasionally what you want on a fresh box, not
# because it is a good default.
set -euo pipefail
source "$(dirname "$0")/lib.sh"

usage() { die "usage: scripts/fetch.sh <model>|--all   (models: $(model_names))"; }
[[ $# -ge 1 ]] || usage

if [[ "$1" == "--all" ]]; then
  # `model.py list` is the registry, so this picks up a new model without an edit here.
  for name in $("$PY" "$HARNESS_DIR/model.py" list | awk '{print $1}'); do
    load_model "$name"
    [[ -n "${HF_REPO:-}" ]] || { echo "[fetch] $name: no hf_repo — skipping"; continue; }
    if [[ -d "$CONTAINER" && -f "$CONTAINER/config.json" ]]; then
      echo "[fetch] $name: present at $CONTAINER — verifying against the Hub"
    fi
    fetch_container
  done
  exit 0
fi

load_model "$1"
fetch_container
echo "[fetch] done: $CONTAINER"
echo "[fetch] serve it with: scripts/serve.sh $COLI_MODEL 8080"
