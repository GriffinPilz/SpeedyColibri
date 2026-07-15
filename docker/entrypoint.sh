#!/usr/bin/env bash
# Container entrypoint for SpeedyColibri on DGX Spark.
#
#   entrypoint [hf_TOKEN] [coli-command [args...]]
#
# - An optional first argument starting with `hf_` is taken as the Hugging Face
#   token (equivalent to HF_TOKEN / HUGGING_FACE_HUB_TOKEN in the environment).
# - For commands that take a model snapshot, the snapshot path is injected:
#   /model if mounted, else the HF cache (offline), else a fresh `hf download`
#   of $COLI_MODEL_REPO using the token if the repo needs one.
# - Everything else passes straight through to `coli`.
set -euo pipefail

# Manual GPU passthrough (host without CDI / nvidia runtime): run-dgx.sh
# bind-mounts the driver's libcuda.so.<ver> etc.; refresh the linker cache so
# the SONAME links (libcuda.so.1, ...) resolve inside the container.
if compgen -G "/usr/lib/*/libcuda.so.*" >/dev/null 2>&1; then
  ldconfig 2>/dev/null || true
fi

# HF token: first argument or environment.
if [[ "${1:-}" == hf_* ]]; then
  export HF_TOKEN="$1"
  shift
fi
if [[ -z "${HF_TOKEN:-}" && -n "${HUGGING_FACE_HUB_TOKEN:-}" ]]; then
  export HF_TOKEN="$HUGGING_FACE_HUB_TOKEN"
fi

: "${COLI_NUM_NODES:=1}"
: "${COLI_NODE_RANK:=0}"
: "${COLI_MODEL_REPO:=mateogrgic/GLM-5.2-colibri-int4-with-int8-mtp}"

if [[ "${COLI_NUM_NODES}" != "1" ]]; then
  echo "[cluster] node ${COLI_NODE_RANK}/${COLI_NUM_NODES} (expert-parallel)" >&2
  # NOTE: multi-node transport (RDMA/RoCE over ConnectX-7) is not wired yet —
  # single-node is the current target. See DEPLOYMENT.md.
fi

# Locate (or fetch) the model snapshot. Prints the snapshot dir on stdout.
resolve_model() {
  if [[ -f /model/config.json ]]; then
    echo "/model"
    return
  fi
  local hf_bin snap
  hf_bin=$(command -v hf || command -v huggingface-cli) || {
    echo "[coli] no snapshot at /model and no hf CLI in the image" >&2
    exit 1
  }
  # `hf download` prints the snapshot dir on stdout — as a bare path
  # (huggingface-cli) or as "  path: <dir>" (hf ≥1.0). Normalize both.
  snap_path() { tail -1 | sed -E 's/^[[:space:]]*(path:[[:space:]]*)?//'; }
  # Cached and complete → instant, no network, no token needed.
  if snap=$(HF_HUB_OFFLINE=1 "$hf_bin" download "$COLI_MODEL_REPO" 2>/dev/null | snap_path) \
     && [[ -f "$snap/config.json" ]]; then
    echo "$snap"
    return
  fi
  echo "[coli] model not cached — downloading ${COLI_MODEL_REPO} to ${HF_HOME:-~/.cache/huggingface}" >&2
  echo "[coli] (mount the host HF cache to persist it: -v ~/.cache/huggingface:/root/.cache/huggingface)" >&2
  snap=$("$hf_bin" download "$COLI_MODEL_REPO" | snap_path)
  [[ -f "$snap/config.json" ]] || {
    echo "[coli] download did not produce a usable snapshot at '$snap'" >&2
    exit 1
  }
  echo "$snap"
}

cmd="${1:-help}"
case "$cmd" in
  # snapshot-taking commands: inject the resolved model dir as their 1st arg
  config | load | gen | tf | capacity | loadbench | repack | chat | web | serve | bench)
    shift
    snap=$(resolve_model)
    echo "[coli] model: $snap" >&2
    exec coli "$cmd" "$snap" "$@"
    ;;
  *)
    exec coli "$@"
    ;;
esac
