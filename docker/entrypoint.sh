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
  # Pull the HF snapshot dir out of `hf download` output — robust against progress
  # bars (\r) and the "path:" prefix (hf ≥1.0): the last ".../snapshots/<rev>".
  snap_path() { tr '\r' '\n' | grep -oE '/[^ ]*/snapshots/[^/ ]+' | tail -1; }

  # Primary path: complete/verify from the Hub. `hf download` is idempotent — it
  # fetches only files that are missing or the wrong size, so a full cache returns
  # in seconds, while a PARTIAL cache (an interrupted earlier download) is
  # COMPLETED here instead of being served broken. The old code short-circuited to
  # HF_HUB_OFFLINE and handed a half-downloaded snapshot to the engine, which then
  # died on the first missing tensor (e.g. `model.norm.weight`). Progress → stderr;
  # the snapshot path → stdout, captured here.
  echo "[coli] resolving ${COLI_MODEL_REPO} — fetching any missing files from the Hub..." >&2
  echo "[coli] (mount the host HF cache to persist it: -v ~/.cache/huggingface:/root/.cache/huggingface)" >&2
  if snap=$("$hf_bin" download "$COLI_MODEL_REPO" | snap_path) \
     && [[ -n "$snap" && -f "$snap/config.json" ]]; then
    echo "$snap"
    return
  fi

  # Fallback: Hub unreachable (offline node). Serve the local cache if present, but
  # it may be incomplete — warn, and let the engine surface any missing tensor.
  echo "[coli] Hub unreachable — falling back to the local cache" >&2
  if snap=$(HF_HUB_OFFLINE=1 "$hf_bin" download "$COLI_MODEL_REPO" 2>/dev/null | snap_path) \
     && [[ -n "$snap" && -f "$snap/config.json" ]]; then
    echo "[coli] WARNING: serving from an UNVERIFIED cache. If load fails with a missing tensor, the download is incomplete — re-run with network access (and HF_TOKEN if the repo is gated)." >&2
    echo "$snap"
    return
  fi

  echo "[coli] could not resolve ${COLI_MODEL_REPO}: no /model mount, Hub unreachable, and no usable cache" >&2
  exit 1
}

cmd="${1:-help}"
case "$cmd" in
  # snapshot-taking commands: inject the resolved model dir as their 1st arg
  config | load | gen | tf | capacity | loadbench | repack | serve | bench)
    shift
    snap=$(resolve_model)
    echo "[coli] model: $snap" >&2
    exec coli "$cmd" "$snap" "$@"
    ;;
  *)
    exec coli "$@"
    ;;
esac
