#!/usr/bin/env bash
# Container entrypoint for SpeedyColibri on DGX Spark.
#
#   entrypoint [hf_TOKEN] [--model <repo|url|path>] [coli-command [args...]]
#
# - An optional first argument starting with `hf_` is taken as the Hugging Face
#   token (equivalent to HF_TOKEN / HUGGING_FACE_HUB_TOKEN in the environment).
# - `--model` (or $COLI_MODEL_REPO) accepts ANY of:
#     * an HF repo id      nvidia/GLM-5.2-NVFP4
#     * an HF URL          https://huggingface.co/unsloth/GLM-5.2-FP8
#     * a local path       /model  (or any mounted dir with a config.json)
#   The model may be an already-converted colibrì container OR a source
#   checkpoint in block-scaled **FP8** or modelopt **NVFP4** — a source
#   checkpoint is converted to the int4 container automatically on first run
#   (`coli probe` decides), cached under $COLI_CONVERT_DIR so later runs reuse it.
# - For commands that take a model snapshot, the resolved (converted) path is
#   injected as their first argument.
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

# Model spec: `--model <repo|url|path>` (or -m) overrides $COLI_MODEL_REPO.
if [[ "${1:-}" == "--model" || "${1:-}" == "-m" ]]; then
  if [[ -z "${2:-}" ]]; then
    echo "[coli] --model needs a value (HF repo id, HF URL, or local path)" >&2
    exit 2
  fi
  COLI_MODEL_REPO="$2"
  shift 2
fi

: "${COLI_NUM_NODES:=1}"
: "${COLI_NODE_RANK:=0}"
: "${COLI_MODEL_REPO:=mateogrgic/GLM-5.2-colibri-int4-with-int8-mtp}"
# Where auto-converted int4 containers are cached. Defaults under the HF cache so
# a mounted host cache persists conversions across container runs.
: "${COLI_CONVERT_DIR:=${HF_HOME:-/root/.cache/huggingface}/colibri-int4}"

# Accept an HF URL as well as a bare repo id: strip scheme/host and any
# /tree/<rev> or trailing slash, leaving `org/name`.
normalize_repo() {
  local s="$1"
  s="${s#https://}"; s="${s#http://}"
  s="${s#huggingface.co/}"; s="${s#hf.co/}"
  s="${s%/}"
  s="${s%%/tree/*}"
  s="${s%%/blob/*}"
  echo "$s"
}
if [[ "$COLI_MODEL_REPO" != /* ]]; then
  COLI_MODEL_REPO="$(normalize_repo "$COLI_MODEL_REPO")"
fi

if [[ "${COLI_NUM_NODES}" != "1" ]]; then
  echo "[cluster] node ${COLI_NODE_RANK}/${COLI_NUM_NODES} (expert-parallel)" >&2
  # NOTE: multi-node transport (RDMA/RoCE over ConnectX-7) is not wired yet —
  # single-node is the current target. See DEPLOYMENT.md.
fi

# Locate (or fetch) the model snapshot. Prints the snapshot dir on stdout.
resolve_model() {
  # An explicit local path wins (mounted checkpoint, or a previous conversion).
  if [[ "$COLI_MODEL_REPO" == /* ]]; then
    if [[ -f "$COLI_MODEL_REPO/config.json" ]]; then
      echo "$COLI_MODEL_REPO"
      return
    fi
    echo "[coli] --model '$COLI_MODEL_REPO' is a path but has no config.json" >&2
    exit 1
  fi
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

# Given a resolved snapshot, return one the engine can actually load: if it is a
# source checkpoint (block-scaled FP8 or modelopt NVFP4), convert it to the int4
# container once and cache the result; if it is already a container, pass through.
# Prints the loadable snapshot dir on stdout.
ensure_container() {
  local src="$1" fmt dest slug
  fmt=$(coli probe "$src" 2>/dev/null) || fmt="unknown"
  case "$fmt" in
    container)
      echo "$src"
      return
      ;;
    fp8 | nvfp4) ;;
    *)
      echo "[coli] WARNING: '$src' has no recognizable FP8/NVFP4/container marker" >&2
      echo "[coli] passing it through unconverted; the engine will error if it cannot load it" >&2
      echo "$src"
      return
      ;;
  esac

  # Cache key: the model spec, flattened. Bit-widths are part of the key so a
  # different COLI_EBITS/COLI_XBITS doesn't silently reuse the wrong container.
  slug="${COLI_MODEL_REPO//\//--}"
  slug="${slug//[^A-Za-z0-9._-]/_}"
  dest="${COLI_CONVERT_DIR}/${slug}-e${COLI_EBITS:-4}x${COLI_XBITS:-${COLI_EBITS:-4}}io${COLI_IO_BITS:-8}"

  if [[ -f "$dest/.convert-complete" ]]; then
    echo "[coli] using cached int4 container: $dest" >&2
    echo "$dest"
    return
  fi

  echo "[coli] source format: $fmt — converting to the int4 container (one time)" >&2
  echo "[coli]   $src" >&2
  echo "[coli]   -> $dest   (cache: \$COLI_CONVERT_DIR=$COLI_CONVERT_DIR)" >&2
  echo "[coli]   this takes a while (~1h+ for GLM-5.2) and needs ~350 GB free" >&2
  # A partial dir from an interrupted run must not be reused: rebuild from scratch,
  # and only mark complete after convert exits 0.
  rm -rf "$dest"
  mkdir -p "$dest"
  coli convert "$src" "$dest" >&2
  touch "$dest/.convert-complete"
  echo "$dest"
}

cmd="${1:-help}"
case "$cmd" in
  # snapshot-taking commands: inject the loadable model dir as their 1st arg
  config | load | gen | tf | capacity | loadbench | repack | serve | bench)
    shift
    snap=$(resolve_model)
    echo "[coli] model: $snap" >&2
    snap=$(ensure_container "$snap")
    echo "[coli] loading: $snap" >&2
    exec coli "$cmd" "$snap" "$@"
    ;;
  # probe/convert take an explicit path — but allow bare `probe`/`convert` to
  # resolve the configured model spec for convenience.
  probe)
    shift
    if [[ -n "${1:-}" ]]; then exec coli probe "$@"; fi
    exec coli probe "$(resolve_model)"
    ;;
  *)
    exec coli "$@"
    ;;
esac
