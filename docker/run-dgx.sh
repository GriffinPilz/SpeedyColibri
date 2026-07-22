#!/usr/bin/env bash
# Spin up the SpeedyColibri container on a DGX Spark host.
#
#   docker/run-dgx.sh [hf_TOKEN] [coli-command [args...]]
#
#   docker/run-dgx.sh gen 100 200 300          # model from HF cache / /model
#   docker/run-dgx.sh hf_abc123 gen 100 200    # token as an argument
#   HF_TOKEN=hf_abc123 docker/run-dgx.sh gen   # ...or from the environment
#
# Handles the three ways a host can expose the GPU, in order of preference:
#   1. CDI specs present            → --device nvidia.com/gpu=all
#   2. nvidia runtime registered    → --gpus all
#   3. neither (stock shared DGX, no root): bind the device nodes and the
#      driver's user-space libraries in by hand — exactly what the CDI spec
#      would do. The entrypoint runs `ldconfig` to wire the SONAME links.
#
# Also mounts the host HF cache (so the 358 GB model is downloaded at most
# once) and passes through COLI_* tuning variables. Extra docker args can be
# injected via COLI_DOCKER_ARGS.
set -euo pipefail

here=$(cd "$(dirname "$0")" && pwd)
IMAGE=${COLI_IMAGE:-speedycolibri:latest}

# HF token: first argument or environment.
if [[ "${1:-}" == hf_* ]]; then
  export HF_TOKEN="$1"
  shift
fi

# Build the image on first use.
if ! docker image inspect "$IMAGE" >/dev/null 2>&1; then
  echo "[run-dgx] image $IMAGE not found — building..." >&2
  docker build -f "$here/Dockerfile" -t "$IMAGE" "$here/.."
fi

# ---- GPU passthrough --------------------------------------------------------
gpu=()
if compgen -G "/etc/cdi/*.yaml" >/dev/null 2>&1 || compgen -G "/var/run/cdi/*.yaml" >/dev/null 2>&1; then
  gpu+=(--device nvidia.com/gpu=all)
elif docker info 2>/dev/null | grep -q "Runtimes:.*nvidia"; then
  gpu+=(--gpus all)
else
  for d in /dev/nvidia0 /dev/nvidiactl /dev/nvidia-uvm /dev/nvidia-uvm-tools /dev/nvidia-modeset; do
    [[ -e "$d" ]] && gpu+=(--device "$d")
  done
  # Driver user-space libraries the CUDA runtime dlopens. Versioned filenames
  # (libcuda.so.580.159.03) — discovered, not hardcoded, so driver updates keep
  # working. ldconfig in the entrypoint recreates the .so.1 links.
  libcuda=$(ls /usr/lib/*/libcuda.so.* 2>/dev/null | head -1) || true
  if [[ -n "${libcuda:-}" ]]; then
    libdir=$(dirname "$libcuda")
    for lib in libcuda libcudadebugger libnvidia-ml libnvidia-nvvm \
               libnvidia-ptxjitcompiler libnvidia-gpucomp libnvidia-cfg \
               libnvidia-allocator; do
      for f in "$libdir/$lib".so.*; do
        [[ -e "$f" ]] && gpu+=(-v "$f:$f:ro")
      done
    done
  fi
  [[ -x /usr/bin/nvidia-smi ]] && gpu+=(-v /usr/bin/nvidia-smi:/usr/bin/nvidia-smi:ro)
  if [[ ${#gpu[@]} -eq 0 ]]; then
    echo "[run-dgx] no NVIDIA devices found — running CPU-only" >&2
  fi
fi

# ---- volumes & environment --------------------------------------------------
vols=()
# Host HF cache → container HF cache (download once, reuse forever).
host_hf="${HF_HOME:-$HOME/.cache/huggingface}"
mkdir -p "$host_hf"
vols+=(-v "$host_hf:/root/.cache/huggingface")
# Optional pre-resolved snapshot dir.
[[ -n "${COLI_MODEL_DIR:-}" ]] && vols+=(-v "$COLI_MODEL_DIR:/model:ro")

envs=()
for v in HF_TOKEN COLI_RAM_GB COLI_VRAM_GB COLI_NGEN COLI_PROFILE COLI_TIMING \
         COLI_LOAD_THREADS COLI_GPU_EXPERTS COLI_NO_ZEROCOPY COLI_BUF_POOL \
         COLI_MODEL_REPO COLI_NUM_NODES COLI_NODE_RANK COLI_PORT COLI_WARMUP \
         COLI_CTX COLI_DISCOVER_SECS \
         COLI_PEERS COLI_EXPERT_PORT COLI_SHARD \
         COLI_PIN_GB COLI_USAGE COLI_PREFETCH COLI_PREFETCH_N COLI_EXPERT_LOG \
         COLI_CONVERT_DIR COLI_EBITS COLI_IO_BITS COLI_XBITS COLI_NLAYERS \
         COLI_XFP8 COLI_KEEP_INDEXER COLI_EXPERT_FP8 \
         COLI_PREFETCH_AHEAD COLI_TC_ATTN COLI_NVFP4_TILED COLI_FFN_DEVCOPY; do
  [[ -n "${!v:-}" ]] && envs+=(-e "$v=${!v}")
done

# Locate the coli subcommand: the entrypoint accepts an optional leading
# `hf_TOKEN` and an optional `--model <spec>` before it, so the subcommand is not
# necessarily $1 (getting this wrong silently skips the network mode for `serve`).
_i=0
_a=("$@")
[[ "${_a[$_i]:-}" == hf_* ]] && _i=$((_i + 1))
[[ "${_a[$_i]:-}" == "--model" || "${_a[$_i]:-}" == "-m" ]] && _i=$((_i + 2))
_cmd="${_a[$_i]:-}"
_next="${_a[$((_i + 1))]:-}"

# `serve`, `worker` and `cluster` need to see the ConnectX/RoCE fabric — the RoCE
# subnet, the kernel ARP table, and UDP broadcast — which the default bridge
# namespace hides. `worker` in particular must be reachable by the driver at its
# RoCE address, which bridge NAT would hide. Run them with host networking; the
# listen port is then already on the host (no `-p` needed, and `-p` conflicts with
# `--network host`). Other commands keep bridge networking.
net=()
ports=()
case "$_cmd" in
  serve | worker | cluster)
    net+=(--network host)
    port="${COLI_PORT:-8080}"
    [[ "$_cmd" == worker ]] && port="${COLI_EXPERT_PORT:-48800}" # matches expert_port()
    [[ "$_next" =~ ^[0-9]+$ ]] && port="$_next"
    case "$_cmd" in
      serve) echo "[run-dgx] host networking; serving on host port ${port}" >&2 ;;
      worker) echo "[run-dgx] host networking; expert shard server on host port ${port}" >&2 ;;
    esac
    ;;
esac

tty=()
[[ -t 0 && -t 1 ]] && tty=(-it)

# shellcheck disable=SC2086
exec docker run --rm "${tty[@]}" "${gpu[@]}" "${net[@]}" "${vols[@]}" "${envs[@]}" "${ports[@]}" \
  ${COLI_DOCKER_ARGS:-} "$IMAGE" "$@"
