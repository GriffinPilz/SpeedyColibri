#!/usr/bin/env bash
# Spin up the SpeedyColibri container on a DGX Spark host.
#
# Quick start — download (+ convert) and serve a model in one line:
#
#   docker/run-dgx.sh -h hf_xxxxxxxxxxxxxxxxxxxx -p 8080 -m m2.7
#     -h <token>   Hugging Face token (first download only; also HF_TOKEN env)
#     -p <port>    port to serve on                    (default 8080)
#     -m <model>   nemotron | m2.7 | m3 | glm | k3 | v4 | maple  (or any org/repo)
#                  (default glm). Names come from scripts/models.toml, and each
#                  resolves to a PREBUILT container — no conversion on a fresh host.
#   With flags and no subcommand it runs `serve`.
#
# Advanced / positional form (any coli subcommand):
#   docker/run-dgx.sh [hf_TOKEN] [coli-command [args...]]
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

# ---- friendly flags: -h <hf_token>  -p <port>  -m <model> ----------------------
# `docker/run-dgx.sh -h hf_xxx -p 8080 -m m2.7` → download (+convert) + serve, one line.
# Short model names resolve to their HF repo; a full `org/repo` also works.
#
# THE REGISTRY IS `scripts/models.toml`, NOT A MAP IN THIS FILE. There used to be a
# hardcoded case block here, and it drifted exactly as a second hand-maintained copy of a
# closed set always does: it never gained `kimi-k3` or `deepseek-v4-flash` (so `-m v4` was
# silently "unknown model"), and its four entries still pointed at the nvidia/unsloth
# SOURCE checkpoints months after prebuilt containers existed — so every fresh host paid a
# multi-hour conversion for a container it could have downloaded ready-made.
#
# `hf_repo` in the registry is the CONTAINER. Handing one to the entrypoint is safe and is
# the fast path: `ensure_container` probes with `coli probe` and passes a container straight
# through, converting only a source checkpoint.
#
# Aliases stay here because they are a UI concern, not registry data.
model_canon() {
  case "$1" in
    nemotron|nemotron-h|nemotron3)  echo "nemotron-3-super" ;;
    m2.7|m2|minimax-m2)             echo "minimax-m2.7" ;;
    m3)                             echo "minimax-m3" ;;
    glm|glm5.2|glm52)               echo "glm-5.2" ;;
    k3|kimi|kimi-k3)                echo "kimi-k3" ;;
    v4|deepseek|deepseek-v4|deepseek-v4-flash) echo "deepseek-v4-flash" ;;
    maple|maple-preview)            echo "maple-preview" ;;
    *)                              echo "$1" ;;
  esac
}
model_repo() {
  case "$1" in
    */*) echo "$1"; return ;;   # already a full org/repo or URL — pass through
  esac
  local py name repo
  py=$(command -v python3) || { echo ""; return; }
  name=$(model_canon "$1")
  # `2>/dev/null` so an unknown name yields "" (the caller's error path) rather than a
  # traceback; `|| true` keeps `set -e` from killing the script on that expected failure.
  repo=$("$py" "$here/../scripts/model.py" get "$name" hf_repo 2>/dev/null) || true
  echo "$repo"
}
used_flags=0
while [[ "${1:-}" == -[hpm] || "${1:-}" == --hf-token || "${1:-}" == --port || "${1:-}" == --model ]]; do
  case "$1" in
    -h|--hf-token) export HF_TOKEN="${2:?-h needs a token}"; shift 2 ;;
    -p|--port)     export COLI_PORT="${2:?-p needs a port}"; shift 2 ;;
    -m|--model)
      # `-m k3` used to be REFUSED here, because the only route was download-then-convert:
      # a 1561 GB source plus its container on disk at once (~2.96 TB) does not fit on a
      # Spark's 3.6 TB root. That refusal is now obsolete — the registry resolves k3 to the
      # prebuilt CONTAINER (which the refusal itself already suggested as the workaround),
      # and `ensure_container` passes a container through without converting, so the 3 TB
      # peak never happens. What remains is a big download, which is a warning, not a wall.
      repo="$(model_repo "${2:?-m needs a model}")"
      if [[ -z "$repo" ]]; then
        echo "[run-dgx] unknown model '$2'. Registered names:" >&2
        if command -v python3 >/dev/null; then
          python3 "$here/../scripts/model.py" list 2>/dev/null | awk '{printf "[run-dgx]   %s\n", $1}' >&2
        fi
        echo "[run-dgx] short forms: nemotron, m2.7, m3, glm, k3, v4, maple — or any org/repo" >&2
        exit 2
      fi
      case "$(model_canon "$2")" in
        kimi-k3)
          echo "[run-dgx] note: kimi-k3's container is ~1.4 TB. It downloads ready-to-run (no" >&2
          echo "[run-dgx] conversion), but budget the disk. To build it locally instead, without" >&2
          echo "[run-dgx] the download, use scripts/k3_fetch_convert.sh (shard-at-a-time)." >&2 ;;
      esac
      export COLI_MODEL_REPO="$repo"; shift 2 ;;
  esac
  used_flags=1
done
# With the friendly flags and no explicit subcommand, default to serving on the
# chosen port (passed positionally so the entrypoint's `coli serve <container> <port>`
# gets it directly).
[[ "$used_flags" == 1 && $# -eq 0 ]] && set -- serve "${COLI_PORT:-8080}"

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
for v in HF_TOKEN COLI_VRAM_GB COLI_NGEN COLI_PROFILE COLI_TIMING \
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
