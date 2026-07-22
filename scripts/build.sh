#!/usr/bin/env bash
# Build the coli binary WITH the CUDA backend, and verify it — never ship a silent
# CPU-only fallback again.
#
# The trap this guards against (cost most of a debugging session on 2026-07-21):
# `cargo build --release -p coli` WITHOUT `--features cuda` compiles a CPU-only binary
# (`coli backend` prints `cpu`). The expert FFN then runs single-threaded on one core,
# ~16-40x slower, with the GPU sitting idle at P8 — looks like a "degraded box", isn't.
#
# Usage:  scripts/build.sh            # release + cuda, verified
#         COLI_NO_CUDA=1 scripts/build.sh   # intentional CPU-only build (skips verify)
set -euo pipefail
cd "$(dirname "$0")/.."

if [[ "${COLI_NO_CUDA:-0}" == "1" ]]; then
  echo "[build] CPU-only build (COLI_NO_CUDA=1) — no GPU backend"
  cargo build --release -p coli
  exit 0
fi

# Locate CUDA. nvcc is often absent from a non-interactive ssh PATH; find it explicitly.
CUDA_HOME="${CUDA_HOME:-/usr/local/cuda}"
NVCC="${NVCC:-$CUDA_HOME/bin/nvcc}"
if [[ ! -x "$NVCC" ]]; then
  NVCC="$(command -v nvcc || true)"
fi
if [[ -z "$NVCC" || ! -x "$NVCC" ]]; then
  echo "[build] ERROR: nvcc not found (looked at \$CUDA_HOME/bin and PATH)." >&2
  echo "        Set CUDA_HOME/NVCC, or run on a CUDA host. For an intentional CPU build: COLI_NO_CUDA=1 $0" >&2
  exit 1
fi
export CUDA_HOME NVCC
export PATH="$(dirname "$NVCC"):$PATH"
export CUDA_ARCH="${CUDA_ARCH:-sm_121}"   # GB10 / Blackwell (DGX Spark)

# `cudart` lives under lib64 (x86) or targets/<arch>-linux/lib (ARM/sbsa DGX Spark).
# build.rs adds these to the rustc link-search too; exporting LIBRARY_PATH is belt-and-braces.
for d in "$CUDA_HOME"/lib64 "$CUDA_HOME"/targets/*/lib; do
  [[ -d "$d" ]] && LIBRARY_PATH="${LIBRARY_PATH:+$LIBRARY_PATH:}$d"
done
export LIBRARY_PATH

echo "[build] cuda backend: NVCC=$NVCC CUDA_ARCH=$CUDA_ARCH"
cargo build --release -p coli --features cuda

# Verify — the whole point of this script.
BACKEND="$(./target/release/coli backend 2>/dev/null | head -1 || true)"
echo "[build] $BACKEND"
if ! echo "$BACKEND" | grep -qi cuda; then
  echo "[build] ERROR: built binary reports '$BACKEND' — expected a CUDA backend." >&2
  echo "        The CUDA feature did not take. Do NOT deploy this binary." >&2
  exit 1
fi
echo "[build] OK — CUDA binary verified."
