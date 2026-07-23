#!/usr/bin/env bash
# Serve a model over HTTP, resolving its container from the registry (no hardcoded
# paths — switch models by changing the name).
#   Usage: scripts/serve.sh <model> [port]
#   Env:   SERVE_PORT (default 8080), SERVE_DETACH=1 (background + wait for ready + exit),
#          COLI_BIN, COLI_MODELS_ROOT.
# Foreground by default (Ctrl-C to stop). `coli serve` binds 0.0.0.0 only AFTER the
# model is fully loaded, so a live listener == ready.
set -euo pipefail
source "$(dirname "$0")/lib.sh"
[[ $# -ge 1 ]] || die "usage: scripts/serve.sh <model> [port]   (models: $(model_names))"

load_model "$1"; need_coli; need_container
PORT="${2:-${SERVE_PORT:-8080}}"
HOST="$(hostname 2>/dev/null || echo localhost)"

echo "[serve] $COLI_MODEL ($ARCH) → $CONTAINER  on :$PORT"

if [[ "${SERVE_DETACH:-0}" == "1" ]]; then
  if ss -ltn 2>/dev/null | grep -q ":$PORT "; then
    die "port $PORT already has a listener — stop it first (pkill -f 'coli serve') or pick another port"
  fi
  local_log="/tmp/coli_serve_${COLI_MODEL}.log"
  nohup "$COLI_BIN" serve "$CONTAINER" "$PORT" </dev/null >"$local_log" 2>&1 &
  pid=$!
  echo "[serve] detached pid $pid, log $local_log — waiting for it to load + bind…"
  for _ in $(seq 1 "${SERVE_WAIT:-300}"); do
    kill -0 "$pid" 2>/dev/null || die "serve died during load — see $local_log"
    ss -ltn 2>/dev/null | grep -q ":$PORT " && { ready=1; break; }
    sleep 1
  done
  [[ "${ready:-0}" == 1 ]] || die "serve never bound :$PORT within ${SERVE_WAIT:-300}s — see $local_log"
  echo "[serve] ready. query it with:"
  echo "  curl -s http://$HOST:$PORT/v1/completions -H 'Content-Type: application/json' \\"
  echo "    -d '{\"prompt\":\"The capital of France is\",\"max_tokens\":32}'"
  echo "[serve] stop with: pkill -f 'coli serve'"
else
  echo "[serve] foreground (Ctrl-C to stop). Query from another shell:"
  echo "  curl -s http://$HOST:$PORT/v1/completions -H 'Content-Type: application/json' -d '{\"prompt\":\"…\",\"max_tokens\":32}'"
  exec "$COLI_BIN" serve "$CONTAINER" "$PORT"
fi
