#!/bin/bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
MODE="${CRYPTO_SWING_PRODUCTION_MODE:-shadow}"
CRYPTO_ROOT="${CRYPTO_REPO_PATH:-$ROOT/../crypto}"
PYTHON="${CRYPTO_SWING_PYTHON:-$ROOT/.venv/bin/python}"
LOCK_DIR="$ROOT/output/crypto_ai_swing/runtime/supervisor.lock"
LOG_DIR="$ROOT/output/crypto_ai_swing/logs"

mkdir -p "$(dirname "$LOCK_DIR")" "$LOG_DIR"

if [[ ! -x "$PYTHON" ]]; then
  echo "Missing Python runtime: $PYTHON" >&2
  exit 2
fi

if [[ ! -d "$CRYPTO_ROOT" ]]; then
  echo "Missing CRYPTO_REPO_PATH: $CRYPTO_ROOT" >&2
  exit 2
fi

case "$MODE" in
  shadow|paper|live) ;;
  *)
    echo "Invalid CRYPTO_SWING_PRODUCTION_MODE=$MODE" >&2
    exit 2
    ;;
esac

if ! mkdir "$LOCK_DIR" 2>/dev/null; then
  if [[ -f "$LOCK_DIR/pid" ]]; then
    OLD_PID="$(cat "$LOCK_DIR/pid" 2>/dev/null || true)"
    if [[ -n "$OLD_PID" ]] && kill -0 "$OLD_PID" 2>/dev/null; then
      echo "Supervisor already running with PID $OLD_PID" >&2
      exit 3
    fi
  fi
  rm -rf "$LOCK_DIR"
  mkdir "$LOCK_DIR"
fi

echo $$ > "$LOCK_DIR/pid"
cleanup() {
  rm -rf "$LOCK_DIR"
}
trap cleanup EXIT INT TERM

export CRYPTO_REPO_PATH="$CRYPTO_ROOT"
export PYTHONUNBUFFERED=1

cd "$ROOT"

if [[ "$MODE" == "live" ]]; then
  echo "Running live startup gate..."
  "$PYTHON" "$ROOT/scripts/live_startup_gate.py"
fi

MODE_UPPER="$(printf '%s' "$MODE" | tr '[:lower:]' '[:upper:]')"
echo "Starting crypto-ai-swing-layer supervisor in ${MODE_UPPER} mode"
echo "Root: $ROOT"
echo "Crypto library: $CRYPTO_ROOT"
echo "PID: $$"

exec "$PYTHON" -m crypto_ai_swing.cli supervisor --mode "$MODE"
