#!/bin/bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PYTHON="${CRYPTO_SWING_PYTHON:-$ROOT/.venv/bin/python}"
MAX_AGE="${CRYPTO_SWING_HEARTBEAT_MAX_AGE_SECONDS:-900}"
HEARTBEAT="$ROOT/output/crypto_ai_swing/supervisor/heartbeat.json"

exec "$PYTHON" "$ROOT/scripts/runtime_health.py" \
  --heartbeat "$HEARTBEAT" \
  --max-age "$MAX_AGE"
