#!/bin/bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PYTHON="${CRYPTO_SWING_PYTHON:-$ROOT/.venv/bin/python}"

cd "$ROOT"

echo "=== PROCESS ==="
pgrep -af "crypto_ai_swing.cli supervisor" || true

echo
echo "=== HEALTH ==="
"$ROOT/scripts/healthcheck.sh" || true

echo
echo "=== FORWARD EVIDENCE ==="
"$PYTHON" -m crypto_ai_swing.cli edge-calibration-status || true

echo
echo "=== LIVE CANARY ==="
"$PYTHON" -m crypto_ai_swing.cli live-canary-status || true
