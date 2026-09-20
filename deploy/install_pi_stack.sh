#!/bin/bash
set -euo pipefail
SWING="/home/pi/sjagil/crypto-ai-swing-layer"
CRYPTO="/home/pi/sjagil/crypto"
PY="$SWING/.venv/bin/python"

[ "$(id -u)" -ne 0 ] || { echo "Run this as pi, not root; sudo is used only for systemd installation."; exit 2; }
test -x "$PY"
test -d "$CRYPTO"
test -d /mnt/data

export CRYPTO_REPO_PATH="$CRYPTO"
export CRYPTO_SWING_PRODUCTION_MODE=shadow
export CRYPTO_SWING_CANARY_EXECUTE=NO
export CRYPTO_SWING_FULL_LIVE=NO
export CRYPTO_FULL_LIVE=NO

"$PY" -m pip check
"$PY" -m py_compile \
  "$CRYPTO/data/cmc_startup_intelligence.py" \
  "$CRYPTO/scripts/cmc_startup_intelligence.py" \
  "$SWING/src/crypto_ai_swing/agents/cmc_pit_features.py" \
  "$SWING/src/crypto_ai_swing/intelligence/cmc_startup_bridge.py" \
  "$SWING/src/crypto_ai_swing/research/cmc_feature_research.py" \
  "$SWING/src/crypto_ai_swing/orchestration/pi_learning_worker.py" \
  "$SWING/scripts/pi_learning_worker.py"

sudo install -m 0644 "$SWING/deploy/crypto-cmc-intelligence.service" /etc/systemd/system/crypto-cmc-intelligence.service
sudo install -m 0644 "$SWING/deploy/crypto-cmc-websocket.service" /etc/systemd/system/crypto-cmc-websocket.service
sudo install -m 0644 "$SWING/deploy/crypto-swing-learning.service" /etc/systemd/system/crypto-swing-learning.service
sudo systemctl daemon-reload

# Install only.  Bootstrap/history verification must pass before enabling.
sudo systemctl disable --now crypto-cmc-intelligence.service crypto-cmc-websocket.service crypto-swing-learning.service 2>/dev/null || true

echo "PI_SERVICE_INSTALL=PASS"
echo "Services are intentionally DISABLED until bootstrap/coverage and smoke tests pass."
