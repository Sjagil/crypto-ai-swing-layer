#!/bin/zsh
set -euo pipefail

ROOT="${CRYPTO_SWING_ROOT:-/Users/ayoubalhari/Downloads/crypto-ai-swing-layer}"
CRYPTO_ROOT="${CRYPTO_REPO_PATH:-/Users/ayoubalhari/crypto-stack/crypto-runtime}"
HIST_ROOT="${CRYPTO_SWING_HISTORICAL_DATA_ROOT:-/Volumes/QUANTDATA/quant-data/crypto/pi-mirror/data_store/normalized}"

cd "$ROOT"
source .venv/bin/activate

export CRYPTO_REPO_PATH="$CRYPTO_ROOT"
export CRYPTO_SWING_HISTORICAL_DATA_ROOT="$HIST_ROOT"
export CRYPTO_SWING_FULL_LIVE=NO
export PYTHONUNBUFFERED=1

exec caffeinate -dimsu \
  .venv/bin/python scripts/continuous_research_loop.py \
  --poll-seconds 300 \
  --tcn-timeframe 1h \
  --tcn-horizon-bars 4 \
  --tcn-retrain-seconds 21600 \
  --factory-seconds 21600 \
  --factory-maximum-rows 50000 \
  --factory-exact
