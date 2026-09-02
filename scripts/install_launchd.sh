#!/bin/bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
LABEL="com.sjagil.crypto-ai-swing-layer"
PLIST="$HOME/Library/LaunchAgents/$LABEL.plist"
MODE="${CRYPTO_SWING_PRODUCTION_MODE:-shadow}"
CRYPTO_ROOT="${CRYPTO_REPO_PATH:-$ROOT/../crypto}"
PYTHON="${CRYPTO_SWING_PYTHON:-$ROOT/.venv/bin/python}"
LOG_DIR="$ROOT/output/crypto_ai_swing/logs"

mkdir -p "$HOME/Library/LaunchAgents" "$LOG_DIR"

case "$MODE" in
  shadow|paper|live) ;;
  *)
    echo "Invalid CRYPTO_SWING_PRODUCTION_MODE=$MODE" >&2
    exit 2
    ;;
esac

if [[ "$MODE" == "live" ]]; then
  echo "Refusing to install LaunchAgent directly in live mode."
  echo "Install in shadow mode first, validate runtime health, then switch deliberately."
  exit 4
fi

cat > "$PLIST" <<EOF
<?xml version="1.0" encoding="UTF-8"?>
<!DOCTYPE plist PUBLIC "-//Apple//DTD PLIST 1.0//EN"
  "http://www.apple.com/DTDs/PropertyList-1.0.dtd">
<plist version="1.0">
<dict>
  <key>Label</key>
  <string>$LABEL</string>
  <key>ProgramArguments</key>
  <array>
    <string>/bin/bash</string>
    <string>$ROOT/scripts/run_production.sh</string>
  </array>
  <key>WorkingDirectory</key>
  <string>$ROOT</string>
  <key>EnvironmentVariables</key>
  <dict>
    <key>CRYPTO_REPO_PATH</key>
    <string>$CRYPTO_ROOT</string>
    <key>CRYPTO_SWING_PYTHON</key>
    <string>$PYTHON</string>
    <key>CRYPTO_SWING_PRODUCTION_MODE</key>
    <string>$MODE</string>
    <key>PYTHONUNBUFFERED</key>
    <string>1</string>
  </dict>
  <key>RunAtLoad</key>
  <true/>
  <key>KeepAlive</key>
  <dict>
    <key>SuccessfulExit</key>
    <false/>
  </dict>
  <key>ThrottleInterval</key>
  <integer>15</integer>
  <key>StandardOutPath</key>
  <string>$LOG_DIR/launchd.stdout.log</string>
  <key>StandardErrorPath</key>
  <string>$LOG_DIR/launchd.stderr.log</string>
</dict>
</plist>
EOF

plutil -lint "$PLIST"
launchctl bootout "gui/$(id -u)/$LABEL" 2>/dev/null || true
launchctl bootstrap "gui/$(id -u)" "$PLIST"
launchctl enable "gui/$(id -u)/$LABEL"
launchctl kickstart -k "gui/$(id -u)/$LABEL"

echo "Installed: $PLIST"
echo "Mode: $MODE"
echo "Check: $ROOT/scripts/healthcheck.sh"
