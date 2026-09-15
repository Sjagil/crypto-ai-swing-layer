#!/bin/bash
set -euo pipefail

ROOT="${1:-/Users/ayoubalhari/Downloads/crypto-ai-swing-layer}"
PYTHON="$ROOT/.venv/bin/python"
BIN="$ROOT/.venv/bin/crypto-swing"
UID_NOW="$(id -u)"
PLIST_DIR="$HOME/Library/LaunchAgents"
LOG_DIR="$ROOT/output/crypto_ai_swing/logs"

mkdir -p "$PLIST_DIR" "$LOG_DIR"

if [[ ! -x "$PYTHON" || ! -x "$BIN" ]]; then
  echo "Missing swing-layer virtualenv." >&2
  exit 2
fi

cd "$ROOT"
"$BIN" live-preflight > "$LOG_DIR/round46_preflight.json"

"$PYTHON" - <<'PY'
import json
from pathlib import Path

p = Path("output/crypto_ai_swing/control/mode_state.json")
if not p.is_file():
    raise SystemExit("CANARY_MODE_STATE_MISSING")

x = json.loads(p.read_text())
if x.get("selected_mode") != "canary" or x.get("runtime_mode") != "live":
    raise SystemExit("PERSISTENT_CANARY_MODE_NOT_SELECTED")

print("CANARY_MODE_READY")
PY

LIVE_PLIST="$PLIST_DIR/com.sjagil.crypto-swing-live.plist"
LEARN_PLIST="$PLIST_DIR/com.sjagil.crypto-swing-learning.plist"

cat > "$LIVE_PLIST" <<EOF
<?xml version="1.0" encoding="UTF-8"?>
<!DOCTYPE plist PUBLIC "-//Apple//DTD PLIST 1.0//EN" "http://www.apple.com/DTDs/PropertyList-1.0.dtd">
<plist version="1.0">
<dict>
  <key>Label</key>
  <string>com.sjagil.crypto-swing-live</string>
  <key>ProgramArguments</key>
  <array>
    <string>/bin/bash</string>
    <string>$ROOT/scripts/run_production.sh</string>
  </array>
  <key>WorkingDirectory</key>
  <string>$ROOT</string>
  <key>EnvironmentVariables</key>
  <dict>
    <key>CRYPTO_SWING_PRODUCTION_MODE</key>
    <string>live</string>
    <key>CRYPTO_SWING_CANARY_EXECUTE</key>
    <string>YES</string>
    <key>CRYPTO_REPO_PATH</key>
    <string>/Users/ayoubalhari/Downloads/crypto</string>
    <key>PYTHONUNBUFFERED</key>
    <string>1</string>
  </dict>
  <key>RunAtLoad</key>
  <true/>
  <key>KeepAlive</key>
  <true/>
  <key>ThrottleInterval</key>
  <integer>10</integer>
  <key>StandardOutPath</key>
  <string>$LOG_DIR/live_supervisor.stdout.log</string>
  <key>StandardErrorPath</key>
  <string>$LOG_DIR/live_supervisor.stderr.log</string>
</dict>
</plist>
EOF

cat > "$LEARN_PLIST" <<EOF
<?xml version="1.0" encoding="UTF-8"?>
<!DOCTYPE plist PUBLIC "-//Apple//DTD PLIST 1.0//EN" "http://www.apple.com/DTDs/PropertyList-1.0.dtd">
<plist version="1.0">
<dict>
  <key>Label</key>
  <string>com.sjagil.crypto-swing-learning</string>
  <key>ProgramArguments</key>
  <array>
    <string>$PYTHON</string>
    <string>-m</string>
    <string>crypto_ai_swing.cli</string>
    <string>learning-worker</string>
    <string>--mode</string>
    <string>live</string>
  </array>
  <key>WorkingDirectory</key>
  <string>$ROOT</string>
  <key>EnvironmentVariables</key>
  <dict>
    <key>CRYPTO_REPO_PATH</key>
    <string>/Users/ayoubalhari/Downloads/crypto</string>
    <key>CRYPTO_SWING_LEARNING_PROCESS</key>
    <string>1</string>
    <key>PYTHONUNBUFFERED</key>
    <string>1</string>
  </dict>
  <key>RunAtLoad</key>
  <true/>
  <key>KeepAlive</key>
  <true/>
  <key>ThrottleInterval</key>
  <integer>15</integer>
  <key>StandardOutPath</key>
  <string>$LOG_DIR/learning_worker.stdout.log</string>
  <key>StandardErrorPath</key>
  <string>$LOG_DIR/learning_worker.stderr.log</string>
</dict>
</plist>
EOF

plutil -lint "$LIVE_PLIST"
plutil -lint "$LEARN_PLIST"

launchctl bootout "gui/$UID_NOW" "$LIVE_PLIST" 2>/dev/null || true
launchctl bootout "gui/$UID_NOW" "$LEARN_PLIST" 2>/dev/null || true

launchctl bootstrap "gui/$UID_NOW" "$LIVE_PLIST"
launchctl bootstrap "gui/$UID_NOW" "$LEARN_PLIST"

launchctl kickstart -k "gui/$UID_NOW/com.sjagil.crypto-swing-live"
launchctl kickstart -k "gui/$UID_NOW/com.sjagil.crypto-swing-learning"

echo "ROUND46_SERVICES_STARTED"
echo "Live trading supervisor: com.sjagil.crypto-swing-live"
echo "Continuous learning: com.sjagil.crypto-swing-learning"
echo "Logs: $LOG_DIR"
