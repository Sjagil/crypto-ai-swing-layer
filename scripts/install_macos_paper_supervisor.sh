#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
BIN="$ROOT/.venv/bin/crypto-swing"
LABEL="com.sjagil.crypto-swing-paper"
PLIST="$HOME/Library/LaunchAgents/$LABEL.plist"
LOGDIR="$ROOT/output/crypto_ai_swing/runtime"

if [[ ! -x "$BIN" ]]; then
  echo "Missing executable: $BIN" >&2
  exit 1
fi

mkdir -p "$HOME/Library/LaunchAgents" "$LOGDIR"
cat > "$PLIST" <<PLIST
<?xml version="1.0" encoding="UTF-8"?>
<!DOCTYPE plist PUBLIC "-//Apple//DTD PLIST 1.0//EN" "http://www.apple.com/DTDs/PropertyList-1.0.dtd">
<plist version="1.0">
<dict>
  <key>Label</key><string>$LABEL</string>
  <key>ProgramArguments</key>
  <array>
    <string>$BIN</string>
    <string>supervisor</string>
    <string>--mode</string><string>paper</string>
  </array>
  <key>WorkingDirectory</key><string>$ROOT</string>
  <key>RunAtLoad</key><true/>
  <key>KeepAlive</key><true/>
  <key>ThrottleInterval</key><integer>15</integer>
  <key>StandardOutPath</key><string>$LOGDIR/supervisor-paper.stdout.log</string>
  <key>StandardErrorPath</key><string>$LOGDIR/supervisor-paper.stderr.log</string>
</dict>
</plist>
PLIST

launchctl bootout "gui/$(id -u)/$LABEL" 2>/dev/null || true
launchctl bootstrap "gui/$(id -u)" "$PLIST"
launchctl enable "gui/$(id -u)/$LABEL"
launchctl kickstart -k "gui/$(id -u)/$LABEL"
echo "Installed $LABEL"
echo "plist: $PLIST"
echo "logs:  $LOGDIR"
