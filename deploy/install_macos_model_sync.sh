#!/bin/bash
set -euo pipefail
SWING="${SWING:-/Users/ayoubalhari/Downloads/crypto-ai-swing-layer}"
LABEL="com.sjagil.crypto-pi-model-sync"
PLIST="$HOME/Library/LaunchAgents/$LABEL.plist"
LOGDIR="$SWING/output/crypto_ai_swing/logs"
mkdir -p "$HOME/Library/LaunchAgents" "$LOGDIR"
cat > "$PLIST" <<EOF
<?xml version="1.0" encoding="UTF-8"?>
<!DOCTYPE plist PUBLIC "-//Apple//DTD PLIST 1.0//EN" "http://www.apple.com/DTDs/PropertyList-1.0.dtd">
<plist version="1.0"><dict>
<key>Label</key><string>$LABEL</string>
<key>ProgramArguments</key><array><string>/bin/bash</string><string>$SWING/deploy/sync_models_from_pi.sh</string></array>
<key>StartInterval</key><integer>300</integer>
<key>RunAtLoad</key><true/>
<key>StandardOutPath</key><string>$LOGDIR/pi-model-sync.stdout.log</string>
<key>StandardErrorPath</key><string>$LOGDIR/pi-model-sync.stderr.log</string>
</dict></plist>
EOF
plutil -lint "$PLIST"
launchctl bootout "gui/$(id -u)/$LABEL" 2>/dev/null || true
launchctl bootstrap "gui/$(id -u)" "$PLIST"
launchctl enable "gui/$(id -u)/$LABEL"
launchctl kickstart -k "gui/$(id -u)/$LABEL"
echo "MAC_MODEL_SYNC_INSTALLED=PASS"
