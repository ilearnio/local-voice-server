#!/usr/bin/env bash
# Install (or reinstall) the local voice server as a macOS login service.
set -euo pipefail

REPO="$(cd "$(dirname "$0")/.." && pwd)"
LABEL="${VOICE_SERVER_LABEL:-io.github.ilearnio.local-voice-server}"
LOGS="$HOME/Library/Logs/local-voice-server"
PLIST="$HOME/Library/LaunchAgents/$LABEL.plist"

mkdir -p "$LOGS" "$HOME/Library/LaunchAgents"
(cd "$REPO" && uv sync --quiet)
sed -e "s#__LABEL__#$LABEL#g" -e "s#__REPO__#$REPO#g" -e "s#__LOGS__#$LOGS#g" \
  "$REPO/launchd/local-voice-server.plist.template" > "$PLIST"
plutil -lint "$PLIST" >/dev/null
launchctl bootout "gui/$(id -u)/$LABEL" 2>/dev/null || true
launchctl bootstrap "gui/$(id -u)" "$PLIST"
echo "Installed $LABEL. Logs: $LOGS/server.log"
