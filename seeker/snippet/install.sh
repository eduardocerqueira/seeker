#date: 2025-12-08T17:04:00Z
#url: https://api.github.com/gists/c4811a090b6c4d73d7faf2c01d16fa6b
#owner: https://api.github.com/users/lucharo

#!/bin/bash
# Install voice-to-text as a background service

set -e

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
SCRIPT_PATH="$SCRIPT_DIR/voice2text.py"
UV_PATH="$(which uv)"
PLIST_NAME="com.user.voice-to-text.plist"
PLIST_DEST="$HOME/Library/LaunchAgents/$PLIST_NAME"

echo "Installing voice-to-text service..."
echo "  Script: $SCRIPT_PATH"
echo "  uv: $UV_PATH"

# Check prerequisites
if [ ! -f "$SCRIPT_PATH" ]; then
    echo "❌ voice2text.py not found in current directory"
    exit 1
fi

if [ -z "$UV_PATH" ]; then
    echo "❌ uv not found. Install with: curl -LsSf https://astral.sh/uv/install.sh | sh"
    exit 1
fi

if ! command -v ollama &> /dev/null; then
    echo "❌ ollama not found. Install with: brew install ollama"
    exit 1
fi

# Ask for flags
read -p "Pause music while recording? [y/N] " pause_music
read -p "Use casual mode (light cleanup)? [y/N] " casual_mode
read -p "Install as background service (runs at login)? [y/N] " install_service

FLAGS=""
[[ "$pause_music" =~ ^[Yy] ]] && FLAGS="$FLAGS --pause-music"
[[ "$casual_mode" =~ ^[Yy] ]] && FLAGS="$FLAGS --casual"

# If not installing as service, just print the run command
if [[ ! "$install_service" =~ ^[Yy] ]]; then
    echo ""
    echo "✅ Ready! Run with:"
    echo "   $SCRIPT_PATH$FLAGS"
    exit 0
fi

# Convert FLAGS for plist format
PLIST_FLAGS=""
[[ "$pause_music" =~ ^[Yy] ]] && PLIST_FLAGS="$PLIST_FLAGS<string>--pause-music</string>"
[[ "$casual_mode" =~ ^[Yy] ]] && PLIST_FLAGS="$PLIST_FLAGS<string>--casual</string>"

# Generate plist
cat > "$PLIST_DEST" << EOF
<?xml version="1.0" encoding="UTF-8"?>
<!DOCTYPE plist PUBLIC "-//Apple//DTD PLIST 1.0//EN" "http://www.apple.com/DTDs/PropertyList-1.0.dtd">
<plist version="1.0">
<dict>
    <key>Label</key>
    <string>com.user.voice-to-text</string>
    <key>ProgramArguments</key>
    <array>
        <string>$UV_PATH</string>
        <string>run</string>
        <string>$SCRIPT_PATH</string>
        $PLIST_FLAGS
    </array>
    <key>RunAtLoad</key>
    <true/>
    <key>KeepAlive</key>
    <true/>
    <key>StandardOutPath</key>
    <string>/tmp/voice-to-text.log</string>
    <key>StandardErrorPath</key>
    <string>/tmp/voice-to-text.err</string>
    <key>EnvironmentVariables</key>
    <dict>
        <key>PATH</key>
        <string>/opt/homebrew/bin:/usr/local/bin:/usr/bin:/bin</string>
    </dict>
</dict>
</plist>
EOF

echo "✅ Plist created at $PLIST_DEST"

# Load service
launchctl unload "$PLIST_DEST" 2>/dev/null || true
launchctl load "$PLIST_DEST"

echo "✅ Service installed and running!"
echo ""
echo "Commands:"
echo "  Logs:    tail -f /tmp/voice-to-text.log"
echo "  Stop:    launchctl unload ~/Library/LaunchAgents/$PLIST_NAME"
echo "  Start:   launchctl load ~/Library/LaunchAgents/$PLIST_NAME"
echo "  Remove:  launchctl unload ~/Library/LaunchAgents/$PLIST_NAME && rm ~/Library/LaunchAgents/$PLIST_NAME"
