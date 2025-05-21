#date: 2025-05-21T16:59:25Z
#url: https://api.github.com/gists/ee5a5ff6801fed1bfad06799e5dad53c
#owner: https://api.github.com/users/lisaross

#!/bin/bash

# Required parameters:
# @raycast.schemaVersion 1
# @raycast.title Tana to Raycast AI
# @raycast.mode silent
# @raycast.packageName Tana Integration

# Optional parameters:
# @raycast.icon 💡

# ============================================================
# USER CONFIGURATION - EDIT THIS SECTION
# ============================================================
# Set your Raycast AI preset ID here
# To get your preset ID:
# 1. Open Raycast
# 2. Go to AI
# 3. Right-click on your preset
# 4. Select "Copy Deeplink"
# 5. Look for the preset ID in the copied URL (it looks like "AA1E944A-8775-45D6-8BB5-C045B992808C")
PRESET_ID="AA1E944A-8775-45D6-8BB5-C045B992808C"

# Set your preset name (for display purposes only)
PRESET_NAME="Tana Thinker"

# Adjust the delay (in seconds) if needed
RAYCAST_OPEN_DELAY=1.5
PASTE_DELAY=0.5
# ============================================================

# First, try to get selected text using the clipboard
# Save the original clipboard content
ORIGINAL_CLIPBOARD=$(pbpaste)

# Simulate Cmd+C to copy selected text
osascript <<EOF
  tell application "System Events"
    keystroke "c" using command down
  end tell
EOF

# Give a moment for the clipboard to update
sleep 0.3

# Get the (hopefully) selected text
SELECTED_TEXT=$(pbpaste)

# Check if we got new text
if [ "$SELECTED_TEXT" = "$ORIGINAL_CLIPBOARD" ]; then
  # No new text was selected, use what was already in clipboard
  echo "No new text selected, using existing clipboard content"
else
  echo "Using newly selected text"
fi

# Check if we have any text to work with
if [ -z "$SELECTED_TEXT" ]; then
  echo "No text available. Please select or copy text first."
  exit 1
fi

# Open Raycast AI with the preset
open "raycast://extensions/raycast/raycast-ai/ai-chat?context=%7B%22preset%22:%22$PRESET_ID%22%7D"

# Wait for Raycast to open
sleep $RAYCAST_OPEN_DELAY

# Use AppleScript to paste the text and press Enter
osascript <<EOF
  tell application "System Events"
    # Paste the text
    keystroke "v" using command down
    delay $PASTE_DELAY
    # Press Enter to submit
    keystroke return
  end tell
EOF

# Restore original clipboard if we got new text
if [ "$SELECTED_TEXT" != "$ORIGINAL_CLIPBOARD" ]; then
  echo "$ORIGINAL_CLIPBOARD" | pbcopy
fi

echo "Text sent to Raycast AI with preset: $PRESET_NAME"
