#date: 2025-03-03T16:55:59Z
#url: https://api.github.com/gists/927bd15094de35cc9505cda27c00f1b7
#owner: https://api.github.com/users/f-honcharenko

#!/bin/bash

# Get current Git branch name
BRANCH_NAME=$(git rev-parse --abbrev-ref HEAD | tr '[:lower:]' '[:upper:]')

# Path to the VS Code settings.json file (update this if necessary)
SETTINGS_FILE="$HOME/Library/Application Support/Code/User/settings.json"  # For macOS
# SETTINGS_FILE="$HOME/.config/Code/User/settings.json"  # For Linux
# SETTINGS_FILE="$APPDATA\\Code\\User\\settings.json"  # For Windows

# Check if settings.json exists
if [[ ! -f "$SETTINGS_FILE" ]]; then
    echo "settings.json not found!"
    exit 1
fi
# Format the text with branch name
FORMATTED_TEXT="Current branch name: ${BRANCH_NAME}"
# Clean up existing dynamic branch instruction if it exists
TMP_FILE_CLEAN=$(mktemp)
jq 'if .["github.copilot.chat.commitMessageGeneration.instructions"] | type == "array" then
    .["github.copilot.chat.commitMessageGeneration.instructions"] |= map(select(.isItDynamicBranchInstruction != true))
else . end' "$SETTINGS_FILE" > "$TMP_FILE_CLEAN"
mv "$TMP_FILE_CLEAN" "$SETTINGS_FILE"
# Use jq to update the array property while preserving existing elements
TMP_FILE=$(mktemp)
if ! jq --arg bn "$FORMATTED_TEXT" '
    if .["github.copilot.chat.commitMessageGeneration.instructions"] | type == "array"
    then .["github.copilot.chat.commitMessageGeneration.instructions"] += [{"text": $bn, "isItDynamicBranchInstruction": true}]
    else .["github.copilot.chat.commitMessageGeneration.instructions"] = [{"text": $bn, "isItDynamicBranchInstruction": true}]
    end
' "$SETTINGS_FILE" > "$TMP_FILE"; then
    echo "Error: Failed to update settings.json"
    rm "$TMP_FILE"
    exit 1
fi
mv "$TMP_FILE" "$SETTINGS_FILE"

echo "Updated settings.json with the next branch name: ${BRANCH_NAME}."