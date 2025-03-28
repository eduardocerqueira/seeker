#date: 2025-03-28T16:57:00Z
#url: https://api.github.com/gists/e2ff23e6b631d240944ccf1cc84f7274
#owner: https://api.github.com/users/qubeio

#!/bin/bash

# Define VS Code bundle ID
VS_CODE_ID="com.microsoft.VSCode"

# Verify VS Code is installed
if ! osascript -e "id of app \"Visual Studio Code\"" &>/dev/null; then
  echo "Error: Visual Studio Code not found. Please ensure it's installed."
  exit 1
fi

# Create a temporary directory
TEMP_DIR=$(mktemp -d)
EXTENSIONS_FILE="$TEMP_DIR/extensions.txt"

echo "Downloading language definitions from GitHub Linguist..."
if ! curl -s "https://raw.githubusercontent.com/github/linguist/master/lib/linguist/languages.yml" -o "$TEMP_DIR/languages.yml"; then
  echo "Error: Failed to download language definitions."
  rm -rf "$TEMP_DIR"
  exit 1
fi

echo "Extracting file extensions..."
# Extract extensions based on yq version
if yq --version | grep -q "version 4"; then
  # For yq v4
  yq '.[] | select(.extensions != null) | .extensions[]' "$TEMP_DIR/languages.yml" | sort | uniq > "$EXTENSIONS_FILE"
else
  # For yq v3 or earlier
  yq -r 'to_entries | map(.value.extensions) | flatten | .[] | select(. != null)' "$TEMP_DIR/languages.yml" | sort | uniq > "$EXTENSIONS_FILE"
fi

# Check if extensions were extracted
if [ ! -s "$EXTENSIONS_FILE" ]; then
  echo "Error: Failed to extract extensions. Check if yq is installed and working correctly."
  rm -rf "$TEMP_DIR"
  exit 1
fi

# Log file for errors
ERROR_LOG="$TEMP_DIR/errors.log"

# Set VS Code as the default application for each extension
echo "Setting VS Code as default for file extensions..."
TOTAL=$(wc -l < "$EXTENSIONS_FILE")
COUNT=0
SUCCESS=0

while IFS= read -r ext; do
  COUNT=$((COUNT + 1))
  printf "Processing: %s (%d/%d)... " "$ext" "$COUNT" "$TOTAL"
  
  if duti -s "$VS_CODE_ID" "$ext" all 2>>"$ERROR_LOG"; then
    echo "Done"
    SUCCESS=$((SUCCESS + 1))
  else
    echo "Failed"
  fi
done < "$EXTENSIONS_FILE"

echo "------------------------------"
echo "Summary: Successfully set VS Code as default for $SUCCESS out of $TOTAL extensions."

if [ "$SUCCESS" -ne "$TOTAL" ]; then
  echo "Some extensions failed. Common reasons include:"
  echo "  - The extension is already assigned to a system app"
  echo "  - You need to run the script with sudo for certain extensions"
  echo "  - The VS Code bundle identifier may be different on your system"
  echo "Error details have been saved to: $ERROR_LOG"
else
  rm -rf "$TEMP_DIR"
fi