#date: 2024-10-04T17:09:54Z
#url: https://api.github.com/gists/45618757d45484aae04af51c78ad4014
#owner: https://api.github.com/users/psnonis

#!/bin/bash

# Variables
VERSION=2.18.36
CLI=runai
TARGET_DIR="$HOME/.runai/bin/$VERSION"
SYMLINK_DIR="$HOME/.local/bin"
CLI_PATH="$TARGET_DIR/$CLI"
URL="https://runai.cw.use4-prod.si.czi.technology/cli/darwin"

# Create target directory
mkdir -p "$TARGET_DIR" "$SYMLINK_DIR"

# Download the CLI
echo "🚀 Downloading $CLI v$VERSION..."
curl -L -o "$CLI_PATH" "$URL" && chmod +x "$CLI_PATH" && echo "✅ Downloaded to $CLI_PATH"

# Create symlink
ln -sf "$CLI_PATH" "$SYMLINK_DIR/$CLI" && echo "🔗 Symlink created at $SYMLINK_DIR/$CLI"

# Run the CLI to verify
"$SYMLINK_DIR/$CLI" list