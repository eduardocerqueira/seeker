#date: 2025-09-25T16:54:52Z
#url: https://api.github.com/gists/f07a1af57f7fd6149fe5038dd34d4bd6
#owner: https://api.github.com/users/ivikasavnish

#!/bin/bash
# Git Cherry Pick Extension Installer
# Install from: curl -sSL https://gist.githubusercontent.com/ivikasavnish/f07a1af57f7fd6149fe5038dd34d4bd6/raw/install.sh | bash

set -e

GIST_URL="https://gist.githubusercontent.com/ivikasavnish/f07a1af57f7fd6149fe5038dd34d4bd6/raw/git-cherrypick"
INSTALL_DIR="/usr/local/bin"
SCRIPT_NAME="git-cherrypick"

echo "🍒 Installing Git Cherry Pick Extension..."

# Check if running as root for system-wide install
if [[ $EUID -ne 0 ]]; then
    echo "Note: Installing to ~/.local/bin (user directory)"
    INSTALL_DIR="$HOME/.local/bin"
    mkdir -p "$INSTALL_DIR"

    # Add to PATH if not already there
    if [[ ":$PATH:" != *":$HOME/.local/bin:"* ]]; then
        echo "Adding ~/.local/bin to PATH..."
        echo 'export PATH="$HOME/.local/bin:$PATH"' >> ~/.bashrc
        echo 'export PATH="$HOME/.local/bin:$PATH"' >> ~/.zshrc 2>/dev/null || true
        export PATH="$HOME/.local/bin:$PATH"
    fi
else
    echo "Installing system-wide to $INSTALL_DIR"
fi

# Download the script
echo "Downloading from GitHub Gist..."
if command -v curl >/dev/null 2>&1; then
    curl -sSL "$GIST_URL" -o "$INSTALL_DIR/$SCRIPT_NAME"
elif command -v wget >/dev/null 2>&1; then
    wget -q "$GIST_URL" -O "$INSTALL_DIR/$SCRIPT_NAME"
else
    echo "Error: Neither curl nor wget found. Please install one of them."
    exit 1
fi

# Make executable
chmod +x "$INSTALL_DIR/$SCRIPT_NAME"

# Verify installation
if command -v git-cherrypick >/dev/null 2>&1; then
    echo "✅ Installation successful!"
    echo ""
    echo "Usage:"
    echo "  git cherrypick track <source-branch> <destination-branch>"
    echo "  git cherrypick list [source-branch]"
    echo "  git cherrypick select <source-branch> <destination-branch>"
    echo "  git cherrypick status"
    echo ""
    echo "Example:"
    echo "  git cherrypick track feature/new-feature main"
    echo "  git cherrypick select feature/new-feature main"
    echo ""
    echo "For help: git cherrypick --help"
else
    echo "⚠️  Installation completed but git-cherrypick not found in PATH."
    echo "You may need to restart your shell or run:"
    echo "  source ~/.bashrc"
    echo ""
    echo "Or manually add to PATH:"
    echo "  export PATH=\"$INSTALL_DIR:\$PATH\""
fi

echo ""
echo "🔗 Source: https://gist.github.com/ivikasavnish/f07a1af57f7fd6149fe5038dd34d4bd6"