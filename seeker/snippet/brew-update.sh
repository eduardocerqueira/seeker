#date: 2025-12-24T17:03:22Z
#url: https://api.github.com/gists/26bd61ba6887746f5b28eccd113a598e
#owner: https://api.github.com/users/thisguymartin

#!/bin/bash

# Homebrew Update & Upgrade Script
# Run periodically to keep everything current

set -e

echo "🍺 Homebrew Maintenance Starting..."
echo "=================================="

# Update Homebrew itself
echo "\n📦 Updating Homebrew..."
brew update

# Show outdated packages before upgrading
echo "\n📋 Outdated packages:"
brew outdated || echo "All packages up to date!"

# Upgrade all packages
echo "\n⬆️  Upgrading packages..."
brew upgrade

# Upgrade casks (GUI applications)
echo "\n🖥️  Upgrading casks..."
brew upgrade --cask --greedy

# Cleanup old versions and cache
echo "\n🧹 Cleaning up..."
brew cleanup -s
brew autoremove

# Check for issues
echo "\n🔍 Running diagnostics..."
brew doctor || true

# Show disk space recovered
echo "\n✅ Homebrew maintenance complete!"
echo "=================================="
