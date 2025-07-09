#date: 2025-07-09T17:12:34Z
#url: https://api.github.com/gists/f7ee94550d6d421ea2675fa4d6aee281
#owner: https://api.github.com/users/surya-ven

#!/bin/bash
# Ensure Homebrew is in the PATH for the script
export PATH="/opt/homebrew/bin:$PATH"

echo "--- Starting Homebrew Update: $(date) ---"

# Update Homebrew package lists
brew update

# Upgrade all outdated packages and applications
brew upgrade
brew upgrade --cask

# Clean up old versions and stale lock files
brew cleanup

echo "--- Finished Homebrew Update: $(date) ---"