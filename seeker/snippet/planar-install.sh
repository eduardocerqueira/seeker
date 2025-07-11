#date: 2025-07-11T17:03:26Z
#url: https://api.github.com/gists/7452a1251a277434c613acd0a6b9e753
#owner: https://api.github.com/users/tarruda

#!/bin/sh

set -e

# Check if uv is installed
if ! command -v uv &> /dev/null; then
  echo "uv is not installed. Installing..."
  curl -LsSf https://astral.sh/uv/install.sh | sh
  echo "uv installed successfully."
fi

# Install planar
echo "Installing planar..."
$HOME/.local/bin/uv tool install planar --extra-index-url=https://coplane.github.io/planar/simple/

if [ $? -eq 0 ]; then
  echo "Planar installed successfully. Restart your shell and run \"planar scaffold myapp\" to get started!"
else
  echo "Error: Failed to install planar."
  exit 1
fi

exit 0