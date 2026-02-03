#date: 2026-02-03T17:31:00Z
#url: https://api.github.com/gists/93a3cdaff36b63ecf65feb60827449bc
#owner: https://api.github.com/users/ericswpark

#!/usr/bin/env bash

set -euo pipefail

TAILSCALE_REPO_NAME="tailscale/tailscale"
GITHUB_API_URL="https://api.github.com/repos"
GITHUB_URL="https://github.com"
GITHUB_TAILSCALE_API_URL="$GITHUB_API_URL/$TAILSCALE_REPO_NAME"
GITHUB_TAILSCALE_URL="$GITHUB_URL/$TAILSCALE_REPO_NAME"

# Check latest version on GitHub
LATEST_TAILSCALE_VERSION=$(curl -s $GITHUB_TAILSCALE_API_URL/releases/latest | jq --raw-output '.name')
if [[ "$LATEST_TAILSCALE_VERSION" == v* ]]; then
    LATEST_TAILSCALE_VERSION="${LATEST_TAILSCALE_VERSION:1}"
fi
echo "The latest release of Tailscale on GitHub is $LATEST_TAILSCALE_VERSION."

# Check locally installed version
CURRENT_TAILSCALE_VERSION=$(tailscale version | head -n 1)
echo "The locally installed version of Tailscale is $CURRENT_TAILSCALE_VERSION."

if [[ "$LATEST_TAILSCALE_VERSION" == "$CURRENT_TAILSCALE_VERSION" ]]; then
    echo "The locally installed version appears to be the latest!"
    exit 0
fi

exit 0

# Initialize directories
CURR_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
WORK_DIR=`mktemp -d`

if [[ ! "$WORK_DIR" || ! -d "$WORK_DIR" ]]; then
  echo "Could not create temporary directory to work in!"
  exit 1
fi

function cleanup {
  rm -rf "$WORK_DIR"
  echo "Deleted temporary working directory $WORK_DIR"
}

trap cleanup EXIT

# Switch to work directory
pushd $WORK_DIR

echo "Downloading new Tailscale binary..."
wget "https://pkgs.tailscale.com/stable/tailscale_${LATEST_TAILSCALE_VERSION}_amd64.tgz"

# Unpack
tar -xzvf tailscale_${LATEST_TAILSCALE_VERSION}_amd64.tgz
cd tailscale_${LATEST_TAILSCALE_VERSION}_amd64

# Copy
mv tailscale ~/.local/bin/
mv tailscaled ~/.local/bin/

popd

# Restart services
systemctl --user restart tailscaled.service

echo "Done! Tailscale has been updated to ${LATEST_TAILSCALE_VERSION}."
