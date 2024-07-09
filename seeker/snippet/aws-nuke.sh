#date: 2024-07-09T16:59:18Z
#url: https://api.github.com/gists/5b08d61c03d2396eaa8ec0650d36415f
#owner: https://api.github.com/users/bruteforks

#!/usr/bin/env bash

# Set the target directory
TARGET_DIR="$HOME/.local/bin"

# Fetch the latest release version
latest_version=$(curl -s https://api.github.com/repos/rebuy-de/aws-nuke/releases/latest | grep -oP '"tag_name": "\K(.*)(?=")')

# Remove the 'v' prefix from the version number
version=${latest_version#v}

# Construct the download URL
download_url="https://github.com/rebuy-de/aws-nuke/releases/download/${latest_version}/aws-nuke-${latest_version}-linux-amd64.tar.gz"

# Download the tar.gz file
curl -L -o "aws-nuke-${version}.tar.gz" "$download_url"
echo "Downloaded aws-nuke-${version}.tar.gz"

echo "Extracting it to $TARGET_DIR"
tar -xz -C "$TARGET_DIR" -f "aws-nuke-${version}.tar.gz"

mv "$TARGET_DIR/aws-nuke-${latest_version}-linux-amd64" "$TARGET_DIR/aws-nuke"
echo "Renaming binary"

echo "Removing tar file"
rm -f "aws-nuke-${version}.tar.gz"

echo "aws-nuke has been installed in $TARGET_DIR"
echo "awsnuke-config in ~/.dotfiles"
