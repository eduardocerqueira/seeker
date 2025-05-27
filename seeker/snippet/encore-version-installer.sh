#date: 2025-05-27T17:13:46Z
#url: https://api.github.com/gists/77042ad1e5e0f813e6c87d8eda2d30a8
#owner: https://api.github.com/users/macleodmac

#!/bin/sh
# Based on Deno installer: Copyright 2019 the Deno authors. All rights reserved. MIT license.
# TODO(everyone): Keep this script simple and easily auditable.

set -e

# Check if version argument is provided
if [ -z "$1" ]; then
    echo "Error: Please provide a version number (e.g., 1.47.0)" 1>&2
    exit 1
fi

VERSION="$1"

case $(uname -sm) in
	"Darwin x86_64") target="darwin_amd64" ;;
	"Darwin arm64")  target="darwin_arm64" ;;
	"Darwin arm64")  target="darwin_arm64" ;;
	"Linux aarch64") target="linux_arm64"  ;;
	"Linux arm64")   target="linux_arm64"  ;;
	*) target="linux_amd64" ;;
esac

# Use version from argument in URL
encore_uri="https://d2f391esomvqpi.cloudfront.net/encore-${VERSION}-${target}.tar.gz"

encore_install="${ENCORE_INSTALL:-$HOME/.encore}"

bin_dir="$encore_install/bin"
exe="$bin_dir/encore"
tar="$encore_install/encore.tar.gz"

if [ ! -d "$bin_dir" ]; then
 	mkdir -p "$bin_dir"
fi

curl --fail --location --progress-bar --output "$tar" "$encore_uri"
cd "$encore_install"

# If encore-go already exists, delete it.
# Merging multiple Go releases into the same directory
# leads to very difficult-to-understand fatal errors.
if [ -d "./encore-go" ]; then
	rm -rf "./encore-go"
fi

# Same goes for runtime
if [ -d "./runtimes" ]; then
	rm -rf "./runtimes"
fi

tar -C "$encore_install" -xzf "$tar"
chmod +x "$bin_dir"/*
rm "$tar"

"$exe" version

echo "Encore was installed successfully to $exe"
if command -v encore >/dev/null; then
	echo "Run 'encore --help' to get started"
else
	case $SHELL in
	/bin/zsh) shell_profile=".zshrc" ;;
	*) shell_profile=".bash_profile" ;;
	esac
	echo "Manually add the directory to your \$HOME/$shell_profile (or similar)"
	echo "  export ENCORE_INSTALL=\"$encore_install\""
	echo "  export PATH=\"\$ENCORE_INSTALL/bin:\$PATH\""
	echo "Run '$exe --help' to get started"
fi 