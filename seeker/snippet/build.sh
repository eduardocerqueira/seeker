#date: 2025-07-18T17:00:17Z
#url: https://api.github.com/gists/90209db68fca6adf06f5acf2e484de2b
#owner: https://api.github.com/users/garymjr

#!/bin/bash

# Define variables (you might want to pass these as arguments or set them dynamically)
pkg_name="opencode" # Replace with your actual package name, e.g., "opencode-cli"
version="0.0.0-dev"             # Replace with your actual version

# Define the single target directly
os="darwin"
arch="arm64"

# Directly define the GOARCH value for arm64, as it's the only target now
go_arch_value="arm64" # Corresponds to GOARCH["arm64"]

# Clean up previous build directory
echo "Removing existing dist directory..."
rm -rf dist

# --- Build steps for darwin arm64 ---

echo "Building ${os}-${arch}"
name="${pkg_name}-${os}-${arch}"

# Create destination directory
mkdir -p "dist/${name}/bin"

# Build Go TUI application
# Note: The original script had `../opencode/dist/${name}/bin/tui` as output path
# and `../tui/cmd/opencode/main.go` as source, with `cwd("../tui")`.
# This means the output path is relative to the original script's location,
# but the build command is run from `../tui`.
# We'll adjust paths assuming the script is run from the project root.
echo "Building Go TUI for ${os}-${arch}..."
(cd "../tui" && CGO_ENABLED=0 GOOS="${os}" GOARCH="${go_arch_value}" go build -ldflags="-s -w -X main.Version=${version}" -o "../opencode/dist/${name}/bin/tui" "../tui/cmd/opencode/main.go")

# Build Bun application
echo "Building Bun application for ${os}-${arch}..."
bun build --define OPENCODE_VERSION="'${version}'" --compile --minify --target="bun-${os}-${arch}" --outfile="dist/${name}/bin/opencode" "./src/index.ts" "./dist/${name}/bin/tui"

# Remove the temporary tui binary
echo "Cleaning up temporary tui binary..."
rm -rf "./dist/${name}/bin/tui"

# Create package.json
echo "Creating package.json for ${name}..."
json_os="${os}"
if [ "${os}" = "windows" ]; then
  json_os="win32"
fi

# Use jq to create the JSON, then write to file
jq -n \
  --arg name "$name" \
  --arg version "$version" \
  --argjson os "[\"$json_os\"]" \
  --argjson cpu "[\"$arch\"]" \
  '{name: $name, version: $version, os: $os, cpu: $cpu}' > "dist/${name}/package.json"

echo "Build process complete."
