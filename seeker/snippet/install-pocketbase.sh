#date: 2024-11-13T16:53:11Z
#url: https://api.github.com/gists/e89857817346b7510fb5a83743a91d8b
#owner: https://api.github.com/users/robo-monk

#!/usr/bin/env bash
set -e

# PocketBase Installation Script
# This script installs PocketBase and sets up an admin user.
# Usage: ./install-pocketbase.sh <admin-email> [pocketbase-version]

# Function to get the latest available PocketBase version from GitHub
getLatestAvailableVersion() {
    # Fetch the URL that the "latest" release redirects to
    local latest_release_url
    latest_release_url=$(curl -Ls -o /dev/null -w %{url_effective} https://github.com/pocketbase/pocketbase/releases/latest)
    # Extract the version number from the URL
    local version
    version=$(basename "$latest_release_url")
    version=${version#v}
    echo "$version"
}

# Function to print usage information
print_usage() {
    echo "Usage: $0 <admin-email> [pocketbase-version]"
}

# Check if the admin email is provided
if [ -z "$1" ]; then
  print_usage
  exit 1
fi

# Set the admin email
POCKETBASE_ADMIN_EMAIL=$1

latestAvailableVersion=$(getLatestAvailableVersion)
echo "Latest available version is $latestAvailableVersion"

# Determine the desired PocketBase version
if [ -n "$2" ]; then
  pocketbaseVersion="$2"
else
  pocketbaseVersion="$latestAvailableVersion"
fi

# Get the latest available version

# If the desired version is not the latest, print a warning
if [ "$pocketbaseVersion" != "$latestAvailableVersion" ]; then
  echo -e "\e[33mWarning: You are installing PocketBase version $pocketbaseVersion, but the latest available version is $latestAvailableVersion.\e[0m"
fi

# Define the PocketBase binary path
dist="./pocketbase"

# Check if PocketBase is already installed
if [ -e "$dist" ]; then
  installedVersion=$("$dist" --version | awk '{print $3}')
  echo "PocketBase already installed at $dist with version $installedVersion"

  if [ "$installedVersion" == "$pocketbaseVersion" ]; then
    echo "Installed version matches desired version $pocketbaseVersion. Exiting."
    exit 0
  else
    echo "Installed version $installedVersion does not match desired version $pocketbaseVersion."
    echo "Proceeding to download and install version $pocketbaseVersion."
  fi
fi

# Determine the system OS and architecture
systemOsValue="$(uname -s | tr '[:upper:]' '[:lower:]')"

if [ "$systemOsValue" == "darwin" ]; then
  systemOs="darwin"
  buildx_arch="amd64"
elif [ "$systemOsValue" == "linux" ]; then
  systemOs="linux"
  currentArch="$(uname -m)"
  if [ "$currentArch" == "x86_64" ] || [ "$currentArch" == "amd64" ]; then
    buildx_arch="amd64"
  elif [ "$currentArch" == "aarch64" ] || [ "$currentArch" == "arm64" ]; then
    buildx_arch="arm64"
  else
    echo "Error: Unsupported architecture '$currentArch'"
    exit 1
  fi
else
  echo "Error: Unsupported operating system '$systemOsValue'"
  exit 1
fi

echo "Using PocketBase version '$pocketbaseVersion' for $systemOs $buildx_arch"

# Construct the download URL
url="https://github.com/pocketbase/pocketbase/releases/download/v${pocketbaseVersion}/pocketbase_${pocketbaseVersion}_${systemOs}_${buildx_arch}.zip"
echo "Downloading from $url"

# Create a temporary directory for the download
tempDir=$(mktemp -d)
echo "Temporary directory: $tempDir"

# Download the PocketBase zip file
if curl --output /dev/null --silent --head --fail "$url"; then
  curl -L -o "$tempDir/pocketbase.zip" "$url"
  echo "Download completed"
else
  echo "Error: URL '$url' does not exist."
  exit 1
fi

# Unzip the downloaded file
unzip "$tempDir/pocketbase.zip" -d "$tempDir"

# Move the PocketBase binary to the destination directory
mv "$tempDir/pocketbase" "$dist"

# Make sure the binary is executable
chmod +x "$dist"

# Remove the temporary directory
rm -rf "$tempDir"

# Generate a random password for the admin user
POCKETBASE_ADMIN_PASSWORD= "**********"

# Run migrations
echo "Running migrations..."
./pocketbase migrate up

# Create the admin user
echo "Creating admin user..."
./pocketbase admin create "$POCKETBASE_ADMIN_EMAIL" "$POCKETBASE_ADMIN_PASSWORD"

echo "PocketBase installation and setup completed successfully."

echo "---"
echo "POCKETBASE_ADMIN_EMAIL=$POCKETBASE_ADMIN_EMAIL"
echo "POCKETBASE_ADMIN_PASSWORD= "**********"
POCKETBASE_ADMIN_PASSWORD"
