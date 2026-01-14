#date: 2026-01-14T17:16:26Z
#url: https://api.github.com/gists/42a1b00dc19aa09f450217c6010319bb
#owner: https://api.github.com/users/zikeji

#!/bin/env bash

# CONFIG

LICENSE_KEY=""
INSTALL_DIR="$HOME/AppImages"
GEARLEVER_COMMAND="flatpak run it.mijorus.gearlever"

# END CONFIG

if ! command -v jq &> /dev/null
then
    echo "jq could not be found. Please install it to continue."
    exit
fi

CURRENT_VERSION=$($GEARLEVER_COMMAND --list-installed 2>/dev/null | grep -i "tinkerwell" | awk '{print $2}' | tr -d '[]')
INFO=$(curl -sL "https://api.beyondco.de/api/versions/tinkerwell?license=$LICENSE_KEY&version=$CURRENT_VERSION")

# Check if this is a fresh install
if [ -z "$CURRENT_VERSION" ]; then
    echo "Fresh install detected. Proceeding with installation..."
    CURRENT_VERSION="none"
else
    echo "Current version: $CURRENT_VERSION"
fi

# Check if update is available
UPDATE_AVAILABLE=$(echo "$INFO" | jq -r '.update_available')
if [ "$UPDATE_AVAILABLE" = "false" ]; then
    echo "No updates available. You are running the latest version."
    exit 0
fi

# Get latest version and construct download URL
LATEST_VERSION=$(echo "$INFO" | jq -r '.latest_version')
DOWNLOAD_BASE_URL=$(echo "$INFO" | jq -r '.license.product.download_base_url')
DOWNLOAD_BASE_NAME=$(echo "$INFO" | jq -r '.license.product.download_base_name')
DOWNLOAD_URL="${DOWNLOAD_BASE_URL}${DOWNLOAD_BASE_NAME}${LATEST_VERSION}.AppImage"

echo "Downloading Tinkerwell $LATEST_VERSION..."
TEMP_FILE=$(mktemp --suffix=.AppImage)
curl -L "$DOWNLOAD_URL" -o "$TEMP_FILE"

if [ $? -ne 0 ]; then
    echo "Download failed."
    rm -f "$TEMP_FILE"
    exit 1
fi

echo "Installing/updating Tinkerwell..."
$GEARLEVER_COMMAND --integrate "$TEMP_FILE" --replace --yes

if [ $? -eq 0 ]; then
    if [ "$CURRENT_VERSION" = "none" ]; then
        echo "Fresh install complete: Tinkerwell $LATEST_VERSION"
    else
        echo "Update complete: $CURRENT_VERSION -> $LATEST_VERSION"
    fi
    rm -f "$TEMP_FILE"
else
    echo "Installation failed."
    rm -f "$TEMP_FILE"
    exit 1
fi

