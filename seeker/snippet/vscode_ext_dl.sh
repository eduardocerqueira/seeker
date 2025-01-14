#date: 2025-01-14T16:46:19Z
#url: https://api.github.com/gists/700b68a791403b3c563c62ceaa198a20
#owner: https://api.github.com/users/mattaereal

#!/bin/bash

# Check if required tools are installed
if ! command -v jq &> /dev/null || ! command -v curl &> /dev/null || ! command -v unzip &> /dev/null; then
    echo "This script requires 'jq', 'curl', and 'unzip' to be installed."
    exit 1
fi

# Check if an extension name is provided
if [ -z "$1" ]; then
    echo "Usage: $0 <publisher.extension>"
    echo "Example: $0 StevenRutlidge.mnemonic-validator"
    exit 1
fi

# Extension name from the argument
EXTENSION_NAME="$1"

# Query Visual Studio Marketplace API
RESPONSE=$(curl -s -X POST "https://marketplace.visualstudio.com/_apis/public/gallery/extensionquery" \
    -H "Content-Type: application/json" \
    -H "Accept: application/json;api-version=3.0-preview.1" \
    -d "{
          \"filters\": [
              {
                  \"criteria\": [
                      {
                          \"filterType\": 7,
                          \"value\": \"$EXTENSION_NAME\"
                      }
                  ]
              }
          ],
          \"flags\": 131
        }")

# Extract the VSIX download URL
VSIX_URL=$(echo "$RESPONSE" | jq -r '.results[0].extensions[0].versions[0].files[] | select(.assetType == "Microsoft.VisualStudio.Services.VSIXPackage") | .source')

# Check if a URL was found
if [ -z "$VSIX_URL" ]; then
    echo "Error: Could not find a download URL for extension '$EXTENSION_NAME'."
    exit 1
fi

# Download the VSIX file
VSIX_FILENAME="${EXTENSION_NAME}.vsix"
echo "Downloading VSIX package for '$EXTENSION_NAME'..."
curl -o "$VSIX_FILENAME" "$VSIX_URL"

# Check if the download was successful
if [ $? -eq 0 ]; then
    echo "Download completed: $VSIX_FILENAME"
else
    echo "Error: Failed to download the VSIX package."
    exit 1
fi

# Unzip the VSIX file
UNZIP_DIR="${EXTENSION_NAME}_unzipped"
echo "Unzipping the VSIX package to '$UNZIP_DIR'..."
mkdir -p "$UNZIP_DIR"
unzip -q "$VSIX_FILENAME" -d "$UNZIP_DIR"

# Check if the unzip was successful
if [ $? -eq 0 ]; then
    echo "Unzip completed. Files extracted to: $UNZIP_DIR"
else
    echo "Error: Failed to unzip the VSIX package."
    exit 1
fi

echo "Done. You can now inspect the extracted files in '$UNZIP_DIR'."

