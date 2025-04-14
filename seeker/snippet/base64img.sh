#date: 2025-04-14T17:07:52Z
#url: https://api.github.com/gists/b70b6c0cc661176d5538065329bb6d81
#owner: https://api.github.com/users/matthewshammond

#!/usr/bin/env bash

# One-line installer:
# curl -fsSL https://gist.githubusercontent.com/matthewshammond/b70b6c0cc661176d5538065329bb6d81/raw > /usr/local/bin/base64img && chmod +x /usr/local/bin/base64img
#
# Usage: base64img [FILE]
# Formats: APNG BMP GIF JPEG PNG WEBP
#
# Example:
# base64img image.png
# Outputs: <img src='data:image/png;base64, [base64 data]' />

usage() {
  echo "Usage: base64img [FILE]"
  echo "Formats: APNG BMP GIF JPEG PNG WEBP"
  echo ""
  echo "Installation:"
  echo "curl -fsSL https://gist.githubusercontent.com/matthewshammond/b70b6c0cc661176d5538065329bb6d81/raw > /usr/local/bin/base64img && chmod +x /usr/local/bin/base64img"
}

# Print usage and exit if the file was not provided
[ $# -eq 0 ] && usage && exit 1

# Check if file exists
[ ! -f "$1" ] && echo "Error: File not found: $1" && exit 1

# Grab the image format
fmt=$(file "$1" | grep -iEo 'apng|bmp|gif|jpeg|png|webp' | head -n1 | tr '[:upper:]' '[:lower:]')

# Check if the image format is supported
[ -z "$fmt" ] && echo "Error: Unsupported image format" && usage && exit 1

# Create an IMG template
img="<img src='data:image/"$fmt";base64, $(base64 -i "$1")' />"

echo "$img"
