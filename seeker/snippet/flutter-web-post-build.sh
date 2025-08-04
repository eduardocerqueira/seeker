#date: 2025-08-04T17:16:08Z
#url: https://api.github.com/gists/6da1152e3636c2b69cbc27d79ba54b54
#owner: https://api.github.com/users/animeshjain

#!/bin/bash

# Post-build script for Flutter Web
# Generates production patch loader with content-based hashes

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
APP_ROOT="/path/to/your_app"

echo "Flutter Web Post-Build Script"
echo "============================"

# Check if build directory exists
if [ ! -d "$APP_ROOT/build/web" ]; then
    echo "Error: Build directory not found at $APP_ROOT/build/web"
    echo "Please run 'flutter build web' first"
    exit 1
fi

# Run the asset hash generator
echo "Generating asset hashes..."
"$SCRIPT_DIR/generate-asset-hashes.sh"

# Update index.html to use production patch loader
INDEX_HTML="$APP_ROOT/build/web/index.html"

echo ""
echo "Updating index.html..."

# The production loader is already generated in build/web by generate-asset-hashes.sh
# No need to copy it

# Add timestamp to production patch loader to ensure it's always fetched fresh
if [ -f "$INDEX_HTML" ]; then
    # Generate timestamp
    TIMESTAMP=$(date +%s)
    
    # Add timestamp to production-patch-loader.js reference
    if grep -q "production-patch-loader.js" "$INDEX_HTML"; then
        # Replace production-patch-loader.js with production-patch-loader.js?t=timestamp
        sed -i '' "s|production-patch-loader\.js\"|production-patch-loader.js?t=$TIMESTAMP\"|g" "$INDEX_HTML"
        echo "Added timestamp to production patch loader: ?t=$TIMESTAMP"
    else
        echo "Warning: Production patch loader not found in index.html"
        echo "Please add: <script src=\"production-patch-loader.js\"></script>"
    fi
fi

echo ""
echo "Post-build complete!"
echo ""
echo "Production build is ready with content-based cache busting."
echo "Files will only be reloaded when their content changes."