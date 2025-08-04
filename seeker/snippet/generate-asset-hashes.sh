#date: 2025-08-04T17:16:08Z
#url: https://api.github.com/gists/6da1152e3636c2b69cbc27d79ba54b54
#owner: https://api.github.com/users/animeshjain

#!/bin/bash

# Generate MD5 hashes for all JS and WASM files in the build directory
# Updates the production patch loader template with content hashes

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="/home/projects"
BUILD_DIR="$PROJECT_ROOT/your_app/build/web"
TEMPLATE_FILE="$PROJECT_ROOT/your_app/web/production-patch-loader.js"
OUTPUT_FILE="$PROJECT_ROOT/your_app/build/web/production-patch-loader.js"

echo "Flutter Web Asset Hash Generator"
echo "================================"

# Check if build directory exists
if [ ! -d "$BUILD_DIR" ]; then
    echo "Error: Build directory not found: $BUILD_DIR"
    echo "Please run 'flutter build web' first."
    exit 1
fi

# Check if template exists
if [ ! -f "$TEMPLATE_FILE" ]; then
    echo "Error: Template file not found: $TEMPLATE_FILE"
    exit 1
fi

# Build the asset map
echo "Generating asset map..."
echo ""

ASSET_MAP_CONTENT="  const ASSET_MAP = {"

# Find and hash files
FILE_COUNT=0
while IFS= read -r -d '' file; do
    # Get relative path from build directory
    REL_PATH="${file#$BUILD_DIR/}"
    
    # Calculate MD5 hash (first 12 characters)
    if [[ "$OSTYPE" == "darwin"* ]]; then
        # macOS
        HASH=$(md5 -q "$file" | cut -c1-12)
    else
        # Linux
        HASH=$(md5sum "$file" | cut -c1-12)
    fi
    
    echo "  $REL_PATH -> $HASH"
    
    # Add to asset map
    if [ $FILE_COUNT -gt 0 ]; then
        ASSET_MAP_CONTENT="$ASSET_MAP_CONTENT,"
    fi
    ASSET_MAP_CONTENT="$ASSET_MAP_CONTENT
    '$REL_PATH': '$HASH'"
    
    ((FILE_COUNT++))
done < <(find "$BUILD_DIR" -type f \( -name "*.js" -o -name "*.mjs" -o -name "*.cjs" -o -name "*.wasm" \) -print0)

ASSET_MAP_CONTENT="$ASSET_MAP_CONTENT
  };"

echo ""
echo "Found $FILE_COUNT files"
echo "Updating production patch loader..."

# Copy template to output
cp "$TEMPLATE_FILE" "$OUTPUT_FILE"

# Create a temporary file with the asset map content
TEMP_MAP_FILE=$(mktemp)
echo "$ASSET_MAP_CONTENT" > "$TEMP_MAP_FILE"

# Use sed to replace the empty ASSET_MAP with the generated one
# This approach handles multi-line content better than awk
if [[ "$OSTYPE" == "darwin"* ]]; then
    # macOS sed requires different syntax
    sed -i '' "/^  const ASSET_MAP = {};$/r $TEMP_MAP_FILE" "$OUTPUT_FILE"
    sed -i '' "/^  const ASSET_MAP = {};$/d" "$OUTPUT_FILE"
else
    # Linux sed
    sed -i "/^  const ASSET_MAP = {};$/r $TEMP_MAP_FILE" "$OUTPUT_FILE"
    sed -i "/^  const ASSET_MAP = {};$/d" "$OUTPUT_FILE"
fi

# Clean up temp file
rm -f "$TEMP_MAP_FILE"

echo ""
echo "Production patch loader generated: $OUTPUT_FILE"
echo ""
echo "Asset map contains $FILE_COUNT files with content-based hashes"