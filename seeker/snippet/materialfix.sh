#date: 2025-02-19T17:00:57Z
#url: https://api.github.com/gists/8b655c93ec2bd5b4e753647d08795c5a
#owner: https://api.github.com/users/OctatonicSunrise

MAPS_DIR="<replace this > steamapps/common/Team Fortress 2/tf/download/maps"
OUTPUT_BASE="<replace this > /steamapps/common/Team Fortress 2/tf/custom/linuxmaterialfix"

RESOLVED_MAPS_DIR=$(readlink -f "$MAPS_DIR")
if [ -n "$RESOLVED_MAPS_DIR" ]; then
    MAPS_DIR="$RESOLVED_MAPS_DIR"
fi

if [ ! -d "$MAPS_DIR" ]; then
    echo "Error: Maps directory '$MAPS_DIR' does not exist."
    exit 1
fi

MATERIALS_OUT="$OUTPUT_BASE/materials"
mkdir -p "$MATERIALS_OUT"

if ! command -v python3 &>/dev/null; then
    echo "Error: 'python3' not found. Please install it to proceed."
    exit 1
fi

if ! command -v unzip &>/dev/null; then
    echo "Error: 'unzip' not found. Please install it to proceed."
    exit 1
fi

for BSP_FILE in "$MAPS_DIR"/*.bsp; do
    if [ ! -f "$BSP_FILE" ]; then
        echo "No BSP files found in '$MAPS_DIR'."
        break
    fi

    echo "Processing BSP file: $BSP_FILE"

    TMP_DIR=$(mktemp -d)
    if [[ "$TMP_DIR" != /tmp/* ]]; then
        echo "Error: Temporary directory '$TMP_DIR' is not under /tmp."
        rm -rf "$TMP_DIR"
        continue
    fi
    echo "Using temporary directory: $TMP_DIR"

    RIP_ZIP="$TMP_DIR/$(basename "$BSP_FILE" .bsp).zip"

    # Extract the embedded ZIP (which contains the materials) using an embedded Python script
    python3 - <<EOF
import sys, os
path = "$BSP_FILE"
with open(path, "rb") as f:
    content = f.read()
offset = content.find(b'\x00\x50\x4b\x03\x04') + 1
if offset == 0:
    print("couldn't find packed content in", path)
    sys.exit(1)
newfile = os.path.join("$TMP_DIR", os.path.basename(path).replace(".bsp", ".zip"))
print("Writing output to", newfile)
with open(newfile, "wb") as output:
    output.write(content[offset:])
EOF

    if [ ! -f "$RIP_ZIP" ]; then
        echo "Error: Failed to create the ZIP file from BSP: $BSP_FILE."
        rm -rf "$TMP_DIR"
        continue
    fi

    unzip -q "$RIP_ZIP" -d "$TMP_DIR"

    # Locate the 'materials' directory among the extracted content (case-insensitive)
    MATERIALS_DIR=$(find "$TMP_DIR" -type d -iname "materials" | head -n 1)
    if [ -z "$MATERIALS_DIR" ]; then
        echo "Warning: No 'materials' directory found in BSP: $BSP_FILE. Skipping..."
        rm -rf "$TMP_DIR"
        continue
    fi
    echo "Found materials directory: $MATERIALS_DIR"

    echo "Copying materials to: $MATERIALS_OUT"
    cp -Rf "$MATERIALS_DIR"/. "$MATERIALS_OUT"

    cd "$MATERIALS_OUT" || {
        echo "Cannot enter materials directory."
        rm -rf "$TMP_DIR"
        continue
    }
    echo "Standardizing file/directory names to lowercase (processing only entries with uppercase characters)..."
    find . -depth | while read -r entry; do
        # Only process entries that contain uppercase letters
        if [[ "$entry" =~ [A-Z] ]]; then
            new_entry=$(echo "$entry" | tr '[:upper:]' '[:lower:]')
            mkdir -p "$(dirname "$new_entry")"
            mv -f "$entry" "$new_entry"
        fi
    done
    echo "Completed processing for BSP file: $BSP_FILE"


    rm -rf "$TMP_DIR"
done
