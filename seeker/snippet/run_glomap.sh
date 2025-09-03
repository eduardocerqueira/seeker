#date: 2025-09-03T16:56:26Z
#url: https://api.github.com/gists/9e877924949d778db12c534ff1bafe36
#owner: https://api.github.com/users/Norgus

#!/bin/bash
# ================================================================
#  BASH SCRIPT FOR AUTOMATED PHOTOGRAMMETRY TRACKING WORKFLOW (GLOMAP + COLMAP)
#  macOS / Apple Silicon version (Could work in Intel Mac - Not tested yet)
# ================================================================

set -e  # Stop on error

# --- Resolve top-level folder (one up from this script) ---
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )"
TOP="$(dirname "$SCRIPT_DIR")"

# --- Key paths ---
VIDEOS_DIR="$TOP/VIDEOS"
SCENES_DIR="$TOP/SCENES"

# --- System-wide executables ---
FFMPEG="/opt/homebrew/bin/ffmpeg"
COLMAP="/usr/local/bin/colmap"
GLOMAP="/usr/local/bin/glomap"

# --- CPU thread count ---
NUM_THREADS=$(sysctl -n hw.ncpu)

# --- Ensure executables exist ---
for cmd in "$FFMPEG" "$COLMAP" "$GLOMAP"; do
    if [ ! -x "$cmd" ]; then
        echo "[ERROR] Executable not found: $cmd" >&2
        exit 1
    fi
done

# --- Ensure required folders exist ---
if [ ! -d "$VIDEOS_DIR" ]; then
    echo "[ERROR] Input folder '$VIDEOS_DIR' missing." >&2
    exit 1
fi
mkdir -p "$SCENES_DIR"

# --- Count videos ---
TOTAL=$(find "$VIDEOS_DIR" -maxdepth 1 -type f | wc -l | tr -d ' ')
if [ "$TOTAL" -eq 0 ]; then
    echo "[INFO] No video files found in '$VIDEOS_DIR'."
    exit 0
fi

echo "=============================================================="
echo " Starting GLOMAP pipeline on $TOTAL video(s) …"
echo "=============================================================="

IDX=0
for VIDEO_FILE in "$VIDEOS_DIR"/*; do
    [ -f "$VIDEO_FILE" ] || continue
    IDX=$((IDX + 1))
    BASENAME=$(basename "$VIDEO_FILE")
    BASE="${BASENAME%.*}"

    echo
    echo "[$IDX/$TOTAL] === Processing \"$BASENAME\" ==="

    SCENE_DIR="$SCENES_DIR/$BASE"
    IMG_DIR="$SCENE_DIR/images"
    SPARSE_DIR="$SCENE_DIR/sparse"

    # Skip if already reconstructed
    if [ -d "$SCENE_DIR" ]; then
        echo "       • Skipping \"$BASE\" – already reconstructed."
        continue
    fi

    # Create directories
    mkdir -p "$IMG_DIR" "$SPARSE_DIR"

    # --- 1) Extract frames ---
    echo "       [1/4] Extracting frames …"
    "$FFMPEG" -loglevel error -stats -i "$VIDEO_FILE" -qscale:v 2 "$IMG_DIR/frame_%06d.jpg"

    # Remove dot-underscore macOS files
    find "$IMG_DIR" -name '._*' -delete

    if ! ls "$IMG_DIR"/*.jpg &> /dev/null; then
        echo "       × No frames extracted – skipping \"$BASE\"."
        rm -rf "$SCENE_DIR"
        continue
    fi

    # --- 2) Feature extraction (COLMAP) ---
    echo "       [2/4] COLMAP feature_extractor …"
    "$COLMAP" feature_extractor \
        --database_path "$SCENE_DIR/database.db" \
        --image_path "$IMG_DIR" \
        --ImageReader.single_camera 1 \
        --SiftExtraction.max_image_size 4096

    # --- 3) Sequential matching (COLMAP) ---
    echo "       [3/4] COLMAP sequential_matcher …"
    "$COLMAP" sequential_matcher \
        --database_path "$SCENE_DIR/database.db" \
        --SequentialMatching.overlap 15

    # --- 4) Sparse reconstruction (GLOMAP) ---
    echo "       [4/4] GLOMAP mapper …"
    "$GLOMAP" mapper \
        --database_path "$SCENE_DIR/database.db" \
        --image_path "$IMG_DIR" \
        --output_path "$SPARSE_DIR"

    # --- Export TXT inside model folder ---
    if [ -d "$SPARSE_DIR/0" ]; then
        "$COLMAP" model_converter \
            --input_path "$SPARSE_DIR/0" \
            --output_path "$SPARSE_DIR/0" \
            --output_type TXT > /dev/null
        "$COLMAP" model_converter \
            --input_path "$SPARSE_DIR/0" \
            --output_path "$SPARSE_DIR" \
            --output_type TXT > /dev/null
    fi

    echo "       ✓ Finished \"$BASE\"  ($IDX/$TOTAL)"
done

echo "--------------------------------------------------------------"
echo " All jobs finished – results are in \"$SCENES_DIR\"."
echo "--------------------------------------------------------------"
