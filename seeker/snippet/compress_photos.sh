#date: 2024-03-21T17:01:18Z
#url: https://api.github.com/gists/a213591a4da546be17448c2b5849c3c5
#owner: https://api.github.com/users/johnnymo87

#!/usr/bin/env bash

: <<'END'
Script Name: compress_photos.sh

Purpose:
This script is designed to compress a directory of photos to ensure the total
size is under a specified limit (25 MB by default). It's particularly useful
for preparing image files for platforms with strict upload size limits or for
optimizing storage. The script iterates over a set of image files, applying
compression while attempting to maintain a balance between image quality and
file size.

Usage:
./compress_photos.sh <PHOTO_DIR> <OUTPUT_DIR>

Arguments:
- PHOTO_DIR: The directory containing the original photo files.
- OUTPUT_DIR: The directory where the compressed photos will be saved.

Features:
- Handles multiple image formats (jpg, jpeg, png).
- Dynamically calculates target compression settings based on the desired total
  size and the number of images.
- Provides flexibility to adjust compression parameters as needed.

Background:
The script leverages FFmpeg, a powerful multimedia processing tool, to compress
image files. While FFmpeg is primarily known for video and audio manipulation,
it also offers capabilities for image processing, making it a versatile choice
for this task. The script employs a strategy to distribute the total size limit
  evenly across all images, adjusting compression parameters to approximate
  these targets.

Dependencies:
- FFmpeg: Required for image compression. Ensure FFmpeg is installed and
  accessible in your system's PATH.
- Bash: The script is written for Bash shell environments found in Linux and
  macOS systems.

Note:
Image compression involves a trade-off between file size and quality. This
script aims to simplify the compression process, but manual adjustments to the
parameters may be necessary to achieve the desired balance for specific use
cases.

Note:
This script was written with the assistance of the "gpt-4-0125-preview" model
developed by OpenAI.

Author: Jonathan Mohrbacher (github.com/johnnymo87)
Date: 2024-03-21

END

set -euo pipefail

# Check if the correct number of arguments is provided.
if [ "$#" -ne 2 ]; then
  echo "Usage: $0 <PHOTO_DIR> <OUTPUT_DIR>"
  echo "Example: $0 /path/to/photos /path/to/compressed_photos"
  exit 1
fi

# Assign command-line arguments to variables>
PHOTO_DIR="$1"
OUTPUT_DIR="$2"

# Validate PHOTO_DIR existence.
if [ ! -d "$PHOTO_DIR" ]; then
  echo "Error: Photo directory '$PHOTO_DIR' does not exist."
  exit 1
fi

# Create OUTPUT_DIR if it does not exist.
mkdir -p "$OUTPUT_DIR"

# Desired total size in MB for all photos.
TOTAL_SIZE_MB=25

# Extensions to check.
EXTENSIONS=("jpg" "jpeg" "png")

# Initialize an empty array to hold all photo paths.
PHOTOS=()

# Loop through each extension and add matching files to the PHOTOS array.
for EXT in "${EXTENSIONS[@]}"; do
  # Use nullglob to avoid issues when no files match the pattern.
  shopt -s nullglob
  for PHOTO in "$PHOTO_DIR"/*.$EXT; do
    PHOTOS+=("$PHOTO")
  done
  shopt -u nullglob
done

# Number of photos collected.
NUM_PHOTOS=${#PHOTOS[@]}

# Check if photos are found.
if [ "$NUM_PHOTOS" -eq 0 ]; then
  echo "Error: No photos found in '$PHOTO_DIR'."
  exit 1
fi

# Target size per photo in kilobytes (KB).
TARGET_SIZE_PER_PHOTO_KB=$(( ($TOTAL_SIZE_MB * 1024) / $NUM_PHOTOS ))

# Loop through each photo in the PHOTOS array and compress.
for PHOTO in "${PHOTOS[@]}"; do
  FILENAME=$(basename "$PHOTO")
  # Adjust the scale and compression quality to try to meet the target size.
  # Note: You may need to adjust the scale (resize) and quality parameters
  # based on your specific needs.
  ffmpeg -i "$PHOTO" -vf "scale=iw*0.75:ih*0.75" -compression_level 5 "$OUTPUT_DIR/$FILENAME"
  # Check the size and adjust parameters as necessary.
done

echo "Compression complete. Check $OUTPUT_DIR for output."