#date: 2024-11-08T17:04:25Z
#url: https://api.github.com/gists/d94a91da156fffab9af8d4bee4652510
#owner: https://api.github.com/users/wteuber

#!/bin/bash

# Usage: ./generate.sh <width> <height> <color> <entropy>
# Example ./generate.sh 100 100 red 0.8

# Default dimensions
WIDTH=${1:-100}
HEIGHT=${2:-100}
PIXELS=$((WIDTH * HEIGHT))

# Get color from command line argument or default to "black"
COLOR_NAME=${3:-black}

# Determine RGB values based on the color name
case "$COLOR_NAME" in
  black)
    COLOR="0 0 0"
    ;;
  red)
    COLOR="255 0 0"
    ;;
  green)
    COLOR="0 255 0"
    ;;
  blue)
    COLOR="0 0 255"
    ;;
  *)
    echo "Invalid color. Choose from: black, red, green, blue."
    exit 1
    ;;
esac

# Entropy value between 0 and 1
ENTROPY=${4:-1}

# Calculate integer probabilities using a linear interpolation
P_COLOR=$(awk -v entropy="$ENTROPY" 'BEGIN { printf "%d", (50 + 50 * (1 - entropy)) }')
P_WHITE=$((100 - P_COLOR))

# Create a temporary file for pixel data
PIXEL_DATA_FILE=$(mktemp)

{
  # Write PPM header
  echo "P3"
  echo "$WIDTH $HEIGHT"
  echo "255"

  # Generate random pixel data using probability scaling
  for ((i = 0; i < PIXELS; i++)); do
    if (( RANDOM % 100 < P_WHITE )); then
      echo "255 255 255"  # White pixel
    else
      echo $COLOR         # Selected color pixel
    fi
  done
} > "$PIXEL_DATA_FILE"

# Output file name based on color, dimensions, and entropy
OUTPUT_FILENAME="random_${COLOR_NAME}_${WIDTH}x${HEIGHT}_e${ENTROPY}.png"

# Create an image from the pixel data
magick "$PIXEL_DATA_FILE" $OUTPUT_FILENAME

# Clean up the temporary file
rm "$PIXEL_DATA_FILE"

echo "Generated image: $OUTPUT_FILENAME"
