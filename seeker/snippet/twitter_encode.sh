#date: 2023-05-22T16:40:55Z
#url: https://api.github.com/gists/770844babf71d8a4d42fdf7d334e1ddd
#owner: https://api.github.com/users/xldeveloper

#!/usr/bin/env bash
set -e

# usage:
#   $ ./encode.sh [INPUT_FILE]
#
# NOTE: The output directory is defined in the script (below) because I use this script with Hazel

# START CONFIGURATION ==================

# Absolute paths to executables
FFMPEG=/usr/local/bin/ffmpeg
FFPROBE=/usr/local/bin/ffprobe
BC=/usr/bin/bc
UUIDGEN=/usr/bin/uuidgen

# Configuration
MAX_SIZE_MiB=512 # twitter's size limit
MAX_BITRATE=2048 # twitter recommended maximum bitrate
OUTPUT_PATH='/Users/vincentriemer/Pictures/Screencasts'

# END CONFIGURATION ==================

# Input/Output Path & File Info
FILEPATH="$1"
FILENAME=$(basename -- "$FILEPATH");
EXTENSION="${FILENAME##*.}"
FILENAME="${FILENAME%.*}"
OUTPUT="$OUTPUT_PATH/$FILENAME.mp4"

# Generate a random filename for ffmpeg's two pass log file
# (this ensures multiple encodings can be run simultaniously)
TWOPASS_LOGFILE=$($UUIDGEN).log

# (200 MiB * 8192 [converts MiB to kBit]) / 600 seconds = ~2730 kBit/s total bitrate
DURATION=$($FFPROBE -v error -show_entries format=duration -of default=noprint_wrappers=1:nokey=1 $FILEPATH)
MAX_SIZE_KBIT=$($BC <<< "$MAX_SIZE_MiB * 8192")
DERIVED_BITRATE=$($BC <<< "$MAX_SIZE_KBIT / $DURATION")

# Clamp derived bitrate to twitter's recommendation
BITRATE=$(( $DERIVED_BITRATE < $MAX_BITRATE ? $DERIVED_BITRATE : $MAX_BITRATE ))

# Define shared args between the two ffmpeg passes
SHARED_OPTIONS_ARR=(
  -i "$FILEPATH" # input
  -y # auto accept prompts
  -c:v libx264 # encode to h264
  -an # disable audio
  -preset veryslow # speed preset
  -movflags +faststart # web specific optimization
  -b:v "${BITRATE}k" # target bitrate
  -profile:v high -level 4.0 # twitter recommended h264 profile
  -color_primaries 1 -color_trc 1 -colorspace 1 # color space passthrough config
  -vf "scale=w=1280:h=1024:force_original_aspect_ratio=decrease:flags=lanczos,pad=ceil(iw/2)*2:ceil(ih/2)*2" # resizing
  -flags +cgop # ensure closed gop (twitter requirement)
  -r 60 # ensure 60fps
  -pix_fmt yuv420p # pixel format
  -threads 2 # number of threads (lower than system count to make sure that doesn't hog the CPU)
  -passlogfile "/tmp/$TWOPASS_LOGFILE"
)
SHARED_OPTIONS="${SHARED_OPTIONS_ARR[@]}"

# Two pass ffmpeg encoding
$FFMPEG $SHARED_OPTIONS -pass 1 -f mp4 /dev/null && \
$FFMPEG $SHARED_OPTIONS -pass 2 $OUTPUT

# Delete the source file
rm -rf "$FILEPATH"
