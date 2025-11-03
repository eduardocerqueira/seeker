#date: 2025-11-03T17:11:32Z
#url: https://api.github.com/gists/b9c54bc647a04167b140ef87caafba6d
#owner: https://api.github.com/users/AdoHaha

#!/bin/bash
set -euo pipefail
shopt -s nullglob

FFMPEG_BIN="/usr/bin/ffmpeg"
if [ ! -x "$FFMPEG_BIN" ]; then
  FFMPEG_BIN="$(command -v ffmpeg || true)"
fi

if [ -z "$FFMPEG_BIN" ]; then
  echo "Error: FFmpeg executable not found." >&2
  exit 1
fi

echo "Using FFmpeg at: $FFMPEG_BIN"

if "$FFMPEG_BIN" -hide_banner -filters 2>/dev/null | grep -q vidstabdetect; then
  has_vidstab=true
else
  has_vidstab=false
  echo "Warning: FFmpeg build is missing vidstab filters (vidstabdetect/vidstabtransform)." >&2
  echo "Falling back to tuned deshake filter; install vid.stab for higher quality." >&2
fi

OUTPUT_DIR="stabilized"
mkdir -p "$OUTPUT_DIR"

for file in *.MOV; do
  [ -e "$file" ] || continue
  basename="${file%.*}"
  output_file="${OUTPUT_DIR}/${basename}_stabilized.MOV"

  if [ "$has_vidstab" = true ]; then
    transform_file="${OUTPUT_DIR}/${basename}.trf"

    echo ">>> Analyzing camera motion for $file (vidstab)"
    "$FFMPEG_BIN" -y -hide_banner -loglevel warning \
      -i "$file" \
      -vf "vidstabdetect=shakiness=4:accuracy=15:stepsize=6:tripod=1:mincontrast=0.3:result=${transform_file}" \
      -f null -

    echo ">>> Rendering stabilized clip to $output_file (vidstab)"
    "$FFMPEG_BIN" -y -hide_banner -loglevel warning \
      -i "$file" \
      -vf "vidstabtransform=input=${transform_file}:smoothing=120:tripod=1:relative=0:optzoom=0:zoom=0:interpol=linear:crop=black:zoomspeed=0.05,unsharp=7:7:0.8:3:3:0.4,format=yuv420p" \
      -c:v libx264 -preset slow -crf 18 -pix_fmt yuv420p -c:a pcm_s16le -movflags +faststart \
      "$output_file"
  else
    echo ">>> Rendering stabilized clip to $output_file (deshake fallback)"
    "$FFMPEG_BIN" -y -hide_banner -loglevel warning \
      -i "$file" \
      -vf "deshake=rx=48:ry=48:edge=mirror:blocksize=16:contrast=80:search=less,unsharp=7:7:0.8:3:3:0.4,format=yuv420p" \
      -c:v libx264 -preset slow -crf 18 -pix_fmt yuv420p -c:a pcm_s16le -movflags +faststart \
      "$output_file"
  fi
done

echo "Done. Stabilized clips are in ${OUTPUT_DIR}/"