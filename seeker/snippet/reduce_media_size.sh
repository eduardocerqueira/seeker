#date: 2025-04-03T16:45:19Z
#url: https://api.github.com/gists/fff9e2c2bafd0da9e6bb02c107f25a1e
#owner: https://api.github.com/users/mmynk

#!/usr/bin/env bash

# Exit on error
set -e

function usage {
  echo "Usage: $0 [options]"
  echo "Options:"
  echo "  -h, --help             Show this help message"
  echo "  -r, --reduce EXT       Reduce image size (jpg, png)"
  echo "  -m, --mp4              Reduce mp4 size"
  echo "  -o, --output DIR       Output directory (default: ./output)"
  echo "  -q, --quality PERCENT  JPEG quality (default: 85)"
  echo "  -c, --crf VALUE        CRF value for video encoding (default: 24)"
}

function reduce_img_size {
  local ext_lower="$(echo "$1" | tr '[:upper:]' '[:lower:]')"
  local output_dir="$2"
  local quality="$3"

  if [[ "$ext_lower" != "jpg" && "$ext_lower" != "png" ]]; then
    echo "Error: Unsupported file extension: $1" >&2
    return 1
  fi

  shopt -s nocaseglob
  local files=(*."$ext_lower")
  shopt -u nocaseglob

  if [ ! -f "${files[0]}" ]; then
    echo "No files with extension $ext_lower found"
    return 0
  fi

  local total=${#files[@]}
  local ctr=0

  echo "Found $total files with extension ${ext_lower}"

  for f in "${files[@]}"; do
    ctr=$((ctr+1))
    echo "Processing $ctr/$total: $f"

    local output_file="${output_dir}/${f}"

    magick "$f" -strip -interlace Plane -gaussian-blur 0.05 -quality "${quality}%" "$output_file"

    echo "Processed $f -> $output_file"
  done
}

function reduce_mp4_size {
  local output_dir="$1"
  local crf="$2"

  shopt -s nocaseglob
  local files=(*.mp4)
  shopt -u nocaseglob

  if [ ! -f "${files[0]}" ]; then
    echo "No mp4 files found"
    return 0
  fi

  local total=${#files[@]}
  local ctr=0

  echo "Found $total mp4 files"

  for f in "${files[@]}"; do
    ctr=$((ctr+1))
    echo "Processing $ctr/$total: $f"

    local output_file="${output_dir}/${f}"

    ffmpeg -i "$f" -vcodec libx265 -crf "$crf" "$output_file"

    echo "Processed $f -> $output_file"
  done
}

# Default values
output_dir="output"
quality=85
crf=24
action=""
ext=""

# Parse command line arguments
while [[ $# -gt 0 ]]; do
  case $1 in
    -h|--help)
      usage
      exit 0
      ;;
    -r|--reduce)
      action="reduce"
      if [[ -n "$2" && ! "$2" =~ ^- ]]; then
        ext="$2"
        shift
      else
        echo "Error: Missing file extension for --reduce option" >&2
        usage
        exit 1
      fi
      shift
      ;;
    -m|--mp4)
      action="mp4"
      shift
      ;;
    -o|--output)
      if [[ -n "$2" && ! "$2" =~ ^- ]]; then
        output_dir="$2"
        shift
      else
        echo "Error: Missing directory for --output option" >&2
        usage
        exit 1
      fi
      shift
      ;;
    -q|--quality)
      if [[ -n "$2" && ! "$2" =~ ^- ]]; then
        quality="$2"
        shift
      else
        echo "Error: Missing value for --quality option" >&2
        usage
        exit 1
      fi
      shift
      ;;
    -c|--crf)
      if [[ -n "$2" && ! "$2" =~ ^- ]]; then
        crf="$2"
        shift
      else
        echo "Error: Missing value for --crf option" >&2
        usage
        exit 1
      fi
      shift
      ;;
    *)
      echo "Error: Unknown option: $1" >&2
      usage
      exit 1
      ;;
  esac
done

if [[ -z "$action" ]]; then
  echo "Error: No action specified" >&2
  usage
  exit 1
fi

if [[ "$action" == "reduce" && -z "$ext" ]]; then
  echo "Error: File extension required for reduce action" >&2
  usage
  exit 1
fi

mkdir -p "$output_dir"
echo "Output directory: $output_dir"

case $action in
  reduce)
    command -v magick > /dev/null 2>&1 || { echo "Error: magick not found" >&2; exit 1; }
    echo "Reducing images with extension: $ext with quality: $quality"
    reduce_img_size "$ext" "$output_dir" "$quality"
    ;;
  mp4)
    command -v ffmpeg > /dev/null 2>&1 || { echo "Error: ffmpeg not found" >&2; exit 1; }
    echo "Reducing videos with extension: mp4 with crf: $crf"
    reduce_mp4_size "$output_dir" "$crf"
    ;;
esac

if [[ "$(ls -A "$output_dir")" ]]; then
  echo "All files processed successfully. Output saved to $output_dir/"
else
  echo "No files were processed."
fi