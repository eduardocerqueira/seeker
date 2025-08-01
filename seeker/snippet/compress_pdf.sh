#date: 2025-08-01T16:48:40Z
#url: https://api.github.com/gists/dc838e1e688829b9d20488a0d2464444
#owner: https://api.github.com/users/jordankobellarz

#!/bin/bash

# compress_pdf.sh
# -------------------------------
# Transform high-resolution, print-ready PDFs into lightweight versions for digital use.
# Ideal for scanned PDFs or large files with embedded images.
#
# Keywords: compress PDF, reduce PDF size, shrink PDF, optimize PDF for web, scan to PDF, convert print PDF to screen version, image-based PDF compression, linux pdf optimizer, bash compress pdf
#
# License: MIT
# Author: Jordan Kobellarz

# === DEFAULT CONFIGURATION ===
DPI=100
JPEG_QUALITY=50
MAX_WIDTH=1200
OUTPUT_PDF=""

# === USAGE ===
show_help() {
  echo "Usage: ./compress_pdf.sh [options] input.pdf"
  echo ""
  echo "Options:"
  echo "  -r, --dpi [value]         Output resolution in DPI (default: 100)"
  echo "  -q, --quality [value]     JPEG quality (1–100, default: 50)"
  echo "  -w, --width [value]       Maximum image width in pixels (default: 1200)"
  echo "  -o, --output [filename]   Output PDF filename"
  echo "  -h, --help                Show this help message"
}

# === DEPENDENCY CHECK ===
install_dependencies() {
  echo "🔍 Checking for required dependencies..."

  missing=()

  command -v pdftoppm >/dev/null 2>&1 || missing+=("poppler-utils")
  command -v convert >/dev/null 2>&1 || missing+=("imagemagick")
  command -v img2pdf >/dev/null 2>&1 || missing+=("img2pdf")

  if [ ${#missing[@]} -eq 0 ]; then
    echo "✅ All dependencies are already installed."
    return
  fi

  echo "⚠️ Missing dependencies: ${missing[*]}"
  
  if [[ "$OSTYPE" == "linux-gnu"* ]]; then
    echo "📦 Installing on Linux via apt..."
    sudo apt update
    sudo apt install -y poppler-utils imagemagick img2pdf
  elif [[ "$OSTYPE" == "darwin"* ]]; then
    echo "📦 Installing on macOS via Homebrew..."
    brew install poppler imagemagick img2pdf
  else
    echo "❌ Unsupported OS. Please install manually: ${missing[*]}"
    exit 1
  fi
}

# === PARSE FLAGS ===
POSITIONAL=()
while [[ $# -gt 0 ]]; do
  key="$1"
  case $key in
    -r|--dpi)
      DPI="$2"
      shift 2
      ;;
    -q|--quality)
      JPEG_QUALITY="$2"
      shift 2
      ;;
    -w|--width)
      MAX_WIDTH="$2"
      shift 2
      ;;
    -o|--output)
      OUTPUT_PDF="$2"
      shift 2
      ;;
    -h|--help)
      show_help
      exit 0
      ;;
    -*|--*)
      echo "❌ Unknown option $1"
      show_help
      exit 1
      ;;
    *)
      POSITIONAL+=("$1") # Save positional argument
      shift
      ;;
  esac
done
set -- "${POSITIONAL[@]}" # Restore positional arguments

# === MAIN FUNCTION ===
compress_pdf() {
  if [ -z "$1" ]; then
    echo "❌ Error: input PDF file is required."
    show_help
    exit 1
  fi

  INPUT_PDF="$1"
  BASENAME=$(basename "$INPUT_PDF" .pdf)
  TMP_DIR="./tmp_$BASENAME"

  if [ -z "$OUTPUT_PDF" ]; then
    OUTPUT_PDF="${BASENAME}_compressed.pdf"
  fi

  mkdir -p "$TMP_DIR"
  cd "$TMP_DIR" || exit 1

  echo "📄 Converting PDF pages to JPEG (DPI=$DPI, quality=$JPEG_QUALITY)..."
  pdftoppm -jpeg -jpegopt quality=$JPEG_QUALITY -r $DPI "../$INPUT_PDF" page

  echo "📏 Resizing images to max width ${MAX_WIDTH}px..."
  for img in page-*.jpg; do
    convert "$img" -resize ${MAX_WIDTH}x -quality $JPEG_QUALITY "$img"
  done

  echo "📦 Merging images into: $OUTPUT_PDF"
  img2pdf page-*.jpg -o "../$OUTPUT_PDF"

  echo "🧹 Cleaning up temporary files..."
  cd ..
  rm -r "$TMP_DIR"

  echo "✅ Done. Output PDF: $OUTPUT_PDF"
}

# === EXECUTION ===
install_dependencies
compress_pdf "$1"
