#date: 2026-01-22T17:03:51Z
#url: https://api.github.com/gists/b825431a7211b04357bd83a3f0f5618b
#owner: https://api.github.com/users/JeanHuit

#!/usr/bin/env bash

# ------------------------------------------------------------
# PDF Compression Script using Ghostscript
# Usage:
#   ./compress.sh input.pdf output.pdf screen|ebook|printer|prepress|default
# ------------------------------------------------------------

set -e

# Check argument count
if [ "$#" -ne 3 ]; then
    echo "Usage: $0 <input.pdf> <output.pdf> <screen|ebook|printer|prepress|default>"
    exit 1
fi

INPUT="$1"
OUTPUT="$2"
QUALITY="$3"

# Validate input file
if [ ! -f "$INPUT" ]; then
    echo "Error: Input file does not exist: $INPUT"
    exit 1
fi

# Validate quality option
case "$QUALITY" in
    screen|ebook|printer|prepress|default)
        PDFSETTINGS="/$QUALITY"
        ;;
    *)
        echo "Error: Invalid quality setting: $QUALITY"
        echo "Allowed values: screen, ebook, printer, prepress, default"
        exit 1
        ;;
esac

# Run Ghostscript
gs -sDEVICE=pdfwrite \
   -dCompatibilityLevel=1.4 \
   -dPDFSETTINGS="$PDFSETTINGS" \
   -dNOPAUSE -dQUIET -dBATCH \
   -sOutputFile="$OUTPUT" \
   "$INPUT"

echo "Compression completed:"
echo "  Input : $INPUT"
echo "  Output: $OUTPUT"
echo "  Mode  : $QUALITY"
