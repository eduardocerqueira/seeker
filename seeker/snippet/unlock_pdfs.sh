#date: 2025-04-09T16:59:41Z
#url: https://api.github.com/gists/809c04402111ac5444a248e49503321b
#owner: https://api.github.com/users/fabsch225

#!/bin/bash

PDF_DIR="${1:-.}"
read -sp "Enter PDF password: "**********"
echo ""
OUTPUT_DIR="$PDF_DIR/unlocked_pdfs"
mkdir -p "$OUTPUT_DIR"

for file in "$PDF_DIR"/*.pdf; do
    # Check if file exists
    [ -e "$file" ] || continue

    # Output file path
    output_file="$OUTPUT_DIR/$(basename "$file")"

    # Unlock the PDF
    qpdf --password= "**********"

    echo "Unlocked: $(basename "$file") -> $output_file"
done

echo "All PDFs have been processed. Unlocked versions are in: $OUTPUT_DIR"
locked versions are in: $OUTPUT_DIR"
