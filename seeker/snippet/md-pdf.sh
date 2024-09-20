#date: 2024-09-20T17:00:22Z
#url: https://api.github.com/gists/85be7d83ceb6750adb6bc0ab496c3f6f
#owner: https://api.github.com/users/ichux

#!/bin/bash

# sudo apt install -y pandoc wkhtmltopdf texlive-xetex texlive-latex-extra

# Check if pandoc is installed
if ! command -v pandoc &> /dev/null; then
  echo "pandoc is not installed. Please install it and try again."
  exit 1
fi

# Directory containing .md files (current directory by default)
input_dir="${1:-.}"

# Convert all .md files in the directory to .pdf
for md_file in "$input_dir"/*.md; do
  if [ -f "$md_file" ]; then
    # Output file name
    output_file="${md_file%.md}.pdf"

    pandoc "$md_file" --pdf-engine=wkhtmltopdf -o "$output_file"
    
    # Check if the conversion was successful
    if [ $? -eq 0 ]; then
      echo "Converted: $md_file -> $output_file"
    else
      echo "Conversion failed for: $md_file"
    fi
  else
    echo "No .md files found in the directory."
    exit 1
  fi
done
