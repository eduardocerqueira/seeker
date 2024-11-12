#date: 2024-11-12T16:58:29Z
#url: https://api.github.com/gists/c4c746d43a4ee3bd50a05b5c9cfb924e
#owner: https://api.github.com/users/gungurbuz

#!/bin/bash

# Check if filename argument is provided
if [ -z "$1" ]; then
    echo "Usage: $0 <filename.tex> [-x|-p]"
    echo "Options:"
    echo "  -x   Use xelatex for compilation"
    echo "  -p   Use pdflatex for compilation (default)"
    exit 1
fi

# Set filename without extension and default compiler
filename="${1%.tex}"
compiler="pdflatex"

# Check for compiler choice
while getopts ":xp" opt; do
  case $opt in
    x)
      compiler="xelatex"
      ;;
    p)
      compiler="pdflatex"
      ;;
    \?)
      echo "Invalid option: -$OPTARG" >&2
      exit 1
      ;;
  esac
done

# Step 1: Parse the .tex file for dependencies until \begin{document}
required_packages=()
bib_required=false
graphics_required=false

while IFS= read -r line; do
    # Stop reading after \begin{document}
    [[ "$line" =~ \\begin\{document\} ]] && break

    # Check for general packages
    if [[ "$line" =~ \\usepackage\{(.+)\} ]]; then
        required_packages+=("${BASH_REMATCH[1]}")
    fi

    # Check if biblatex is used (requires biber)
    if [[ "$line" =~ \\usepackage\{biblatex\} ]]; then
        bib_required=true
    fi

    # Check if graphics or TikZ is required
    if [[ "$line" =~ \\usepackage\{(graphicx|tikz)\} ]]; then
        graphics_required=true
    fi
done < "$1"

# Display detected packages
echo "Detected packages: ${required_packages[*]}"
echo "Bibliography required: $bib_required"
echo "Graphics required: $graphics_required"

# Step 2: Compile the LaTeX document
echo "Compiling LaTeX document with $compiler..."

# Run the chosen compiler to generate auxiliary files
$compiler "$filename.tex"

# If biber is needed, run biber for bibliography management
if $bib_required; then
    biber "$filename"
fi

# Run the chosen compiler two more times to resolve references and finalize the document
$compiler "$filename.tex"
$compiler "$filename.tex"

# Clean up auxiliary files
echo "Cleaning up auxiliary files..."
rm -f "$filename.aux" "$filename.bbl" "$filename.bcf" "$filename.blg" "$filename.log" "$filename.out" "$filename.run.xml" "$filename.toc"

echo "Compilation complete. Output PDF: $filename.pdf"
