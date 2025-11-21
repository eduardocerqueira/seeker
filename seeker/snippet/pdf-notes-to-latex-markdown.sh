#date: 2025-11-21T16:52:45Z
#url: https://api.github.com/gists/998af19c92bc3bd3e7186a0b9fa81d48
#owner: https://api.github.com/users/aziis98

#!/bin/bash

read -r -d '' OCR_PROMPT <<'END_PROMPT'
You are an expert OCR system specialized in converting images of text and mathematical notes into well formatted markdown with LaTeX for mathematical formulas.

When you encounter figures in the page, do not attempt to describe them in detail. Instead, simply write [missing figure: <page>, <index-in-page>, <description>] where <description> is a brief description of what the figure represents (e.g., "graph of a function", "geometric diagram", "table of values"). Omit the page number in the bottom right corner of the page, now you will be given page PAGE_NUM.

Ensure that all mathematical expressions are accurately converted into LaTeX format and that the overall structure of the notes is preserved in markdown format (for no reason use unicode symbols instead of latex). Add a newline at the start of the output if the content starts mid-sentence. Here are some conventions to follow:

- Use `#`, `##`, `###`, etc. for headings.

- Use `$...$` for inline math

- Use `$$\n...\n$$` for display math and put the display math on its own line with each dollar sign on its own line.

- Use `-` for bullet points.

- Convert enumerated lists like i., ii., iii. into markdown numbered lists using `1.`, `2.`, `3.`, etc.

- Prefer using the `aligned` and `gathered` environments for multi-line equations.

- Use `\mathbb{R}` for the set of real numbers.

- Use `\mathbb{S}` for the unit sphere.

- Write lemmas, theorems, and definitions in bold like **Teorema 1.1.** and **Lemma 2.3.**

- Prefer using display math for lines that contain only mathematical expressions in inline math.

END_PROMPT

# INPUT_PDF="$1"

# OUTPUT_MD="$2"
# if [ -z "$OUTPUT_MD" ]; then
#     OUTPUT_MD="${INPUT_PDF%.pdf}.md"
# fi

# START_PAGE=1

# Parse options
START_PAGE=1

while [[ $# -gt 0 ]]; do
    case $1 in
        -s|--start-page)
            START_PAGE="$2"
            shift 2
            ;;
        -h|--help)
            echo "Usage: $0 [-s start_page] <input_pdf> [output_md]"
            exit 0
            ;;
        -*)
            echo "Unknown option: $1"
            exit 1
            ;;
        *)
            break
            ;;
    esac
done

INPUT_PDF="$1"
OUTPUT_MD="$2"
if [ -z "$OUTPUT_MD" ]; then
    OUTPUT_MD="${INPUT_PDF%.pdf}.md"
fi

# Convert each page of the PDF to an image using pdftoppm
mkdir -p pages
if [ -z "$(ls -A pages)" ]; then
    echo "Converting PDF pages to images..."
    pdftoppm -progress -png -r 200 "$INPUT_PDF" pages/page
fi

# Process each image with aichat and the defined OCR prompt
page_count=$(ls pages/page-*.png | wc -l)
for ((i=START_PAGE; i<=page_count; i++)); do
    page_num=$(printf "%03d" "$i")
    img="pages/page-${page_num}.png"
    
    if [ -f "$img" ]; then
        echo "Processing $img..."
        aichat -c \
               -m gemini:gemini-2.5-flash-lite \
               -f "$img" \
               --prompt "$(echo $OCR_PROMPT | sed "s/PAGE_NUM/${i}/g")" > "pages/page-${page_num}.txt"
    fi
done

# Combine all the text files into a single markdown file
cat pages/page-*.txt > "$OUTPUT_MD"
