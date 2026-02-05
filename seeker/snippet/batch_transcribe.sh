#date: 2026-02-05T17:40:03Z
#url: https://api.github.com/gists/9610a93371a294afebc20cb8698825dc
#owner: https://api.github.com/users/renoirb

#!/bin/bash
set -e

MODEL="${1:-base}"
LANGUAGE="${2:-en}"

# Activate venv
if [ -d "venv" ]; then
    echo "Activating virtual environment..."
    source venv/bin/activate
else
    echo "Error: venv directory not found"
    echo "Please run setup first:"
    echo "  python3 -m venv venv"
    echo "  source venv/bin/activate"
    echo "  pip install openai-whisper"
    exit 1
fi

echo "Processing all .m4a files in current directory"
echo "Model: $MODEL, Language: $LANGUAGE"
echo ""

count=0
for file in *.m4a; do
    if [ -f "$file" ]; then
        count=$((count + 1))
        echo "[$count] Processing: $file"
        whisper "$file" \
            --model "$MODEL" \
            --language "$LANGUAGE" \
            --output_format txt
        echo "---"
    fi
done

if [ $count -eq 0 ]; then
    echo "No .m4a files found in current directory"
else
    echo "Batch transcription complete! Processed $count files."
fi
