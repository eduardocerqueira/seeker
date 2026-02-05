#date: 2026-02-05T17:40:03Z
#url: https://api.github.com/gists/9610a93371a294afebc20cb8698825dc
#owner: https://api.github.com/users/renoirb

#!/bin/bash
set -e

INPUT_FILE="$1"
MODEL="${2:-base}"
LANGUAGE="${3:-en}"

if [ -z "$INPUT_FILE" ]; then
    echo "Usage: $0 <audio_file> [model] [language]"
    echo ""
    echo "Examples:"
    echo "  $0 recording.m4a"
    echo "  $0 recording.m4a base en"
    echo "  $0 recording.m4a medium en"
    echo ""
    echo "Available models: tiny, base, small, medium, large-v2, large-v3"
    exit 1
fi

if [ ! -f "$INPUT_FILE" ]; then
    echo "Error: File '$INPUT_FILE' not found"
    exit 1
fi

# Activate venv if it exists
if [ -d "venv" ]; then
    echo "Activating virtual environment..."
    source venv/bin/activate
fi

# Transcribe
echo "Transcribing with model: $MODEL, language: $LANGUAGE"
whisper "$INPUT_FILE" \
    --model "$MODEL" \
    --language "$LANGUAGE" \
    --output_format txt \
    --output_dir "$(dirname "$INPUT_FILE")"

echo "Transcription complete!"
