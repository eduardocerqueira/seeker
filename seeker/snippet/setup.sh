#date: 2026-02-05T17:40:03Z
#url: https://api.github.com/gists/9610a93371a294afebc20cb8698825dc
#owner: https://api.github.com/users/renoirb

#!/bin/bash
set -e

PYTHON="${PYTHON:-python3}"
VENV_DIR="venv"

# Check prerequisites
if ! command -v "$PYTHON" &>/dev/null; then
    echo "Error: $PYTHON not found. Install Python 3.11+ first."
    echo "  macOS: brew install python@3.11"
    echo "  Linux: sudo apt-get install python3 python3-venv"
    exit 1
fi

if ! command -v ffmpeg &>/dev/null; then
    echo "Error: ffmpeg not found. Required for audio processing."
    echo "  macOS: brew install ffmpeg"
    echo "  Linux: sudo apt-get install ffmpeg"
    exit 1
fi

# Show Python version
echo "Using: $($PYTHON --version)"

# Create venv
if [ -d "$VENV_DIR" ]; then
    echo "venv already exists. Delete it first to recreate:"
    echo "  rm -rf $VENV_DIR && ./setup.sh"
    exit 1
fi

echo "Creating virtual environment..."
"$PYTHON" -m venv "$VENV_DIR"

echo "Installing dependencies..."
"$VENV_DIR/bin/pip" install --upgrade pip --quiet
"$VENV_DIR/bin/pip" install -r requirements.txt

# Make scripts executable
chmod +x transcribe.py transcribe_faster.py transcribe.sh batch_transcribe.sh

echo ""
echo "Setup complete. Usage:"
echo ""
echo "  source venv/bin/activate"
echo "  python transcribe_faster.py your-audio.m4a"
echo ""
echo "Or specify a model (tiny/base/small/medium/large-v3):"
echo ""
echo "  python transcribe.py your-audio.m4a --model medium"
