#date: 2025-04-15T16:59:42Z
#url: https://api.github.com/gists/59eb297f67171867e80bea964f757f55
#owner: https://api.github.com/users/jamesbrink

#!/usr/bin/env bash
# Script to create an Ollama model with a custom context window size
# Usage: ./create_context_model.sh <source_model> <context_size_in_k>
# Example: ./create_context_model.sh qwen2.5-coder:14b 32

set -e

# Check if Ollama is installed
if ! command -v ollama &> /dev/null; then
    echo "Error: Ollama is not installed or not in PATH"
    exit 1
fi

# Function to display help message
show_help() {
    echo "Create an Ollama model with a custom context window size"
    echo ""
    echo "Usage: $0 <source_model> <context_size_in_k>"
    echo ""
    echo "Arguments:"
    echo "  <source_model>       The name of an existing Ollama model (e.g., qwen2.5-coder:14b)"
    echo "  <context_size_in_k>  The desired context window size in K (e.g., 8, 16, 32, 64)"
    echo ""
    echo "Examples:"
    echo "  $0 qwen2.5-coder:14b 32     # Creates qwen2.5-coder-32k:14b with 32K context"
    echo "  $0 llama3:8b 16            # Creates llama3-16k:8b with 16K context"
    echo "  $0 phi3:mini 8             # Creates phi3-8k:mini with 8K context"
    echo ""
    echo "Available models:"
    ollama list | awk '{print "  "$1}' | tail -n +2
}

# Check arguments
if [ $# -lt 2 ]; then
    show_help
    exit 1
fi

# Check if help was requested
if [[ "$1" == "-h" || "$1" == "--help" ]]; then
    show_help
    exit 0
fi

SOURCE_MODEL=$1
CONTEXT_SIZE=$2

# Validate source model exists
if ! ollama list | grep -q "$SOURCE_MODEL"; then
    echo "Error: Source model '$SOURCE_MODEL' not found in Ollama"
    echo "Available models:"
    ollama list | awk '{print "  "$1}' | tail -n +2
    exit 1
fi

# Validate context size is a positive number
if ! [[ "$CONTEXT_SIZE" =~ ^[0-9]+$ ]] || [ "$CONTEXT_SIZE" -lt 1 ]; then
    echo "Error: Context size must be a positive integer"
    echo "You provided: '$CONTEXT_SIZE'"
    echo "Example valid values: 8, 16, 32, 64"
    exit 1
fi

# Convert context size from K to actual number
CONTEXT_TOKENS= "**********"

# Extract base model name and tag
if [[ "$SOURCE_MODEL" == *":"* ]]; then
    MODEL_NAME=$(echo "$SOURCE_MODEL" | cut -d':' -f1)
    MODEL_TAG=$(echo "$SOURCE_MODEL" | cut -d':' -f2)
    TARGET_MODEL="${MODEL_NAME}-${CONTEXT_SIZE}k:${MODEL_TAG}"
else
    MODEL_NAME="$SOURCE_MODEL"
    TARGET_MODEL="${MODEL_NAME}-${CONTEXT_SIZE}k"
fi

echo "Creating model '$TARGET_MODEL' with ${CONTEXT_SIZE}k (${CONTEXT_TOKENS}) context window..."

# Create temporary directory
TEMP_DIR=$(mktemp -d)
trap 'rm -rf "$TEMP_DIR"' EXIT

# Create Modelfile
cat > "$TEMP_DIR/Modelfile" << EOF
FROM $SOURCE_MODEL

# Set context window to ${CONTEXT_SIZE}k
PARAMETER num_ctx $CONTEXT_TOKENS

# Other optional parameters for better performance
PARAMETER num_thread 8
PARAMETER num_gpu 50
PARAMETER temperature 0.7
PARAMETER top_p 0.9
PARAMETER top_k 40
EOF

# Create the model
echo "Creating model using Modelfile..."
ollama create "$TARGET_MODEL" -f "$TEMP_DIR/Modelfile"

# Verify the model was created
if ollama list | grep -q "$TARGET_MODEL"; then
    echo "✅ Success! Model '$TARGET_MODEL' created with ${CONTEXT_SIZE}k context window"
    echo "Run it with: ollama run $TARGET_MODEL"
else
    echo "❌ Error: Failed to create model '$TARGET_MODEL'"
    exit 1
fi
 exit 1
fi
