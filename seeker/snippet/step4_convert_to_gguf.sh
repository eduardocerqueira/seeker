#date: 2025-11-20T17:08:34Z
#url: https://api.github.com/gists/9311d687fae1438defda7a4efa658afe
#owner: https://api.github.com/users/anubhaw2091

#!/bin/bash
# Step 4: Convert HuggingFace model to GGUF format using llama.cpp
#
# This script provides instructions and commands for converting
# the merged HuggingFace model to GGUF format for Ollama deployment.
#
# Prerequisites:
#   git clone https://github.com/ggerganov/llama.cpp
#   cd llama.cpp
#   pip install -r requirements.txt

set -e

echo "Step 4: Convert HuggingFace Model to GGUF Format"
echo ""

# Default paths
MERGED_MODEL_PATH="${1:-merged_model}"
MODEL_NAME="${2:-mymodel}"  # Model name (used for directory structure)
QUANT_TYPE="${3:-f16}"  # Default to f16 (half precision). Valid: f32, f16, bf16, q8_0, tq1_0, tq2_0, auto
LLAMA_CPP_PATH="${4:-${LLAMA_CPP_PATH:-}}"

# Valid quantization types for convert_hf_to_gguf.py
VALID_QUANT_TYPES=("f32" "f16" "bf16" "q8_0" "tq1_0" "tq2_0" "auto")

echo "Configuration:"
echo "  Merged model path: $MERGED_MODEL_PATH"
echo "  Model name: $MODEL_NAME"
echo "  Quantization type: $QUANT_TYPE"
echo ""

# Validate quantization type
if [[ ! " ${VALID_QUANT_TYPES[@]} " =~ " ${QUANT_TYPE} " ]]; then
    echo "  Warning: Invalid quantization type '$QUANT_TYPE'"
    echo "  Valid types: ${VALID_QUANT_TYPES[*]}"
    echo "  Using default: f16"
    echo ""
    echo "  Note: For q4_k_m quantization, first convert to GGUF (f16 recommended),"
    echo "        then use llama-quantize tool for further quantization."
    echo ""
    QUANT_TYPE="f16"
fi
echo ""

# Find llama.cpp directory
LLAMA_CPP_DIR=""

# Check if provided as argument or environment variable
if [ -n "$LLAMA_CPP_PATH" ] && [ -d "$LLAMA_CPP_PATH" ]; then
    LLAMA_CPP_DIR="$LLAMA_CPP_PATH"
    echo "  Using llama.cpp path: $LLAMA_CPP_DIR"
elif [ -f "convert-hf-to-gguf.py" ] || [ -f "convert_hf_to_gguf.py" ]; then
    # Running from llama.cpp directory
    LLAMA_CPP_DIR="$(pwd)"
    echo "  Using llama.cpp from current directory"
else
    # Try common locations
    COMMON_PATHS=(
        "$HOME/Documents/local_llm/llama.cpp"
        "$HOME/llama.cpp"
        "/usr/local/llama.cpp"
        "./llama.cpp"
    )
    
    for path in "${COMMON_PATHS[@]}"; do
        # Check for both naming conventions (hyphens and underscores)
        if [ -f "$path/convert-hf-to-gguf.py" ] || [ -f "$path/convert_hf_to_gguf.py" ]; then
            LLAMA_CPP_DIR="$path"
            echo "  Found llama.cpp at: $LLAMA_CPP_DIR"
            break
        fi
    done
fi

# Find the actual conversion script name
CONVERT_SCRIPT=""
if [ -n "$LLAMA_CPP_DIR" ]; then
    if [ -f "$LLAMA_CPP_DIR/convert-hf-to-gguf.py" ]; then
        CONVERT_SCRIPT="convert-hf-to-gguf.py"
    elif [ -f "$LLAMA_CPP_DIR/convert_hf_to_gguf.py" ]; then
        CONVERT_SCRIPT="convert_hf_to_gguf.py"
    fi
fi

# Verify llama.cpp is found
if [ -z "$LLAMA_CPP_DIR" ] || [ -z "$CONVERT_SCRIPT" ]; then
    echo "  llama.cpp conversion script not found"
    echo ""
    echo " To set up llama.cpp (REQUIRED for conversion):"
    echo "  1. git clone https://github.com/ggerganov/llama.cpp"
    echo "  2. cd llama.cpp"
    echo "  3. pip install -r requirements.txt"
    echo ""
    echo "Then run this script with:"
    echo "  bash step4_convert_to_gguf.sh <merged_model_path> <output.gguf> <quant_type> <llama.cpp_path>"
    echo ""
    echo "Or set environment variable:"
    echo "  export LLAMA_CPP_PATH=/path/to/llama.cpp"
    echo "  bash step4_convert_to_gguf.sh <merged_model_path> <output.gguf> <quant_type>"
    echo ""
    echo "Or run from llama.cpp directory:"
    echo "  cd llama.cpp"
    echo "  bash /path/to/step4_convert_to_gguf.sh <merged_model_path> <output.gguf> <quant_type>"
    echo ""
    exit 1
fi

echo ""

# Resolve absolute paths
if [ ! -d "$MERGED_MODEL_PATH" ]; then
    # Try relative to current directory
    if [ -d "$(pwd)/$MERGED_MODEL_PATH" ]; then
        MERGED_MODEL_PATH="$(cd "$(dirname "$(pwd)/$MERGED_MODEL_PATH")" && pwd)/$(basename "$MERGED_MODEL_PATH")"
    else
        echo " Merged model path not found: $MERGED_MODEL_PATH"
        exit 1
    fi
else
    MERGED_MODEL_PATH="$(cd "$(dirname "$MERGED_MODEL_PATH")" && pwd)/$(basename "$MERGED_MODEL_PATH")"
fi

# Create custom_models directory structure
CUSTOM_MODELS_DIR="$LLAMA_CPP_DIR/custom_models"
MODEL_DIR="$CUSTOM_MODELS_DIR/$MODEL_NAME"
OUTPUT_GGUF="$MODEL_DIR/$MODEL_NAME.gguf"

# Create directory structure
mkdir -p "$MODEL_DIR"
echo "  Created directory: $MODEL_DIR"

echo " Converting HuggingFace model to GGUF..."
echo "  Using conversion script: $LLAMA_CPP_DIR/$CONVERT_SCRIPT"
echo "  Input model: $MERGED_MODEL_PATH"
echo "  Output file: $OUTPUT_GGUF"
echo ""

# Determine which Python to use (explicitly use conda environment)
PYTHON_CMD="python3"
CONDA_ENV="coding_practise"

if command -v conda &> /dev/null; then
    # Check if conda environment exists
    if conda env list | grep -q "^${CONDA_ENV}[[:space:]]"; then
        PYTHON_CMD="conda run -n ${CONDA_ENV} python"
        echo "  Using conda environment: ${CONDA_ENV}"
    else
        echo "  Warning: conda environment '${CONDA_ENV}' not found, using system python3"
    fi
else
    echo "  Warning: conda not found, using system python3"
fi

# Run conversion from llama.cpp directory
cd "$LLAMA_CPP_DIR"
$PYTHON_CMD "$CONVERT_SCRIPT" "$MERGED_MODEL_PATH" \
    --outfile "$OUTPUT_GGUF" \
    --outtype "$QUANT_TYPE"

if [ $? -eq 0 ]; then
    echo ""
    echo "============================================================"
    echo " Conversion Successful!"
    echo "============================================================"
    echo ""
    echo "GGUF file created: $OUTPUT_GGUF"
    echo "Modelfile location: $MODEL_DIR/Modelfile (will be created in step5)"
    echo ""
    echo "Next step: Deploy to Ollama using step5_deploy_ollama.sh"
    echo "  Command: bash step5_deploy_ollama.sh $MODEL_NAME"
    echo ""
else
    echo ""
    echo " Conversion failed. Check the error messages above."
    exit 1
fi

