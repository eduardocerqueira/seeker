#date: 2025-11-20T17:13:21Z
#url: https://api.github.com/gists/baff2457f6ba6f90fb1b92f91b136217
#owner: https://api.github.com/users/anubhaw2091

#!/bin/bash
# Step 5: Deploy converted GGUF model to Ollama
#

set -e

echo "Step 5: Deploy Model to Ollama"
echo ""

# Default paths
MODEL_NAME="${1:-mymodel}"
LLAMA_CPP_PATH="${2:-${LLAMA_CPP_PATH:-}}"

echo "Configuration:"
echo "  Model name: $MODEL_NAME"
echo ""

# Find llama.cpp directory (same logic as step4)
LLAMA_CPP_DIR=""

if [ -n "$LLAMA_CPP_PATH" ] && [ -d "$LLAMA_CPP_PATH" ]; then
    LLAMA_CPP_DIR="$LLAMA_CPP_PATH"
    echo "  Using llama.cpp path: $LLAMA_CPP_DIR"
elif [ -f "convert-hf-to-gguf.py" ] || [ -f "convert_hf_to_gguf.py" ]; then
    LLAMA_CPP_DIR="$(pwd)"
    echo "  Using llama.cpp from current directory"
else
    COMMON_PATHS=(
        "$HOME/Documents/local_llm/llama.cpp"
        "$HOME/llama.cpp"
        "/usr/local/llama.cpp"
        "./llama.cpp"
    )
    
    for path in "${COMMON_PATHS[@]}"; do
        if [ -f "$path/convert-hf-to-gguf.py" ] || [ -f "$path/convert_hf_to_gguf.py" ]; then
            LLAMA_CPP_DIR="$path"
            echo "  Found llama.cpp at: $LLAMA_CPP_DIR"
            break
        fi
    done
fi

if [ -z "$LLAMA_CPP_DIR" ]; then
    echo " llama.cpp directory not found"
    echo "  Please provide path: bash step5_deploy_ollama.sh <model_name> <llama.cpp_path>"
    exit 1
fi

# Look for GGUF file in custom_models directory structure
CUSTOM_MODELS_DIR="$LLAMA_CPP_DIR/custom_models"
MODEL_DIR="$CUSTOM_MODELS_DIR/$MODEL_NAME"
BASE_GGUF="$MODEL_DIR/$MODEL_NAME.gguf"

# Prefer quantized versions (in order of preference: q8_0, q4_k_m, q4_0, then base)
GGUF_FILE=""
QUANT_TYPES=("q8_0" "q4_k_m" "q4_0" "q5_k_m" "q5_0")

for quant_type in "${QUANT_TYPES[@]}"; do
    quantized_file="$MODEL_DIR/${MODEL_NAME}_${quant_type}.gguf"
    if [ -f "$quantized_file" ]; then
        GGUF_FILE="$quantized_file"
        GGUF_NAME="${MODEL_NAME}_${quant_type}.gguf"
        echo "  Found quantized GGUF file: $GGUF_FILE (${quant_type})"
        break
    fi
done

# Fallback to base F16 if no quantized version found
if [ -z "$GGUF_FILE" ]; then
    if [ -f "$BASE_GGUF" ]; then
        GGUF_FILE="$BASE_GGUF"
        GGUF_NAME="$MODEL_NAME.gguf"
        echo "  Found base GGUF file: $GGUF_FILE (F16)"
    else
        echo " GGUF file not found: $BASE_GGUF"
        echo ""
        echo "Expected location: $BASE_GGUF"
        echo ""
        echo "Make sure you've run step4_convert_to_gguf.sh first to create the GGUF file."
        exit 1
    fi
fi

# Check if Ollama is running
if ! curl -s http://127.0.0.1:11434/api/tags > /dev/null 2>&1; then
    echo " Ollama doesn't appear to be running"
    echo ""
    echo "Please start Ollama in any terminal and keep it running:"
    echo "  ollama serve"
    echo ""
    exit 1
fi

echo " Creating Modelfile..."

# Create Modelfile in the same directory as the GGUF file (MODEL_DIR)
MODELFILE="$MODEL_DIR/Modelfile"

# Change to model directory and use relative path in Modelfile
cd "$MODEL_DIR"

# Create Modelfile with relative path to GGUF (same directory)
# Use the detected GGUF file name (quantized or base)
cat > "$MODELFILE" << EOF
FROM ./$GGUF_NAME
PARAMETER temperature 0.2
PARAMETER top_p 0.9
TEMPLATE """{{ .Prompt }}"""
EOF

echo " Modelfile created: $MODELFILE"
echo ""
echo "Modelfile content:"
cat "$MODELFILE"
echo ""

echo " Registering model with Ollama..."
# Create model from the model directory (using ./ prefix format that works)
# ollama create mymodel -f Modelfile
# if ! ollama create "$MODEL_NAME" -f "$MODELFILE"; then
if ! /opt/anaconda3/bin/ollama create "$MODEL_NAME" -f "Modelfile"; then
    echo ""
    echo " Failed to create model with Ollama"
    echo "current directory: $(pwd)"
    echo "Used the following command: ollama create $MODEL_NAME -f Modelfile"
    echo ""
    echo "Troubleshooting:"
    echo "  1. Ensure Ollama is running: ollama serve"
    echo "  2. Check the Modelfile content above"
    echo "  3. Verify the GGUF file exists: $GGUF_FILE"
    echo "  4. Try manually:"
    echo "     cd $MODEL_DIR"
    echo "     ollama create $MODEL_NAME -f Modelfile"
    echo ""
    exit 1
fi

echo ""
echo " Model Deployed Successfully!"
echo ""
echo "Model name: $MODEL_NAME"
echo "Modelfile saved at: $MODELFILE"
echo ""
echo "Test the model:"
echo "  ollama run $MODEL_NAME 'Hello, how are you?'"
echo ""
echo "Or use the API:"
echo "  curl http://127.0.0.1:11434/api/generate -d '{\"model\": \"$MODEL_NAME\", \"prompt\": \"Hello\"}'"
echo ""
echo "Next step: Run step6_validate_outputs.py to validate outputs"
echo ""

