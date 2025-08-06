#date: 2025-08-06T17:20:13Z
#url: https://api.github.com/gists/911e3831b0def4694a367b3d341ce6df
#owner: https://api.github.com/users/RobotSail

#!/bin/bash
#
# Clean Reinstall Script - Completely rebuild environment without cached conflicts
#

set -eo pipefail

# Set cache-busting environment variables
export UV_NO_CACHE=1
export PIP_NO_CACHE_DIR=1
export TRANSFORMERS_CACHE="/tmp/transformers-cache-$$"
export HF_HOME="/tmp/hf-cache-$$"
export TORCH_HOME="/tmp/torch-cache-$$"
export TRITON_CACHE_DIR="/tmp/triton-cache-$$"

# Function to clean and uninstall
clean_uninstall() {
    echo "🧹 Starting clean uninstall process..."
    
    # Check if we're in a virtual environment
    if [[ -z "$VIRTUAL_ENV" ]]; then
        if [[ -d ".dep-venv" ]]; then
            source .dep-venv/bin/activate
        else
            echo "❌ No virtual environment found. Please activate one first."
            exit 1
        fi
    fi
    
    echo "✅ Using virtual environment: $VIRTUAL_ENV"
    
    
    # Clear all caches
    echo "🗑️  Clearing caches..."
    uv cache clean 2>/dev/null || true
    pip cache purge 2>/dev/null || true
    
    # Uninstall everything except core tools
    echo "🗑️  Uninstalling all packages..."
    PACKAGES=$(uv pip list --format=freeze | grep -v "^pip=\|^setuptools=\|^wheel=" | cut -d'=' -f1 || true)
    if [[ -n "$PACKAGES" ]]; then
        echo "$PACKAGES" | xargs -r uv pip uninstall 
    fi
    
    echo "🧹 Clean uninstall complete!"
}

# Function to install with uv pip
install_uv_pip() {
    echo "📦 Installing with uv pip..."
    
    # Install base package first (without flash-attn)
    echo "📦 Installing base package..."
    uv pip install --no-cache --force-reinstall -e .
    
    # Install flash-attn after PyTorch is available
    echo "⚡ Installing flash-attn..."
    uv pip install --no-cache --no-build-isolation -e .[cuda]
    
    # Verify installation
    echo "🔍 Verifying installation..."
    python -c "
import torch; print(f'✅ PyTorch: {torch.__version__}')
import flash_attn; print(f'✅ Flash Attention: {flash_attn.__version__}')
import transformers; print(f'✅ Transformers: {transformers.__version__}')
import typer; print(f'✅ Typer: {typer.__version__}')
"
    
    echo "🎉 uv pip install complete!"
}

# Function to install with uv sync
install_uv_sync() {
    echo "📦 Installing with uv sync..."
    
    # Install base package first 
    echo "📦 Installing base package..."
    uv sync --no-cache
    
    # Install flash-attn separately due to build dependency issues
    echo "⚡ Installing flash-attn (using uv pip due to build dependencies)..."
    # uv pip install --no-cache --no-build-isolation flash-attn>=2.8.2
    uv pip install -e .[cuda]
    
    # Verify installation
    echo "🔍 Verifying installation..."
    python -c "
import torch; print(f'✅ PyTorch: {torch.__version__}')
import flash_attn; print(f'✅ Flash Attention: {flash_attn.__version__}')
import transformers; print(f'✅ Transformers: {transformers.__version__}')
import typer; print(f'✅ Typer: {typer.__version__}')
"
    
    echo "🎉 uv sync install complete!"
}

# Main script logic
case "${1:-all}" in
    "uninstall")
        clean_uninstall
        ;;
    "uv-pip")
        clean_uninstall
        install_uv_pip
        ;;
    "uv-sync")
        clean_uninstall
        install_uv_sync
        ;;
    "all")
        clean_uninstall
        install_uv_pip
        ;;
    *)
        echo "Usage: $0 [uninstall|uv-pip|uv-sync|all]"
        echo "  uninstall: Only clean and uninstall packages"
        echo "  uv-pip:    Clean uninstall + reinstall with uv pip"
        echo "  uv-sync:   Clean uninstall + reinstall with uv sync"
        echo "  all:       Clean uninstall + reinstall with uv pip (default)"
        exit 1
        ;;
esac
