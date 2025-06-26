#date: 2025-06-26T16:57:00Z
#url: https://api.github.com/gists/381f824d17667041bb0cd9ede3aa300d
#owner: https://api.github.com/users/erman-gurses

#!/bin/bash

set -uxo pipefail

# Set default INDEX_URL if not provided
INDEX_URL="${INDEX_URL:-https://d2awnip2yjpvqn.cloudfront.net/v2/gfx94X-dcgpu/}"

# Determine script and project root directories
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="$(realpath "$SCRIPT_DIR/../..")"
PYTORCH_DIR="$ROOT_DIR/external-builds/pytorch/pytorch"

# Check Python availability
if ! command -v python3.11 >/dev/null 2>&1; then
  echo "Error: python3.11 not found. Please install Python 3.11."
  exit 1
fi

# Check for requirements.txt
REQUIREMENTS_FILE="$ROOT_DIR/requirements.txt"
if [ ! -f "$REQUIREMENTS_FILE" ]; then
  echo "Error: requirements.txt not found at $REQUIREMENTS_FILE"
  exit 1
fi

# Clean and create virtual environment (.venv)
VENV_DIR="$ROOT_DIR/.venv"
rm -rf "$VENV_DIR" || {
  echo "Error: Failed to clean virtual environment at $VENV_DIR"
  exit 1
}
python3.11 -m venv "$VENV_DIR" || {
  echo "Error: Failed to create virtual environment at $VENV_DIR"
  exit 1
}
source "$VENV_DIR/bin/activate" || {
  echo "Error: Failed to activate virtual environment (.venv)"
  exit 1
}

# Set trap to deactivate virtual environment on exit
trap "deactivate 2> /dev/null" EXIT

# Install dependencies
echo "Installing dependencies..."
python -m pip install --upgrade pip
python -m pip install --index-url "$INDEX_URL" torch --force-reinstall -v
python -m pip install pytest>=8.0.0 pytest-xdist>=3.5.0 numpy>=1.24.0 psutil>=5.9.0 expecttest>=0.2.0 hypothesis>=6.75.0 -v
python -m pip install -r "$REQUIREMENTS_FILE" -v

# Verify installations
echo "Verifying installations:"
python -m pip show torch pytest pytest-xdist numpy psutil expecttest hypothesis
python -c "import torch; print('torch.__file__:', torch.__file__); print('torch.__version__:', torch.__version__)"
python -c "import pytest; print('pytest.__version__:', pytest.__version__)"

# Checkout repositories
cd "$ROOT_DIR"
if [ ! -f "external-builds/pytorch/pytorch_torch_repo.py" ]; then
  echo "Error: pytorch_torch_repo.py not found at external-builds/pytorch/"
  exit 1
fi
./external-builds/pytorch/pytorch_torch_repo.py checkout --no-hipify --no-patch

# Set PYTHONPATH for site-packages and pytest_shard_custom
PYTHON_VERSION="3.11"
export PYTHONPATH="$VENV_DIR/lib/python${PYTHON_VERSION}/site-packages:$PYTORCH_DIR/test"

# Temporarily rename torch directory to prevent local import
if [ -d "$PYTORCH_DIR/torch" ]; then
  mv "$PYTORCH_DIR/torch" "$PYTORCH_DIR/torch.bak" || {
    echo "Error: Failed to rename $PYTORCH_DIR/torch"
    exit 1
  }
fi
trap "mv '$PYTORCH_DIR/torch.bak' '$PYTORCH_DIR/torch' 2> /dev/null; deactivate 2> /dev/null" EXIT

export PYTORCH_PRINT_REPRO_ON_FAILURE=0
export PYTORCH_TEST_WITH_ROCM=1
export MIOPEN_DISABLE_CACHE=1

# Run pytest without exiting on error
set +e
echo "Running pytest..."
python -m pytest \
  "$PYTORCH_DIR/test/test_nn.py" \
  "$PYTORCH_DIR/test/test_torch.py" \
  "$PYTORCH_DIR/test/test_cuda.py" \
  "$PYTORCH_DIR/test/test_ops.py" \
  "$PYTORCH_DIR/test/test_unary_ufuncs.py" \
  "$PYTORCH_DIR/test/test_binary_ufuncs.py" \
  "$PYTORCH_DIR/test/test_autograd.py" \
  "$PYTORCH_DIR/test/inductor/test_torchinductor.py" \
  -v \
  --continue-on-collection-errors \
  --import-mode=importlib \
  -k "not test_unused_output_device_cuda and not test_pinned_memory_empty_cache" \
  --maxfail=0 \
  -n 0
PYTEST_EXIT_STATUS=$?
set -e
if [ $PYTEST_EXIT_STATUS -ne 0 ]; then
  echo "pytest failed with exit code $PYTEST_EXIT_STATUS"
fi
