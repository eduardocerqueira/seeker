#date: 2025-10-03T16:52:57Z
#url: https://api.github.com/gists/4286fd48c96e98bf8b4f5d30d12c97e5
#owner: https://api.github.com/users/ImmarKarim

#!/bin/bash 
set -euo pipefail

apt-get update
apt-get install libmagic-dev -y
apt-get install libmagic1 -y

export COMFYUI_API_BASE=${COMFYUI_API_BASE:-http://127.0.0.1:8188}
export BACKEND=${BACKEND:-comfyui-json}
export MODEL_SERVER_URL=${MODEL_SERVER_URL:-http://localhost:8000}
export MODEL_HEALTH_ENDPOINT=${MODEL_HEALTH_ENDPOINT:-http://localhost:8000/health}

# Service binding and paths (override if needed)
export WRAPPER_HOST=${WRAPPER_HOST:-0.0.0.0}
export WRAPPER_PORT=${WRAPPER_PORT:-8000}
export WRAPPER_DIR=${WRAPPER_DIR:-/root/comfyui-api-wrapper}
export COMFYUI_DIR=${COMFYUI_DIR:-/ComfyUI}
export COMFYUI_HOST=${COMFYUI_HOST:-0.0.0.0}
export COMFYUI_PORT=${COMFYUI_PORT:-8188}


[[ -f ~/.local/bin/env ]] && source ~/.local/bin/env

if ! which uv; then
    curl -LsSf https://astral.sh/uv/install.sh | sh
    . ~/.local/bin/env
fi

if [[ ! -d "${WRAPPER_DIR}" ]]; then
    git clone https://github.com/ImmarKarim/comfyui-api-wrapper.git "${WRAPPER_DIR}"
fi

if [[ ! -d  "${WRAPPER_DIR}/.venv" ]]; then
    uv venv -p 3.12 "${WRAPPER_DIR}/.venv"
    (
        set -e
        . "${WRAPPER_DIR}/.venv/bin/activate"
        uv pip install -r "${WRAPPER_DIR}/requirements.txt"
    )
fi

    # Function to start API wrapper with readiness + auto-restart  
    start_api_wrapper() {
        while true; do
            echo "Starting API wrapper on ${WRAPPER_HOST}:${WRAPPER_PORT} ..."
            bash -lc "
                cd '${WRAPPER_DIR}'
                . .venv/bin/activate
                uvicorn main:app --host '${WRAPPER_HOST}' --port '${WRAPPER_PORT}' 2>&1 | tee -a /wrapper.log
            "
            EXIT_CODE=$?
            echo "API wrapper exited with code: ${EXIT_CODE}. Restarting in 2s..."
            sleep 2
        done
    }

# Best-effort Jupyter (non-critical)
[ -x /opt/jupyter.sh ] && /opt/jupyter.sh || true &

# Start wrapper in background, then ComfyUI in foreground
start_api_wrapper &

cd "${COMFYUI_DIR}" && exec python3 main.py --listen "${COMFYUI_HOST}" --port "${COMFYUI_PORT}"