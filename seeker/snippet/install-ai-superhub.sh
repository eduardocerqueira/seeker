#date: 2026-03-18T17:41:47Z
#url: https://api.github.com/gists/cb4fcabf23c76e8816a0f730c03dd50f
#owner: https://api.github.com/users/getgoingbb

#!/bin/bash
# AI SuperHub — Fresh Ubuntu Install Script
# Installs everything on a new Ubuntu 22.04+ / 24.04 machine (bare metal or WSL2)
# Usage: bash install-ai-superhub.sh
# GitHub: https://github.com/getgoingbb/ai-superhub-main

set -e

GREEN='\033[0;32m'
CYAN='\033[0;36m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
NC='\033[0m'

GITHUB_USER="getgoingbb"
SUPERHUB_DIR="$HOME/AI-SuperHub"
MOA_VENV="$HOME/.moa-venv"

banner() {
    echo ""
    echo -e "${CYAN}╔══════════════════════════════════════════════════╗${NC}"
    echo -e "${CYAN}║   AI SuperHub — Ubuntu Install Script            ║${NC}"
    echo -e "${CYAN}║   github.com/getgoingbb/ai-superhub-main         ║${NC}"
    echo -e "${CYAN}╚══════════════════════════════════════════════════╝${NC}"
    echo ""
}

step() { echo -e "\n${YELLOW}[$1] $2${NC}"; }
ok()   { echo -e "  ${GREEN}✓ $1${NC}"; }
warn() { echo -e "  ${YELLOW}⚠ $1${NC}"; }
fail() { echo -e "  ${RED}✗ $1${NC}"; }

# ─── detect environment ──────────────────────────────────────────────────────
is_wsl() { grep -qi microsoft /proc/version 2>/dev/null; }
HAS_GPU=false
if command -v nvidia-smi &>/dev/null && nvidia-smi &>/dev/null; then
    HAS_GPU=true
fi

banner

echo -e "  User:      ${CYAN}$(whoami)${NC}"
echo -e "  Home:      ${CYAN}$HOME${NC}"
echo -e "  OS:        ${CYAN}$(lsb_release -ds 2>/dev/null || uname -a)${NC}"
is_wsl && echo -e "  Env:       ${CYAN}WSL2${NC}" || echo -e "  Env:       ${CYAN}Native Linux${NC}"
$HAS_GPU && echo -e "  GPU:       ${CYAN}$(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null | head -1)${NC}" || echo -e "  GPU:       ${YELLOW}None detected (CPU mode)${NC}"

echo ""
echo -e "  This will install:"
echo -e "  • System packages (curl, git, jq, tmux, ffmpeg, docker)"
echo -e "  • Ollama (local LLM runner)"
echo -e "  • Python 3 + venv with all AI packages"
echo -e "  • AI SuperHub repos from GitHub"
echo -e "  • Claude Code CLI (optional)"
echo ""
read -p "  Continue? [y/N] " -n 1 -r; echo
[[ ! $REPLY =~ ^[Yy]$ ]] && echo "Aborted." && exit 0

# ─── 1. system packages ──────────────────────────────────────────────────────
step "1/8" "System packages"
sudo apt-get update -qq
sudo apt-get install -y -qq \
    curl wget git jq tmux htop \
    python3 python3-venv python3-pip python3-dev \
    ffmpeg build-essential \
    ca-certificates gnupg lsb-release \
    apt-transport-https software-properties-common \
    nginx
ok "System packages installed"

# Node.js 22 via nvm
step "1b/8" "Node.js 22 + uv"
if ! command -v nvm &>/dev/null && [ ! -f "$HOME/.nvm/nvm.sh" ]; then
    curl -o- https://raw.githubusercontent.com/nvm-sh/nvm/v0.40.2/install.sh | bash
fi
export NVM_DIR="$HOME/.nvm"
[ -s "$NVM_DIR/nvm.sh" ] && source "$NVM_DIR/nvm.sh"
nvm install 22 --default &>/dev/null && ok "Node.js 22 ready"

# uv (Python package manager for DeerFlow)
if ! command -v uv &>/dev/null; then
    curl -LsSf https://astral.sh/uv/install.sh | sh
    export PATH="$HOME/.local/bin:$PATH"
fi
ok "uv $(uv --version 2>/dev/null | awk '{print $2}') ready"

# pnpm (for DeerFlow frontend)
source "$NVM_DIR/nvm.sh" 2>/dev/null
if ! command -v pnpm &>/dev/null; then
    npm install -g pnpm -q
fi
ok "pnpm $(pnpm -v 2>/dev/null) ready"

# ─── 1c. NVIDIA drivers + CUDA (bare metal Linux with NVIDIA GPU) ────────────
if ! is_wsl && $HAS_GPU; then
    step "1c/8" "NVIDIA CUDA + container toolkit"

    # Install CUDA keyring if not present
    if ! dpkg -l | grep -q cuda-keyring 2>/dev/null; then
        echo "  Adding NVIDIA CUDA repo..."
        CUDA_KEYRING_URL="https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2404/x86_64/cuda-keyring_1.1-1_all.deb"
        curl -fsSL "$CUDA_KEYRING_URL" -o /tmp/cuda-keyring.deb 2>/dev/null \
            || { warn "CUDA keyring download failed — skipping GPU setup"; }
        [ -f /tmp/cuda-keyring.deb ] && sudo dpkg -i /tmp/cuda-keyring.deb > /dev/null 2>&1
        sudo apt-get update -qq
    fi

    # Install nvidia-container-toolkit so Docker can use the GPU
    if ! command -v nvidia-ctk &>/dev/null; then
        echo "  Installing nvidia-container-toolkit..."
        sudo apt-get install -y -qq nvidia-container-toolkit 2>/dev/null \
            && ok "nvidia-container-toolkit installed" \
            || warn "nvidia-container-toolkit install failed — GPU won't be available in Docker"
    else
        ok "nvidia-container-toolkit already installed"
    fi

    # Configure Docker runtime for NVIDIA
    if command -v nvidia-ctk &>/dev/null; then
        sudo nvidia-ctk runtime configure --runtime=docker > /dev/null 2>&1
        ok "Docker NVIDIA runtime configured"
    fi
elif is_wsl && $HAS_GPU; then
    ok "WSL2 GPU passthrough active (CUDA provided by Windows driver)"
fi

# ─── 2. Docker ───────────────────────────────────────────────────────────────
step "2/8" "Docker"
if command -v docker &>/dev/null; then
    ok "Docker already installed: $(docker --version)"
else
    echo "  Installing Docker..."
    curl -fsSL https://download.docker.com/linux/ubuntu/gpg | sudo gpg --dearmor -o /usr/share/keyrings/docker-archive-keyring.gpg
    echo "deb [arch=$(dpkg --print-architecture) signed-by=/usr/share/keyrings/docker-archive-keyring.gpg] \
        https://download.docker.com/linux/ubuntu $(lsb_release -cs) stable" \
        | sudo tee /etc/apt/sources.list.d/docker.list > /dev/null
    sudo apt-get update -qq
    sudo apt-get install -y -qq docker-ce docker-ce-cli containerd.io docker-compose-plugin
    ok "Docker installed"
fi

# Add user to docker group
if groups | grep -q docker; then
    ok "User already in docker group"
else
    sudo usermod -aG docker "$USER"
    warn "Added to docker group — run: newgrp docker  (or log out and back in)"
fi

# Start Docker daemon if not running
if ! docker info &>/dev/null 2>&1; then
    sudo service docker start 2>/dev/null || sudo dockerd &>/dev/null &
    sleep 3
fi
ok "Docker running"

# ─── 3. Ollama ───────────────────────────────────────────────────────────────
step "3/8" "Ollama"
if command -v ollama &>/dev/null; then
    ok "Ollama already installed: $(ollama --version 2>/dev/null)"
else
    echo "  Installing Ollama..."
    curl -fsSL https://ollama.com/install.sh | sh
    ok "Ollama installed"
fi

# Configure Ollama to listen on 0.0.0.0 (needed for Docker containers)
OLLAMA_ENV_FILE="/etc/systemd/system/ollama.service.d/override.conf"
if ! grep -q "OLLAMA_HOST" "$OLLAMA_ENV_FILE" 2>/dev/null; then
    sudo mkdir -p "$(dirname $OLLAMA_ENV_FILE)"
    sudo tee "$OLLAMA_ENV_FILE" > /dev/null <<'EOF'
[Service]
Environment="OLLAMA_HOST=0.0.0.0:11434"
EOF
    ok "Ollama configured to listen on 0.0.0.0:11434"
else
    ok "Ollama already configured"
fi

# Start Ollama
if ! pgrep -x ollama &>/dev/null; then
    OLLAMA_HOST=0.0.0.0 nohup ollama serve > /tmp/ollama.log 2>&1 &
    sleep 2
fi
ok "Ollama running"

# Pull starter model
echo "  Pulling qwen2.5:7b (default model — ~4.7GB)..."
ollama pull qwen2.5:7b && ok "qwen2.5:7b ready" || warn "Model pull failed — run manually: ollama pull qwen2.5:7b"

# ─── 4. Python venv + packages ───────────────────────────────────────────────
step "4/8" "Python venv + AI packages"
if [ ! -d "$MOA_VENV" ]; then
    python3 -m venv "$MOA_VENV"
    ok "Venv created at $MOA_VENV"
else
    ok "Venv already exists"
fi

"$MOA_VENV/bin/pip" install --upgrade pip -q

echo "  Installing packages..."
"$MOA_VENV/bin/pip" install -q \
    ollama typer \
    fastapi uvicorn python-multipart \
    flask flask-cors \
    duckduckgo-search \
    requests \
    PyMuPDF \
    faster-whisper \
    gradio \
    openai \
    litellm \
    torch torchvision torchaudio \
    --extra-index-url https://download.pytorch.org/whl/cpu

ok "Python packages installed"

# ─── 5. Clone repos ──────────────────────────────────────────────────────────
step "5/8" "AI SuperHub repos"
mkdir -p "$SUPERHUB_DIR"

repos=(
    "ai-superhub-main"
    "moa-core"
    "deer-flow"
    "voice-integration"
    "infrastructure"
    "privategpt"
)

for repo in "${repos[@]}"; do
    target="$SUPERHUB_DIR/$repo"
    if [ -d "$target/.git" ]; then
        ok "$repo already cloned — pulling latest"
        git -C "$target" pull -q 2>/dev/null || warn "$repo pull failed (maybe local changes)"
    else
        echo "  Cloning $repo..."
        git clone -q "https://github.com/$GITHUB_USER/$repo.git" "$target" && ok "$repo cloned" || warn "$repo clone failed"
    fi
done

# ─── 5b. DeerFlow dependencies ───────────────────────────────────────────────
step "5b/8" "DeerFlow super-agent"
DEERFLOW_DIR="$SUPERHUB_DIR/deer-flow"
if [ -d "$DEERFLOW_DIR/backend" ]; then
    export PATH="$HOME/.local/bin:$PATH"
    export NVM_DIR="$HOME/.nvm"
    [ -s "$NVM_DIR/nvm.sh" ] && source "$NVM_DIR/nvm.sh"

    # Install langchain-ollama in DeerFlow venv
    cd "$DEERFLOW_DIR/backend"
    uv sync -q 2>/dev/null && uv add langchain-ollama -q 2>/dev/null || true
    ok "DeerFlow backend deps installed"

    # Install frontend deps
    cd "$DEERFLOW_DIR/frontend"
    pnpm install -q 2>/dev/null && ok "DeerFlow frontend deps installed" || warn "DeerFlow frontend install failed"

    # Create .env if missing
    [ -f "$DEERFLOW_DIR/.env" ] || cp "$DEERFLOW_DIR/.env.example" "$DEERFLOW_DIR/.env" 2>/dev/null || echo "" > "$DEERFLOW_DIR/.env"
    [ -f "$DEERFLOW_DIR/frontend/.env" ] || cp "$DEERFLOW_DIR/frontend/.env.example" "$DEERFLOW_DIR/frontend/.env" 2>/dev/null || echo "" > "$DEERFLOW_DIR/frontend/.env"
    ok "DeerFlow ready — start: bash $SUPERHUB_DIR/ai-superhub-main/deer-flow-start.sh"
else
    warn "DeerFlow not found at $DEERFLOW_DIR — clone it manually or add to repos list"
fi
cd "$SUPERHUB_DIR/ai-superhub-main" 2>/dev/null || true

# ─── 6. Open WebUI (Docker) ──────────────────────────────────────────────────
step "6/8" "Open WebUI"
WSL_IP=$(hostname -I 2>/dev/null | awk '{print $1}')

if docker ps -a --format '{{.Names}}' | grep -q '^open-webui$'; then
    if docker ps --format '{{.Names}}' | grep -q '^open-webui$'; then
        ok "Open WebUI already running"
    else
        docker start open-webui && ok "Open WebUI started"
    fi
else
    echo "  Starting Open WebUI container..."
    # On bare metal with GPU, use --gpus all + connect to local Ollama
    if ! is_wsl && $HAS_GPU; then
        GPU_FLAGS="--gpus all"
        OLLAMA_URL="http://host-gateway:11434"
    elif is_wsl; then
        GPU_FLAGS=""
        OLLAMA_URL="http://${WSL_IP}:11434"
    else
        GPU_FLAGS=""
        OLLAMA_URL="http://host-gateway:11434"
    fi
    sg docker -c "docker run -d \
        --name open-webui \
        --restart unless-stopped \
        --add-host=host-gateway:host-gateway \
        $GPU_FLAGS \
        -p 8080:8080 \
        -v open-webui:/app/backend/data \
        -e OLLAMA_BASE_URL=${OLLAMA_URL} \
        ghcr.io/open-webui/open-webui:main" > /dev/null 2>&1 \
        && ok "Open WebUI started at http://localhost:8080" \
        || warn "Open WebUI start failed — try: docker run ... ghcr.io/open-webui/open-webui:main"
fi

# ─── 7. n8n (Docker) ─────────────────────────────────────────────────────────
step "7/8" "n8n workflow automation"
if docker ps -a --format '{{.Names}}' | grep -q '^n8n$'; then
    docker ps --format '{{.Names}}' | grep -q '^n8n$' && ok "n8n already running" || docker start n8n && ok "n8n started"
else
    docker run -d --name n8n --restart unless-stopped \
        -p 5678:5678 \
        -v n8n_data:/home/node/.n8n \
        n8nio/n8n:latest > /dev/null 2>&1 \
        && ok "n8n started at http://localhost:5678" \
        || warn "n8n start failed"
fi

# Import Whisper workflow into n8n
echo "  Importing Whisper→AI workflow into n8n..."
sleep 3
WHISPER_WF="$SUPERHUB_DIR/ai-superhub-main/n8n/whisper-workflow.json"
if [ -f "$WHISPER_WF" ]; then
    docker cp "$WHISPER_WF" n8n:/tmp/whisper-workflow.json 2>/dev/null && \
    docker exec n8n n8n import:workflow --input=/tmp/whisper-workflow.json 2>/dev/null && \
    ok "Whisper workflow imported into n8n" || warn "n8n workflow import failed — import manually at http://localhost:5678"
fi

# ─── 7b. SearXNG (local private metasearch — 70+ sources) ───────────────────
step "7b/8" "SearXNG private search engine"
SEARXNG_SETTINGS="/etc/searxng/settings.yml"
if docker ps -a --format '{{.Names}}' | grep -q '^searxng$'; then
    docker ps --format '{{.Names}}' | grep -q '^searxng$' && ok "SearXNG already running" || docker start searxng && ok "SearXNG started"
else
    echo "  Starting SearXNG container..."
    # Create settings dir and base config
    sudo mkdir -p /etc/searxng
    # Generate secret key
    SEARXNG_SECRET= "**********"
    sudo tee /etc/searxng/settings.yml > /dev/null <<EOF
use_default_settings: true
server:
  secret_key: "**********"
  bind_address: "0.0.0.0:8090"
  image_proxy: false
search:
  safe_search: 0
  default_lang: "en"
ui:
  default_theme: simple
  query_in_title: true
formats:
  - html
  - json
EOF
    docker run -d \
        --name searxng \
        --restart unless-stopped \
        -p 8090:8090 \
        -v /etc/searxng:/etc/searxng \
        searxng/searxng:latest > /dev/null 2>&1 \
        && ok "SearXNG started at http://localhost:8090" \
        || warn "SearXNG start failed"
fi

# ─── 8. Shell config + aliases ───────────────────────────────────────────────
step "8/8" "Shell aliases & config"

ALIAS_BLOCK='
# ── AI SuperHub ──────────────────────────────────────────────────────────────
export MOA_VENV="$HOME/.moa-venv"
alias superhub="sg docker -c \"bash $HOME/AI-SuperHub/ai-superhub-main/ai-superhub-start.sh\""
alias moa="cd $HOME/AI-SuperHub/ai-superhub-main/moa-core && $HOME/.moa-venv/bin/python moa-web.py"
alias portal="cd $HOME/AI-SuperHub/ai-superhub-main/portal && $HOME/.moa-venv/bin/python portal-start.py"
alias whisper="$HOME/.moa-venv/bin/python $HOME/AI-SuperHub/ai-superhub-main/voice/whisper-server.py"
alias ollama-restart="OLLAMA_HOST=0.0.0.0 nohup ollama serve > /tmp/ollama.log 2>&1 &"
alias deerflow="bash $HOME/AI-SuperHub/ai-superhub-main/deer-flow-start.sh"
alias logs="cat $HOME/AI-SuperHub/chat-logs/last-session.txt | tail -60"
# ─────────────────────────────────────────────────────────────────────────────
'

SHELL_RC="$HOME/.bashrc"
[ -n "$ZSH_VERSION" ] && SHELL_RC="$HOME/.zshrc"

if grep -q "AI SuperHub" "$SHELL_RC" 2>/dev/null; then
    ok "Aliases already in $SHELL_RC"
else
    echo "$ALIAS_BLOCK" >> "$SHELL_RC"
    ok "Aliases added to $SHELL_RC"
fi

# ─── Optional: Claude Code ───────────────────────────────────────────────────
echo ""
read -p "  Install Claude Code CLI? [y/N] " -n 1 -r; echo
if [[ $REPLY =~ ^[Yy]$ ]]; then
    if command -v claude &>/dev/null; then
        ok "Claude Code already installed"
    else
        npm install -g @anthropic-ai/claude-code 2>/dev/null \
            && ok "Claude Code installed" \
            || warn "npm not found — install Node.js first, then: npm install -g @anthropic-ai/claude-code"
    fi
fi

# ─── Summary ─────────────────────────────────────────────────────────────────
echo ""
echo -e "${CYAN}════════════════════════════════════════════════════${NC}"
echo -e "${GREEN}  Install Complete!${NC}"
echo -e "${CYAN}════════════════════════════════════════════════════${NC}"
echo ""
echo -e "  ${CYAN}Start everything:${NC}"
echo -e "    sg docker -c \"bash ~/AI-SuperHub/ai-superhub-main/ai-superhub-start.sh\""
echo ""
echo -e "  ${CYAN}Services after start:${NC}"
echo -e "    Open WebUI:    http://localhost:8080"
echo -e "    Portal:        http://localhost:7777"
echo -e "    MOA Web UI:    http://localhost:8888"
echo -e "    Whisper:       http://localhost:8765"
echo -e "    n8n:           http://localhost:5678"
echo -e "    Skill Browser: http://localhost:7860"
echo -e "    Video Studio:  http://localhost:7862"
echo -e "    DeerFlow:      http://localhost:3000  (run: deerflow)"
echo -e "    SearXNG:       http://localhost:8090"
echo ""
echo -e "  ${YELLOW}Next steps:${NC}"
echo -e "    1. source ~/.bashrc        # load aliases"
echo -e "    2. newgrp docker           # activate docker group (or log out/in)"
echo -e "    3. superhub                # start all services"
echo ""
if ! $HAS_GPU; then
    echo -e "  ${YELLOW}Note: No GPU detected. Models will run on CPU (slower).${NC}"
    echo -e "  ${YELLOW}For NVIDIA GPU support, install CUDA + nvidia-container-toolkit.${NC}"
    echo ""
fi
if ! is_wsl; then
    echo -e "  ${CYAN}Bare metal tips:${NC}"
    echo -e "  • If booting from USB: use USB 3.x port for best I/O speed"
    echo -e "  • Ollama runs as systemd service — check: systemctl status ollama"
    echo -e "  • GPU acceleration active: ollama ps will show GPU usage"
    echo ""
fi
echo -e "${CYAN}════════════════════════════════════════════════════${NC}"
echo ""
��${NC}"
echo ""
