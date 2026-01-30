#date: 2026-01-30T17:18:23Z
#url: https://api.github.com/gists/d9df8c7e60201b5a7a5b4ab0bbe796d8
#owner: https://api.github.com/users/vinter-man

#!/bin/bash
set -e # Stop execution if any command fails

# -----------------------------------------------------------
# 💀 ULTIMATE ROOT SERVER SETUP
# Docker, Node.js (NVM), Python (Pyenv), Playwright, LazyTools
# -----------------------------------------------------------

echo "🚀 --- LET'S GO! SERVER SETUP (ROOT MODE) ---"

# Disable interactive prompts for apt
export DEBIAN_FRONTEND=noninteractive
export HOME="/root"

echo "📦 [1/7] Updating system & installing base tools..."
apt-get update && apt-get upgrade -y
apt-get install -y curl wget git unzip build-essential libssl-dev zlib1g-dev \
libbz2-dev libreadline-dev libsqlite3-dev llvm libncursesw5-dev xz-utils \
tk-dev libxml2-dev libxmlsec1-dev libffi-dev liblzma-dev htop bat jq \
ca-certificates gnupg lsb-release

echo "🐳 [2/7] Installing Docker..."
if ! command -v docker &> /dev/null; then
    curl -fsSL https://get.docker.com | sh
    echo "✅ Docker installed"
else
    echo "⏩ Docker already exists"
fi

echo "😎 [3/7] Installing human-friendly tools (LazyDocker & LazyGit)..."
# LazyDocker
curl https://raw.githubusercontent.com/jesseduffield/lazydocker/master/scripts/install_update_linux.sh | bash

# LazyGit (auto-detect latest version)
LAZYGIT_VERSION=$(curl -s "https://api.github.com/repos/jesseduffield/lazygit/releases/latest" | grep -Po '"tag_name": "v\K[^"]*')
curl -Lo lazygit.tar.gz "https://github.com/jesseduffield/lazygit/releases/latest/download/lazygit_${LAZYGIT_VERSION}_Linux_x86_64.tar.gz"
tar xf lazygit.tar.gz lazygit
install lazygit /usr/local/bin
rm lazygit lazygit.tar.gz
echo "✅ Lazy tools installed"

echo "🟢 [4/7] Installing NVM & Node.js..."
# Install NVM
curl -o- https://raw.githubusercontent.com/nvm-sh/nvm/v0.39.7/install.sh | bash

# Load NVM right now for the current session
export NVM_DIR="$HOME/.nvm"
[ -s "$NVM_DIR/nvm.sh" ] && \. "$NVM_DIR/nvm.sh"

# Install LTS Node
nvm install --lts
nvm use --lts
# Global packages
npm install -g yarn pnpm pm2 typescript ts-node nodemon
echo "✅ Node.js & npm tools ready"

echo "🐍 [5/7] Installing Pyenv & Python 3.11..."
if [ ! -d "$HOME/.pyenv" ]; then
    curl https://pyenv.run | bash
    
    # Add config to .bashrc for future sessions
    echo '' >> ~/.bashrc
    echo '# Pyenv Config' >> ~/.bashrc
    echo 'export PYENV_ROOT="$HOME/.pyenv"' >> ~/.bashrc
    echo '[[ -d $PYENV_ROOT/bin ]] && export PATH="$PYENV_ROOT/bin:$PATH"' >> ~/.bashrc
    echo 'eval "$(pyenv init -)"' >> ~/.bashrc
    
    # Export for current session
    export PYENV_ROOT="$HOME/.pyenv"
    export PATH="$PYENV_ROOT/bin:$PATH"
    eval "$(pyenv init -)"
fi

# Install stable Python
pyenv install -s 3.11.8
pyenv global 3.11.8
# Upgrade pip
pip install --upgrade pip
echo "✅ Python 3.11 ready"

echo "🎭 [6/7] Rolling out Playwright..."
# Install the package globally (for CLI access)
npm install -g playwright

# Install system dependencies (browsers engines for Linux)
# --with-deps flag pulls all necessary Ubuntu libs immediately
npx playwright install --with-deps
echo "✅ Playwright ready"

echo "💅 [7/7] Final touches..."
# Install ZSH if missing (optional but nice)
if ! command -v zsh &> /dev/null; then
    apt-get install -y zsh
    chsh -s $(which zsh) root
fi

echo "=============================================="
echo "🎉 DONE, BRO! EVERYTHING INSTALLED."
echo "=============================================="
echo "Versions:"
echo "🐳 Docker: $(docker --version)"
echo "🟢 Node: $(node -v)"
echo "🐍 Python: $(python --version)"
echo "----------------------------------------------"
echo "⚠️  IMPORTANT:"
echo "1. Playwright under root requires the browser launch flag:"
echo "   args: ['--no-sandbox']"
echo "2. To load paths (NVM, Pyenv) in a new session, run:"
echo "   source ~/.bashrc"
echo "----------------------------------------------"