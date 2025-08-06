#date: 2025-08-06T16:56:22Z
#url: https://api.github.com/gists/20fdcbb064b1234748cf62161625c7e1
#owner: https://api.github.com/users/chethanuk

#!/bin/bash

# Comprehensive Zsh Setup Script for Fresh Ubuntu/Linux VM
# This script installs and configures zsh, oh-my-zsh, mise, and development tools
# Created for Staff Engineer at Meta - Optimized for Python 3.12+ development + Go

set -euo pipefail

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Logging functions
log_info() {
    echo -e "${BLUE}[INFO]${NC} $1"
}

log_success() {
    echo -e "${GREEN}[SUCCESS]${NC} $1"
}

log_warning() {
    echo -e "${YELLOW}[WARNING]${NC} $1"
}

log_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

# Check if running as root
if [[ $EUID -eq 0 ]]; then
    log_error "This script should not be run as root. Please run as your regular user."
    exit 1
fi

log_info "Starting comprehensive zsh setup for development environment..."

# Step 1: Update system and install dependencies
log_info "Updating system packages and installing dependencies..."
# sudo apt update && sudo apt upgrade -y
# sudo apt install -y zsh git curl wget build-essential fonts-powerline dconf-cli ca-certificates

# Step 2: Verify zsh installation
log_info "Verifying zsh installation..."
if ! command -v zsh &> /dev/null; then
    log_error "Failed to install zsh"
    exit 1
fi

ZSH_VERSION=$(zsh --version)
log_success "Zsh installed successfully: $ZSH_VERSION"

# Step 3: Install Oh My Zsh
log_info "Installing Oh My Zsh framework..."
if [[ -d "$HOME/.oh-my-zsh" ]]; then
    log_warning "Oh My Zsh already exists, backing up..."
    mv "$HOME/.oh-my-zsh" "$HOME/.oh-my-zsh.backup.$(date +%Y%m%d-%H%M%S)"
fi

# Install Oh My Zsh without changing shell yet
export RUNZSH=no
export CHSH=no
sh -c "$(curl -fsSL https://raw.githubusercontent.com/ohmyzsh/ohmyzsh/master/tools/install.sh)" "" --unattended
log_success "Oh My Zsh installed successfully"

# Step 4: Install mise (development tool version manager)
log_info "Installing mise for managing Python, Node.js, Go, and other development tools..."
curl https://mise.run | sh

# Add mise to PATH for current session
export PATH="$HOME/.local/bin:$PATH"

# Verify mise installation
if [[ -f "$HOME/.local/bin/mise" ]]; then
    log_success "Mise installed successfully at $HOME/.local/bin/mise"
else
    log_error "Failed to install mise"
    exit 1
fi

# Step 5: Temporarily activate mise for bash (current session)
log_info "Temporarily activating mise for current bash session..."
eval "$($HOME/.local/bin/mise activate bash)"

# Step 6: Create .zprofile for login shell
log_info "Creating .zprofile for login shell configuration..."
cat > "$HOME/.zprofile" << 'EOF'
# .zprofile - executed for login shells

# Add user's private bin to PATH
if [ -d "$HOME/.local/bin" ] ; then
    PATH="$HOME/.local/bin:$PATH"
fi

# Add user's scripts to PATH
if [ -d "$HOME/scripts" ] ; then
    PATH="$HOME/scripts:$PATH"
fi

# Set default editor
export EDITOR='nano'

# Python development settings
export PYTHONDONTWRITEBYTECODE=1
export PYTHONUNBUFFERED=1

# Node.js settings
export NODE_OPTIONS="--max-old-space-size=4096"

# Go settings
export GOPATH="$HOME/go"
export PATH="$GOPATH/bin:$PATH"
EOF

# Step 7: Configure zsh with mise activation
log_info "Configuring zsh with mise activation..."
if [[ -f "$HOME/.zshrc" ]]; then
    cp "$HOME/.zshrc" "$HOME/.zshrc.backup.$(date +%Y%m%d-%H%M%S)"
fi

cat > "$HOME/.zshrc" << 'EOF'
# Path to your oh-my-zsh installation.
export ZSH="$HOME/.oh-my-zsh"

# Set name of the theme to load
ZSH_THEME="robbyrussell"

# Plugins
plugins=(
    git
    colored-man-pages
    command-not-found
    history-substring-search
    zsh-autosuggestions
    zsh-syntax-highlighting
)

source $ZSH/oh-my-zsh.sh

# Mise activation for zsh
eval "$(~/.local/bin/mise activate zsh)"

# User configuration
export EDITOR='nano'
export ARCHFLAGS="-arch $(uname -m)"

# Custom aliases
alias ll='ls -alF'
alias la='ls -A'
alias l='ls -CF'
alias grep='grep --color=auto'
alias fgrep='fgrep --color=auto'
alias egrep='egrep --color=auto'

# Development aliases
alias py='python'
alias py3='python3'
alias pip3='python3 -m pip'
alias venv='python3 -m venv'
alias serve='python3 -m http.server'

# Mise shortcuts
alias mi='mise install'
alias mu='mise use'
alias ml='mise list'
alias mls='mise ls'

# Git aliases
alias gs='git status'
alias ga='git add'
alias gc='git commit'
alias gp='git push'
alias gl='git log --oneline'
alias gb='git branch'
alias gco='git checkout'

# History configuration
HISTSIZE=10000
SAVEHIST=10000
setopt HIST_VERIFY
setopt SHARE_HISTORY
setopt APPEND_HISTORY
setopt INC_APPEND_HISTORY
setopt HIST_IGNORE_DUPS
setopt HIST_IGNORE_ALL_DUPS
setopt HIST_REDUCE_BLANKS
setopt HIST_IGNORE_SPACE

# Case insensitive globbing
setopt NO_CASE_GLOB

# Auto correction
setopt CORRECT
setopt CORRECT_ALL

# Completion settings
autoload -Uz compinit
compinit

# Better completion
zstyle ':completion:*' matcher-list 'm:{a-zA-Z}={A-Za-z}'
zstyle ':completion:*' list-colors "${(s.:.)LS_COLORS}"
zstyle ':completion:*' rehash true
zstyle ':completion:*' accept-exact '*(N)'
zstyle ':completion:*' use-cache on
zstyle ':completion:*' cache-path ~/.zsh/cache

# Load any local customizations
[[ -f ~/.zshrc.local ]] && source ~/.zshrc.local
EOF

log_success "Zsh configuration created"

# Step 8: Install zsh plugins
log_info "Installing useful zsh plugins..."
ZSH_CUSTOM="${ZSH_CUSTOM:-$HOME/.oh-my-zsh/custom}"

if [[ ! -d "$ZSH_CUSTOM/plugins/zsh-autosuggestions" ]]; then
    git clone https://github.com/zsh-users/zsh-autosuggestions.git "$ZSH_CUSTOM/plugins/zsh-autosuggestions"
    log_success "zsh-autosuggestions installed"
fi

if [[ ! -d "$ZSH_CUSTOM/plugins/zsh-syntax-highlighting" ]]; then
    git clone https://github.com/zsh-users/zsh-syntax-highlighting.git "$ZSH_CUSTOM/plugins/zsh-syntax-highlighting"
    log_success "zsh-syntax-highlighting installed"
fi

# Step 9: Create development directory structure
log_info "Creating development directory structure..."
mkdir -p "$HOME/dev" "$HOME/projects" "$HOME/scripts" "$HOME/.local/bin"

# Step 10: Install Python versions using mise
log_info "Installing Python 3.13 and 3.12 using mise..."
mise install python@3.13
mise install python@3.12
mise use --global python@3.13
log_success "Python 3.13 and 3.12 installed via mise"

# Step 11: Install Node.js and npm using mise
log_info "Installing latest Node.js and npm using mise..."
mise install node@latest
mise use --global node@latest
log_success "Latest Node.js and npm installed via mise"

# Step 12: Install Go lang using mise
log_info "Installing latest Go language using mise..."
mise install go@latest
mise use --global go@latest
log_success "Latest Go installed via mise"

# Step 13: Install essential Python packages
log_info "Installing essential Python development packages..."
mise exec python@3.13 -- python -m pip install --upgrade pip
mise exec python@3.13 -- python -m pip install black flake8 mypy pytest ipython jupyter requests
log_success "Essential Python packages installed"

# Step 14: Install useful npm packages globally
log_info "Installing useful npm packages globally..."
mise exec node@latest -- npm install -g yarn pnpm @angular/cli typescript ts-node nodemon
log_success "Global npm packages installed"


# Install uvx
curl -LsSf https://astral.sh/uv/install.sh | sh

# Install pipx
sudo apt update
sudo apt install pipx -y
pipx ensurepath
sudo pipx ensurepath --global

# Step 15: Create helper scripts
log_info "Creating helper scripts..."

# mkvenv
cat > "$HOME/scripts/mkvenv" << 'EOF'
#!/bin/bash
# Quick virtual environment creator

VENV_NAME=${1:-venv}
PYTHON_VERSION=${2:-3.13}

echo "Creating virtual environment '$VENV_NAME' with Python $PYTHON_VERSION..."
mise exec python@$PYTHON_VERSION -- python -m venv "$VENV_NAME"
echo "Virtual environment created. Activate with: source $VENV_NAME/bin/activate"
EOF
chmod +x "$HOME/scripts/mkvenv"

# verify-setup
cat > "$HOME/scripts/verify-setup" << 'EOF'
#!/bin/bash
# Verification script for development environment

echo "=== Development Environment Verification ==="
echo

echo "Shell: $SHELL"
echo "Zsh version: $(zsh --version)"
echo "Oh My Zsh: $([ -d ~/.oh-my-zsh ] && echo "Installed" || echo "Not found")"
echo "Mise: $(~/.local/bin/mise --version)"
echo

echo "=== Running mise doctor in zsh context ==="
zsh -c "source ~/.zshrc && mise doctor"
echo

echo "=== Available Tools ==="
~/.local/bin/mise list 2>/dev/null || echo "No tools found"
echo

echo "=== Python Information ==="
~/.local/bin/mise exec python@3.13 -- python --version
~/.local/bin/mise exec python@3.12 -- python --version
echo

echo "=== Node.js Information ==="
~/.local/bin/mise exec node@latest -- node --version
~/.local/bin/mise exec node@latest -- npm --version
echo

echo "=== Go Information ==="
~/.local/bin/mise exec go@latest -- go version
echo

echo "=== Git Configuration ==="
echo "Name: $(git config --global user.name || echo 'Not set')"
echo "Email: $(git config --global user.email || echo 'Not set')"

echo
echo "=== Quick Test Commands ==="
echo "Test Python: mise exec python@3.13 -- python --version"
echo "Test Node: mise exec node@latest -- node --version"
echo "Test Go: mise exec go@latest -- go version"
echo "List tools: mise ls"
EOF
chmod +x "$HOME/scripts/verify-setup"

# switch-to-zsh helper
cat > "$HOME/scripts/switch-to-zsh" << 'EOF'
#!/bin/bash
# Helper script to switch to zsh and verify setup

echo "Switching to zsh and loading configuration..."
sudo chsh -s "$(which zsh)" "$USER"
echo "Default shell changed to zsh (will take effect after logout/login)"
echo
echo "Starting new zsh session to test configuration..."
exec zsh -l
EOF
chmod +x "$HOME/scripts/switch-to-zsh"

# Step 16: Set zsh as default shell
log_info "Setting zsh as default shell..."
if [[ "$SHELL" != "$(which zsh)" ]]; then
    sudo chsh -s "$(which zsh)" "$USER"
    log_success "Default shell changed to zsh (will take effect after logout/login)"
else
    log_success "Zsh is already the default shell"
fi

# Step 17: Test mise in zsh context
log_info "Testing mise configuration in zsh context..."
zsh -c "source ~/.zshrc && mise doctor" || log_warning "Mise activation will work after switching to zsh"

# Step 18: Final verification
log_info "Running final verification..."
mise ls

log_success "=== Setup Complete! ==="
echo
log_info "Next steps:"
echo "1. **IMPORTANT**: Logout and login (or restart terminal) to activate zsh as default shell"
echo "2. Run: ~/scripts/verify-setup"
echo "3. Or run: ~/scripts/switch-to-zsh (to test immediately)"
echo "4. Configure Git: git config --global user.name 'Your Name'"
echo "5. Configure Git: git config --global user.email 'your.email@meta.com'"
echo "6. Test tools: mise ls"
echo
log_warning "Note: mise doctor will show 'not activated' until you switch to zsh shell"
log_success "Your development environment is ready!"
