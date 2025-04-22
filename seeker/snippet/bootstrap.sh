#date: 2025-04-22T17:08:21Z
#url: https://api.github.com/gists/6c681c0aa2dd302b63cdaa6f93130116
#owner: https://api.github.com/users/JohnVeni

#!/bin/bash

set -e

echo "🚀 Starting Mac DevOps Bootstrap..."

# ----------- Xcode CLI Tools -----------
echo "🧰 Installing Xcode CLI tools..."
xcode-select --install 2>/dev/null || echo "✅ Xcode CLI already installed"

# ----------- Homebrew Setup -----------
if ! command -v brew &>/dev/null; then
  echo "🍺 Installing Homebrew..."
  /bin/bash -c "$(curl -fsSL https://raw.githubusercontent.com/Homebrew/install/HEAD/install.sh)"
else
  echo "✅ Homebrew already installed"
fi

echo "🔄 Updating Homebrew..."
brew update

# ----------- Core Tools -----------
echo "📦 Installing essential tools..."
brew install \
  curl \
  git \
  screen \
  wget \
  awscli \
  kubectl \
  minikube \
  helm \
  terraform \
  direnv \
  gnupg \
  docker \
  kubectx \
  kubens \
  #bat \
  #exa \
  vim \
  zsh

# ----------- Oh My Zsh -----------
if [ ! -d "$HOME/.oh-my-zsh" ]; then
  echo "⚡ Installing Oh My Zsh..."
  sh -c "$(curl -fsSL https://raw.githubusercontent.com/ohmyzsh/ohmyzsh/master/tools/install.sh)"
fi

# ----------- SSH Key -----------
SSH_KEY="$HOME/.ssh/id_ed25519"
if [ ! -f "$SSH_KEY" ]; then
  echo "🔐 Generating SSH key (ed25519)..."
  ssh-keygen -t ed25519 -C "$USER@$(hostname)" -f "$SSH_KEY" -N ""
else
  echo "✅ SSH key already exists at $SSH_KEY"
fi

# ----------- Folder Structure -----------
echo "📂 Creating DevOps folders..."
mkdir -p ~/DevOps/{lab,bin,dotfiles,infra,notes}

# ----------- Aliases -----------
ALIASES_FILE=~/.zsh_aliases
cat <<'EOF' > $ALIASES_FILE
# Kubernetes
alias k='kubectl'
alias kn='kubens'
alias kx='kubectx'

# Terraform
alias tf='terraform'
alias tfi='terraform init'
alias tfp='terraform plan'
alias tfa='terraform apply'
alias tfd='terraform destroy -auto-approve'

# Docker
alias d='docker'
alias dc='docker compose'

# Git
alias gs='git status'
alias gl='git pull'
alias gp='git push'
alias gc='git commit'
alias ga='git add .'

# Shortcuts
alias devops="cd ~/DevOps"
alias lab="cd ~/DevOps/lab"
#alias dotfiles="cd ~/DevOps/dotfiles"

# Better tools
#alias cat='bat'
#alias ls='exa -la --color=always --group-directories-first'
EOF

# ----------- Hook Aliases into Zsh -----------
if ! grep -q "source ~/.zsh_aliases" ~/.zshrc; then
  echo "📎 Adding aliases to .zshrc..."
  echo "source ~/.zsh_aliases" >> ~/.zshrc
fi