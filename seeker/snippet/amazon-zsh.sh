#date: 2025-10-06T16:37:36Z
#url: https://api.github.com/gists/23e7cf4d3599b3a5dabcc255f173c65b
#owner: https://api.github.com/users/bhanusanghi

#!/bin/bash
# Oh My Zsh Setup for Amazon Linux EC2 - Productivity Focused
# Run with: curl -fsSL <your-url> | bash

set -e

echo "🚀 Installing Zsh and dependencies..."
sudo yum install -y zsh git curl wget

echo "📦 Installing Oh My Zsh..."
sh -c "$(curl -fsSL https://raw.githubusercontent.com/ohmyzsh/ohmyzsh/master/tools/install.sh)" "" --unattended

echo "🔌 Installing essential plugins..."

# zsh-autosuggestions (Fish-like suggestions)
git clone https://github.com/zsh-users/zsh-autosuggestions ${ZSH_CUSTOM:-~/.oh-my-zsh/custom}/plugins/zsh-autosuggestions

# zsh-syntax-highlighting (colors for valid/invalid commands)
git clone https://github.com/zsh-users/zsh-syntax-highlighting.git ${ZSH_CUSTOM:-~/.oh-my-zsh/custom}/plugins/zsh-syntax-highlighting

# zsh-completions (additional completion definitions)
git clone https://github.com/zsh-users/zsh-completions ${ZSH_CUSTOM:-~/.oh-my-zsh/custom}/plugins/zsh-completions

# Powerlevel10k theme (fast and beautiful)
git clone --depth=1 https://github.com/romkatv/powerlevel10k.git ${ZSH_CUSTOM:-~/.oh-my-zsh/custom}/themes/powerlevel10k

echo "📝 Creating optimized .zshrc..."
cat > ~/.zshrc << 'EOF'
# ============================================
# OH MY ZSH CONFIGURATION
# ============================================
export ZSH="$HOME/.oh-my-zsh"

# Theme - Powerlevel10k (fast & beautiful)
ZSH_THEME="powerlevel10k/powerlevel10k"

# ============================================
# PLUGINS (Productivity Focused)
# ============================================
plugins=(
  # Core productivity
  git                          # Git aliases and functions
  zsh-autosuggestions         # Fish-like autosuggestions
  zsh-syntax-highlighting     # Syntax highlighting
  zsh-completions             # Additional completions
  
  # Directory navigation
  z                           # Jump to frequent directories
  dirhistory                  # Alt+Left/Right to navigate dir history
  sudo                        # ESC ESC to add sudo to command
  
  # Utilities
  extract                     # Smart extraction: extract <file>
  copypath                    # Copy current path: copypath
  copyfile                    # Copy file contents: copyfile <file>
  jsontools                   # JSON formatting and validation
  encode64                    # Base64 encode/decode
  urltools                    # URL encoding/decoding
  colored-man-pages          # Colorful man pages
  command-not-found          # Suggest package for missing commands
  
  # Development
  npm                        # NPM completions and aliases
  pip                        # Python pip completions
  python                     # Python aliases
  systemd                    # Systemd completions and aliases
  
  # Productivity
  web-search                 # google/ddg/bing from terminal
  history                    # History aliases (h, hs, hsi)
  aliases                    # Search aliases: als <keyword>
)

# ============================================
# OH MY ZSH SETTINGS
# ============================================
CASE_SENSITIVE="false"
HYPHEN_INSENSITIVE="true"
DISABLE_AUTO_UPDATE="false"
UPDATE_ZSH_DAYS=30
DISABLE_LS_COLORS="false"
HIST_STAMPS="dd-mm-yyyy"

# Uncomment for faster Oh My Zsh load (skip verification)
ZSH_DISABLE_COMPFIX="true"

source $ZSH/oh-my-zsh.sh

# ============================================
# HISTORY CONFIGURATION (Massive History)
# ============================================
HISTFILE=~/.zsh_history
HISTSIZE=100000
SAVEHIST=100000
setopt EXTENDED_HISTORY
setopt HIST_EXPIRE_DUPS_FIRST
setopt HIST_IGNORE_DUPS
setopt HIST_IGNORE_ALL_DUPS
setopt HIST_FIND_NO_DUPS
setopt HIST_IGNORE_SPACE
setopt HIST_SAVE_NO_DUPS
setopt SHARE_HISTORY
setopt INC_APPEND_HISTORY

# ============================================
# KEY BINDINGS (Productivity)
# ============================================


# ============================================
# ENVIRONMENT VARIABLES
# ============================================
export EDITOR='vim'
export VISUAL='vim'
export PAGER='less'
export LESS='-R'

# Colors for ls
export LS_COLORS='di=34:ln=35:so=32:pi=33:ex=31:bd=34;46:cd=34;43:su=30;41:sg=30;46:tw=30;42:ow=30;43'

# Process management
alias killport='kill -9 $(lsof -t -i:$1)'

# ============================================
# NETWORK UTILITIES
# ============================================
alias myip='curl -s ifconfig.me'
alias localip='hostname -I | awk "{print \$1}"'
alias ports-used='sudo lsof -i -P -n | grep LISTEN'

# ============================================
# PACKAGE MANAGEMENT (Amazon Linux)
# ============================================
alias update='sudo yum update -y'
alias upgrade='sudo yum upgrade -y'
alias install='sudo yum install -y'
alias remove='sudo yum remove -y'
alias search='yum search'
alias installed='yum list installed'

# ============================================
# PRODUCTIVITY FUNCTIONS
# ============================================

# Show disk usage sorted
usage() {
  du -h --max-depth=1 2>/dev/null | sort -hr | head -20
}

# Network info summary
netinfo() {
  echo "=== Network Information ==="
  echo "Private IP: $(hostname -I | awk '{print $1}')"
  echo "Public IP:  $(curl -s ifconfig.me)"
  echo "Gateway:    $(ip route | grep default | awk '{print $3}')"
  echo "DNS:        $(cat /etc/resolv.conf | grep nameserver | awk '{print $2}' | head -1)"
  if command -v ec2-ip &> /dev/null; then
    echo "EC2 Public: $(ec2-ip)"
    echo "EC2 Region: $(ec2-region)"
  fi
}

# EC2 info summary
ec2info() {
  echo "=== EC2 Instance Information ==="
  echo "Instance ID:   $(ec2-id)"
  echo "Instance Type: $(ec2-type)"
  echo "Region:        $(ec2-region)"
  echo "AZ:            $(ec2-az)"
  echo "AMI:           $(ec2-ami)"
  echo "Public IP:     $(ec2-ip)"
  echo "Private IP:    $(ec2-private-ip)"
  echo "Security Groups: $(ec2-sg)"
}

# Find large files
findlarge() {
  find "${1:-.}" -type f -size +"${2:-100M}" -exec ls -lh {} \; | awk '{ print $9 ": " $5 }'
}

# Show top 10 largest directories
dirsize() {
  du -h "${1:-.}" 2>/dev/null | sort -hr | head -20
}

# Grep history
histgrep() {
  history | grep "$1"
}

# Count files in directory
count() {
  echo "Files: $(find "${1:-.}" -type f | wc -l)"
  echo "Directories: $(find "${1:-.}" -type d | wc -l)"
}

# ============================================
# WELCOME MESSAGE
# ============================================
clear
echo "╔════════════════════════════════════════╗"
echo "║   🚀 Oh My Zsh Loaded Successfully!   ║"
echo "╚════════════════════════════════════════╝"
echo ""
echo "💡 Quick Tips:"
echo "  • Press Ctrl+Space to accept suggestions"
echo "  • Use ESC ESC to add sudo to any command"
echo "  • Type 'ec2info' for instance details"
echo "  • Type 'als <keyword>' to search aliases"
echo "  • Use 'z <partial-name>' to jump to directories"
echo ""
echo "🔧 Run 'p10k configure' to customize your prompt"
echo ""
EOF

echo "⚙️ Setting Zsh as default shell..."
sudo chsh -s $(which zsh) $(whoami)

echo ""
echo "✅ Installation Complete!"
echo ""
echo "🎨 Configure Powerlevel10k prompt:"
echo "   exec zsh"
echo "   p10k configure"
echo ""
echo "📚 Useful commands to try:"
echo "   ec2info       - Show EC2 instance details"
echo "   netinfo       - Network information"
echo "   z <dir>       - Jump to frequent directory"
echo "   als git       - Search git-related aliases"
echo ""
echo "🔄 Activate now: exec zsh"