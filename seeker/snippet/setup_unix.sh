#date: 2025-10-16T17:11:15Z
#url: https://api.github.com/gists/5f48ae8e1173f7b1db105d80c70c2542
#owner: https://api.github.com/users/Prana-vvb

#!/bin/bash
set -euo pipefail

echo "=== HackNight Git & SSH Setup Script ==="

read -p "Enter your GitHub username: " GIT_NAME
read -p "Enter your GitHub email: " GIT_EMAIL

configure() {
    git config --global user.name "$GIT_NAME"
    git config --global user.email "$GIT_EMAIL"
    echo "Git configured with name: $GIT_NAME and email: $GIT_EMAIL"

    SSH_KEY="$HOME/.ssh/id_ed25519"
    if [ -f "$SSH_KEY" ]; then
        echo "SSH key already exists at $SSH_KEY"
    else
        echo "Generating new SSH key..."
        ssh-keygen -t ed25519 -C "$GIT_EMAIL" -f "$SSH_KEY" -N ""
    fi

    eval "$(ssh-agent -s)"
    ssh-add "$SSH_KEY"

    echo ""
    echo "Copy your public key and add it to GitHub:"
    echo "Go to GitHub > Settings > SSH and GPG keys > New SSH key, and paste it there."
    echo ""
    cat "$SSH_KEY.pub"
    echo ""
}

# Detect OS
OS_TYPE="$(uname)"
case "$OS_TYPE" in
    Darwin*) 
        echo "Detected macOS"
        if ! command -v git &> /dev/null; then
            if ! command -v brew &> /dev/null; then
                echo "Homebrew is not found. Installing Homebrew..."
                /bin/bash -c "$(curl -fsSL https://raw.githubusercontent.com/Homebrew/install/HEAD/install.sh)"
            fi
            echo "Installing Git via Homebrew..."
            brew install git
        fi
        configure
        ;;
    Linux*) 
        echo "Detected Linux"
        if [ -f /etc/debian_version ]; then
            echo "Debian/Ubuntu detected"
            sudo apt update && sudo apt install -y git openssh-client
        elif [ -f /etc/fedora-release ]; then
            echo "Fedora detected"
            sudo dnf install -y git openssh-clients
        elif [ -f /etc/arch-release ]; then
            echo "Arch Linux detected"
            sudo pacman -Sy git openssh
        fi
        configure
        ;;
    *)
        echo "Unknown OS: $OS_TYPE. Please install Git and SSH manually."
        ;;
esac