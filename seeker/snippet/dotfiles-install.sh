#date: 2025-05-01T16:59:13Z
#url: https://api.github.com/gists/ec60784b65710d289b895857d673cde7
#owner: https://api.github.com/users/TA-Solaris

#!/bin/bash

set -e

REPO="git@github.com:TA-Solaris/dotfiles.git"
DOTFILES_DIR="$HOME/dotfiles"

# Step 1: Configure Git identity
git config --global user.name "Edward Potter"
git config --global user.email "pottered2@gmail.com"

# Step 2: Generate SSH key if not present
if [ ! -f "$HOME/.ssh/id_ed25519" ]; then
    echo "🔑 Generating SSH key..."
    ssh-keygen -t ed25519 -C "pottered2@gmail.com" -f "$HOME/.ssh/id_ed25519" -N ""
    eval "$(ssh-agent -s)"
    ssh-add ~/.ssh/id_ed25519
else
    echo "✅ SSH key already exists."
fi

# Step 3: Show public key and prompt user to add to GitHub
echo "📋 Your public SSH key is:"
cat "$HOME/.ssh/id_ed25519.pub"
echo "🔗 Add this key to GitHub: https://github.com/settings/ssh/new"
read -p "Press enter to continue when done..."

# Step 4: Only clone if not already present
if [ ! -d "$DOTFILES_DIR" ]; then
    echo "📦 Cloning dotfiles repo as a bare repo..."
    echo "dotfiles" >> "$HOME/.gitignore"
    git clone --bare "$REPO" "$DOTFILES_DIR"
else
    echo "✅ Dotfiles repo already exists at $DOTFILES_DIR"
fi

# Step 5: Define config alias
alias config='/usr/bin/git --git-dir=$HOME/dotfiles/ --work-tree=$HOME'

# Step 6: Attempt to check out dotfiles
echo "📁 Attempting to check out dotfiles into ~"
if ! config checkout; then
    echo "⚠️ Conflicts detected. Backing up conflicting files..."
    mkdir -p "$HOME/.config-backup"
    config checkout 2>&1 | egrep "\s+\." | awk {'print $1'} | \
        xargs -I{} mv {} "$HOME/.config-backup"/{}
    echo "🔁 Retrying checkout..."
    config checkout
fi

# Step 7: Hide untracked files
config config --local status.showUntrackedFiles no

echo "✅ Dotfiles setup complete!"