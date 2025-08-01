#date: 2025-08-01T16:57:12Z
#url: https://api.github.com/gists/ad4d5b7f24d090f9670815f6cf190531
#owner: https://api.github.com/users/alexfilatov

#!/bin/bash

# Git Worktree Creation Script
# 
# INSTALLATION INSTRUCTIONS:
# To use this script globally from any directory:
# 
# 1. Copy this script to a global location:
#    sudo cp create_worktree.sh /usr/local/bin/create-worktree
#    sudo chmod +x /usr/local/bin/create-worktree
# 
# 2. Create an alias in your shell profile (~/.zshrc, ~/.bashrc, or ~/.bash_profile):
#    echo 'alias wt="create-worktree"' >> ~/.zshrc
#    source ~/.zshrc
# 
# 3. Now you can use it from any git repository:
#    wt feature/new-api
#    wt bugfix/vector-search
# 
# USAGE:
# ./create_worktree.sh <branch_name>  (local usage)
#  wt <branch_name>                   (global usage with alias)
# 
# EXAMPLES:
# wt feature/new-api
# wt bugfix/vector-search
# wt experiment/performance-test

# Configuration: Set your preferred AI editor (windsurf or cursor)
AI_EDITOR="windsurf"

set -e  # Exit on any error

# Check if branch name is provided
if [ $# -eq 0 ]; then
    echo "Usage: $0 <branch_name>"
    echo "Example: $0 feature/new-api"
    exit 1
fi

BRANCH_NAME="$1"

# Step 1: Get the current project's folder name
CURRENT_DIR=$(pwd)
PROJECT_NAME=$(basename "$CURRENT_DIR")

echo "Current project: $PROJECT_NAME"

# Step 2: Create a folder adjacent to the current project's folder
PARENT_DIR=$(dirname "$CURRENT_DIR")
WORKTREES_DIR="$PARENT_DIR/${PROJECT_NAME}-worktrees"

echo "Creating worktrees directory: $WORKTREES_DIR"
mkdir -p "$WORKTREES_DIR"

# Step 3: Create a git worktree and branch named $PROJECT_NAME-$BRANCH_NAME from the main project folder
WORKTREE_PATH="$WORKTREES_DIR/$PROJECT_NAME-$BRANCH_NAME"

echo "Creating git worktree at: $WORKTREE_PATH"
echo "Branch name: $BRANCH_NAME"

# Create the worktree with a new branch
git worktree add "$WORKTREE_PATH" -b "$BRANCH_NAME"

# Copy untracked files that are commonly needed
echo "Copying untracked configuration files..."

# List of common untracked files to copy
FILES_TO_COPY=(
    ".env"
    ".env.local"
    ".env.development"
    ".env.test"
    ".env.production"
    ".vscode/settings.json"
)

for file in "${FILES_TO_COPY[@]}"; do
    if [ -e "$CURRENT_DIR/$file" ]; then
        # Create directory structure if needed
        TARGET_DIR="$WORKTREE_PATH/$(dirname "$file")"
        mkdir -p "$TARGET_DIR"
        
        # Copy the file or directory
        cp -r "$CURRENT_DIR/$file" "$WORKTREE_PATH/$file"
        echo "  ✓ Copied: $file"
    fi
done

echo "✅ Git worktree created successfully!"
echo "📁 Worktree location: $WORKTREE_PATH"
echo "🌿 Branch: $BRANCH_NAME"

# Step 4: Open the worktree in the configured AI editor
echo ""
echo "Opening worktree in $AI_EDITOR..."

case "$AI_EDITOR" in
    "windsurf")
        if command -v windsurf &> /dev/null; then
            windsurf "$WORKTREE_PATH" &
            echo "✓ Opened in Windsurf"
        else
            echo "⚠️  Windsurf not found in PATH. Please install Windsurf or update AI_EDITOR variable."
        fi
        ;;
    "cursor")
        if command -v cursor &> /dev/null; then
            cursor "$WORKTREE_PATH" &
            echo "✓ Opened in Cursor"
        else
            echo "⚠️  Cursor not found in PATH. Please install Cursor or update AI_EDITOR variable."
        fi
        ;;
    *)
        echo "⚠️  Unknown editor: $AI_EDITOR. Supported editors: windsurf, cursor"
        ;;
esac

# Step 5: cd into the new worktree folder
echo ""
echo "To switch to the new worktree in terminal, run:"
echo "cd \"$WORKTREE_PATH\""

# Optional: Useful commands
echo ""
echo "Useful commands:"
echo "  List all worktrees: git worktree list"
echo "  Remove worktree: git worktree remove \"$WORKTREE_PATH\""
echo "  Prune worktrees: git worktree prune"
