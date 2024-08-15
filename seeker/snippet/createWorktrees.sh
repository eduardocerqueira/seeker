#date: 2024-08-15T16:41:11Z
#url: https://api.github.com/gists/61c42612a26986e64c0e7d9ef9a2cbc1
#owner: https://api.github.com/users/ChristopherHarwell

#!/bin/bash

# Check if the user provided a base directory as an argument
if [ -z "$1" ]; then
    echo "Usage: $0 <worktree_base_dir>"
    exit 1
fi

# Use the first argument as the base directory for the worktrees
WORKTREE_BASE_DIR="$1"

# Create the base directory if it doesn't exist
mkdir -p "$WORKTREE_BASE_DIR"

# Get the list of all branches
branches=$(git branch -r | grep -v '\->' | sed 's/origin\///' | sort | uniq)

# Loop through each branch and create a worktree
for branch in $branches; do
    # Define the path for the worktree
    worktree_path="$WORKTREE_BASE_DIR/$branch"

    # Check if the worktree already exists
    if [ -d "$worktree_path" ]; then
        echo "Worktree for branch '$branch' already exists at '$worktree_path'. Skipping..."
    else
        # Create the worktree
        echo "Creating worktree for branch '$branch' at '$worktree_path'..."
        git worktree add "$worktree_path" "origin/$branch"
    fi
done

echo "All worktrees have been created."