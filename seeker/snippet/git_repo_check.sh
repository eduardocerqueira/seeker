#date: 2024-02-29T17:05:57Z
#url: https://api.github.com/gists/75ede4f21865b533579028b386479031
#owner: https://api.github.com/users/dhinojosa

#!/bin/bash

# Find all folders that are git repositories
echo "Scanning for git repositories..."
repo_list=$(find ~ -type d -name .git 2>/dev/null)

# Check each repository
for repo in $repo_list; do
    dir=$(dirname "$repo")
    
    # Move to the directory
    cd "$dir"

    # Check for uncommitted changes
    uncommitted_changes=$(git status --porcelain)
    if [ ! -z "$uncommitted_changes" ]; then
        echo "[UNCOMMITTED CHANGES] $dir"
        continue
    fi

    # Check for unpushed commits
    unpushed_commits=$(git log --branches --not --remotes)
    if [ ! -z "$unpushed_commits" ]; then
        echo "[UNPUSHED COMMITS] $dir"
        continue
    fi

    # Report no issues
    echo "[CLEAN] $dir"
done
