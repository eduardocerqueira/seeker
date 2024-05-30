#date: 2024-05-30T16:41:34Z
#url: https://api.github.com/gists/5585fe12ffa44252127f057d487d934e
#owner: https://api.github.com/users/matt-development-work

#!/usr/bin/env zsh

# Get the default branch name from Git configuration (if set)
default_branch=$(git config --get init.defaultBranch 2>/dev/null)

# Check if default branch is configured
if [ -z "$default_branch" ]; then
  echo "Warning: Could not determine default branch from Git config (init.defaultBranch)."
  echo "Consider setting a default branch or modifying the script with a static name."
  exit 1
fi

# Get all branch names (excluding detached HEAD)
branches=$(git for-each-ref --format '%(refname:short)' refs/heads/ | grep -v '\*')

# Loop through each branch
for branch in $branches; do
  # Skip the default branch
  if [ "$branch" == "$default_branch" ]; then
    continue
  fi

  # Delete the branch forcefully
  git branch -D -d "$branch"
  echo "Deleted branch: $branch"
done

# Prune remotely-tracked branches
git fetch origin --prune

echo "All branches except the default branch ($default_branch) have been deleted and remotely-tracked branches have been pruned."