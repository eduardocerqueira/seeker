#date: 2025-12-16T17:10:16Z
#url: https://api.github.com/gists/2629551a9d02f1ea9427e23a0cceb83d
#owner: https://api.github.com/users/eskil

#!/usr/bin/env bash

set -euo pipefail

echo "Pruning..."
git fetch --prune --quiet

# Collect branches whose upstream is gone
branches=$(git for-each-ref \
  --format='%(refname:short) %(upstream:track)' refs/heads \
  | awk '$2 == "[gone]" {print $1}')

if [[ -z "$branches" ]]; then
  echo "No gone branches."
  exit 0
fi

# fzf selection with preview: last commit
selected=$(echo "$branches" | fzf \
  --multi \
  --prompt="Select gone branches to delete: " \
  --ansi \
  --reverse \
  --marker="x " \
  --pointer="- " \
  --preview='git log -n 3 --color=always --decorate --oneline --graph --pretty=medium {1}' \
  --color='pointer:161,marker:168' \
  --preview-window=right:50%)

if [[ -z "$selected" ]]; then
  echo "No branches selected."
  exit 0
fi

current_branch=$(git symbolic-ref --short HEAD 2>/dev/null || echo "")
needs_checkout=false

# If current branch is among the branches to delete → auto-checkout safe branch
if echo "$selected" | grep -qx "$current_branch"; then
  needs_checkout=true
fi

if $needs_checkout; then
  echo "Current branch '$current_branch' is selected for deletion."
  echo "Checking out a safe branch..."

  safe=""

  # Priority 1: main
  if git rev-parse --verify main >/dev/null 2>&1; then
    safe="main"
  # Priority 2: master
  elif git rev-parse --verify master >/dev/null 2>&1; then
    safe="master"
  # Priority 3: upstream default branch (if available)
  elif upstream_default=$(git symbolic-ref refs/remotes/origin/HEAD 2>/dev/null); then
    safe="${upstream_default#refs/remotes/origin/}"
  fi

  if [[ -n "$safe" ]]; then
    git checkout "$safe"
  else
    echo "No safe branch found. Detaching HEAD."
    git checkout --detach
  fi
fi

# Proceed with deletions
echo "$selected" | while read -r br; do
  echo "Deleting branch: $br"
  git branch -D "$br"
done
