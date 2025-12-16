#date: 2025-12-16T17:10:16Z
#url: https://api.github.com/gists/2629551a9d02f1ea9427e23a0cceb83d
#owner: https://api.github.com/users/eskil

#!/usr/bin/env bash
set -euo pipefail

# Usage: git recent [compare_branch] [count]
refbranch="${1:-origin/master}"
count="${2:-20}"

# Collect branch metadata
rawlist=$(
  git for-each-ref \
    --sort=-committerdate \
    --format='%(refname:short)|%(HEAD)%(color:yellow)%(refname:short)|%(color:bold green)%(committerdate:relative)|%(color:blue)%(subject)|%(color:magenta)%(authorname)%(color:reset)' \
    --color=always \
    --count="$count" \
    refs/heads \
  | while IFS='|' read -r branch colored_branch date msg author; do

      # Remove possible leading '*' from current branch indicator
      clean_branch=$(echo "$branch" | tr -d '*')

      # Ahead/behind counts
      ahead=$( git rev-list --count "$refbranch..$clean_branch" 2>/dev/null || echo 0 )
      behind=$( git rev-list --count "$clean_branch..$refbranch" 2>/dev/null || echo 0 )

      # Output format for further processing
      printf "%s|%s|%s|%s|%.70s|%s\n" \
        "$ahead" "$behind" "$colored_branch" "$date" "$msg" "$author"
    done
)

# Nicely aligned table for fzf display
displaylist=$(printf "%s\n" "$rawlist" | column -ts '|')

# Launch fzf
selected=$(printf "%b" "$displaylist" | fzf \
  --ansi \
  --with-nth=1.. \
  --delimiter='|' \
  --prompt='Recent Branches › ' \
  --reverse \
  --color='pointer:161,marker:168'
)

# If user cancelled
if [[ -z "$selected" ]]; then
  exit 0
fi

# Extract the branch name (3rd column)
branch=$(echo "$selected" | awk '{print $3}')

echo "Switching to $branch"
git switch "$branch"
