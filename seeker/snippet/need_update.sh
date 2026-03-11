#date: 2026-03-11T17:38:09Z
#url: https://api.github.com/gists/b01f69c3930eec05214c01a6799468ec
#owner: https://api.github.com/users/josemarcosrf

#!/usr/bin/env bash
set -euo pipefail

# Use existing agent if it has keys, otherwise start one and add from keychain
if ! ssh-add -l &>/dev/null; then
  eval "$(ssh-agent -s)"
  ssh-add --apple-use-keychain ~/.ssh/id_rsa || {
    echo "Failed to add SSH key"
    exit 1
  }
fi

# With args: recursive search for git repos under each path (same as: find path -name '.git' -type d)
# No args: top-level dirs only (legacy behavior)
if [ $# -gt 0 ]; then
  repo_dirs=()
  for path in "$@"; do
    path="${path%/}"
    if [ ! -d "$path" ]; then
      echo "⚠️  Skipping $path (not a directory)"
      continue
    fi
    count_before=${#repo_dirs[@]}
    gitdirs=$(find "$path" -name ".git" -type d 2>/dev/null)
    while IFS= read -r gitdir; do
      [ -n "$gitdir" ] && repo_dirs+=("$(dirname "$gitdir")")
    done <<< "$gitdirs"
    if [ ${#repo_dirs[@]} -eq "$count_before" ]; then
      echo "⚠️  No git repos found under $path"
    fi
  done
else
  repo_dirs=(*/)
fi

for dir in "${repo_dirs[@]}"; do
  dir="${dir%/}"
  if [ -d "$dir/.git" ]; then
    echo "🔍 Checking $dir"
    (
      cd "$dir"

     if ! git rev-parse --verify HEAD &>/dev/null; then
       echo "  ⏭️  Skipping (no commits yet)"
       exit 0
     fi

      # Make sure we have the latest info
      git fetch --quiet

      branch=$(git rev-parse --abbrev-ref HEAD)
      upstream="origin/$branch"

      if git show-ref --verify --quiet "refs/remotes/$upstream"; then
        ahead=$(git rev-list --count "$upstream..$branch")
        behind=$(git rev-list --count "$branch..$upstream")

        if [ "$behind" -gt 0 ]; then
          echo "  ⚠️  Behind $upstream by $behind commit(s)"
        elif [ "$ahead" -gt 0 ]; then
          echo "  ⬆️  Ahead of $upstream by $ahead commit(s)"
        else
          echo "  ✅ Up to date with $upstream"
        fi
      else
        echo "  ❌ No upstream branch found for $branch"
      fi
    )
  fi
done