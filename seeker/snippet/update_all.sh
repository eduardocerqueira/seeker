#date: 2026-03-11T17:39:40Z
#url: https://api.github.com/gists/be7a27fdc4fa00149bacd44a1c9a733b
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
    echo "Processing $dir"
    (
      cd "$dir"

      if ! git remote get-url origin &>/dev/null; then
        echo "  ⏭️  Skipping (no remote)"
        exit 0
      fi

      # Git pull from main or master
      echo "Checking for 'main' or 'master' in $dir"
      git fetch origin

      if git show-ref --verify --quiet refs/remotes/origin/main; then
        branch="main"
      elif git show-ref --verify --quiet refs/remotes/origin/master; then
        branch="master"
      else
        echo "💥 No 'main' or 'master' branch found in $dir"
        exit 0
      fi

      echo "⬇️ Pulling latest changes from $branch in $dir"
      git switch "$branch"
      git pull origin "$branch"

      # Docker Compose build and restart
      if [ -f "docker-compose.yml" ] || [ -f "docker-compose.yaml" ]; then
        echo "🛠️ Building Docker images in $dir"
        docker compose build || echo "Docker build failed in $dir"

        echo "🔄 Restarting Docker services in $dir"
        docker compose up -d --force-recreate || echo "Docker restart failed in $dir"
      else
        echo "🦘 No docker-compose.yml found in $dir, skipping Docker commands."
      fi
    )
  fi
done

echo "✅ All done."