#date: 2025-04-28T17:06:07Z
#url: https://api.github.com/gists/bbabb3cac8ac254d21b331674b311329
#owner: https://api.github.com/users/dasfrosty

#!/usr/bin/env bash

set -euo pipefail

for dir in */; do
  if [ -d "$dir/.git" ]; then
    cd "$dir"

    # Check for local changes (uncommitted or untracked)
    if ! git diff --quiet || ! git diff --cached --quiet; then
      echo "========================================"
      echo "Stashing local changes in: $dir"
      echo "========================================"
      git status
      echo
      git stash -m "stash local changes for pull"

      echo "git pull ->"
      git pull

      echo "========================================"
      echo "Unstashing local changes in: $dir"
      echo "========================================"
      echo
      git stash pop

    else
      echo "git pull ->"
      git pull
    fi

    cd ..
  fi
done
