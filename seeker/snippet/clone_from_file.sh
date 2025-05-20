#date: 2025-05-20T17:09:01Z
#url: https://api.github.com/gists/198d5125743f265748dd7862b30f9918
#owner: https://api.github.com/users/atakanargn

#!/bin/bash

if [ -z "$1" ]; then
  echo "Usage: $0 <filename.txt>"
  exit 1
fi

REPO_FILE="$1"

if [ ! -f "$REPO_FILE" ]; then
  echo "File not found: $REPO_FILE"
  exit 1
fi

echo "Clone starting: $REPO_FILE"

while IFS= read -r repo_url; do
  if [[ -n "$repo_url" ]]; then
    echo "Cloning: $repo_url"
    git clone "$repo_url"
    echo ""
  fi
done < "$REPO_FILE"

echo "All done."
