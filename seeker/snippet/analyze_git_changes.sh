#date: 2025-05-20T17:11:00Z
#url: https://api.github.com/gists/e9cb772c0b28c83d0d78afee395c58d8
#owner: https://api.github.com/users/akinncar

#!/bin/bash

# 1. List all files in the codebase, excluding specific folders
find . -type f \
  -not -path "./.git/*" \
  -not -path "./node_modules/*" \
  -not -path "./src/node_modules/*" \
  -not -path "./vendor/*" \
  -not -path "./src/ios/Pods/*" \
  | sed 's|^\./||' \
  | sort -u > all_files.txt

# 2. Get changed files from Git logs (last 12 months), normalize path
git log --since="12 months ago" --name-only --pretty=format: \
  | sed '/^$/d' \
  | sed 's|^\./||' \
  | sort -u > changed_files.txt

# 3. Get unchanged files: files in all_files but not in changed_files
comm -23 all_files.txt changed_files.txt > unchanged_files.txt

# 4. Summary
echo "📦 Total files in project: $(wc -l < all_files.txt)"
echo "✏️  Files changed in the last 12 months: $(wc -l < changed_files.txt)"
echo "🧊 Files NOT changed in the last 12 months: $(wc -l < unchanged_files.txt)"