#date: 2025-05-21T16:44:34Z
#url: https://api.github.com/gists/3f5bd0444b4dc8a13b137f1672b5be01
#owner: https://api.github.com/users/mtrojas

#!/usr/bin/env bash

set -euo pipefail

ISLANDS_DIR="re-store/islands"
TMP_ALL_DEPS="/tmp/all-packages.txt"
TMP_UNIQUE="/tmp/unique-packages.txt"
TMP_PER_ISLAND="/tmp/island-deprecated"

> "$TMP_ALL_DEPS"
mkdir -p "$TMP_PER_ISLAND"

start=$(date +%s)

echo "🔍 Collecting production dependencies from islands..."

# Step 1: Extract all production dependencies
for island in "$ISLANDS_DIR"/*; do
  island_name=$(basename "$island")

  for part in frontend backend; do
    lockfile="$island/$part/yarn.lock"
    if [ -f "$lockfile" ]; then
      echo "📦 Found: $lockfile"

      deps=$(cd "$island/$part" && NODE_ENV=production yarn list --production --depth=0 -s --ignore-optional --json \
        | jq -r '.data.trees[].name')

      # Save list for global deduplication
      echo "$deps" >> "$TMP_ALL_DEPS"

      # Also store per-island dependencies
      echo "$deps" > "$TMP_PER_ISLAND/$island_name-$part.txt"
    fi
  done
done

# Step 2: Deduplicate globally
sort -u "$TMP_ALL_DEPS" > "$TMP_UNIQUE"

echo
echo "🔎 Querying npm registry for deprecations (sequential)..."
echo

# Step 3: Query NPM registry sequentially
declare -A DEPRECATIONS

while read -r pkg; do
  [ -z "$pkg" ] && continue

  # Extract name and version safely (handles scoped packages)
  if [[ "$pkg" == @* ]]; then
    name=$(echo "$pkg" | sed 's/@[^@]*$//' )     # handles scope
    version=$(echo "$pkg" | sed 's/.*@//')
  else
    name=$(echo "$pkg" | cut -d'@' -f1)
    version=$(echo "$pkg" | cut -d'@' -f2-)
  fi

  url="https://registry.npmjs.org/${name}"
  result=$(curl -s "$url")
  deprecated=$(echo "$result" | jq -r --arg v "$version" '.versions[$v].deprecated // empty')

  if [ -n "$deprecated" ]; then
    DEPRECATIONS["$pkg"]="$deprecated"
  fi
done < "$TMP_UNIQUE"

# Step 4: Print results per island
echo
echo "📊 Deprecated packages per island:"
echo

for file in "$TMP_PER_ISLAND"/*.txt; do
  island_part=$(basename "$file" .txt)
  found_any=false

  while read -r pkg; do
    [ -z "$pkg" ] && continue

    if [[ -n "${DEPRECATIONS[$pkg]:-}" ]]; then
      if [ "$found_any" = false ]; then
        echo "🔹 $island_part:"
        found_any=true
      fi
      printf "   ⚠ %-40s %s\n" "$pkg" "${DEPRECATIONS[$pkg]}"
    fi
  done < "$file"
done

end=$(date +%s)
duration=$(( end - start ))

echo
echo "✅ Completed in $duration seconds."
