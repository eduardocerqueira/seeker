#date: 2025-06-02T17:07:14Z
#url: https://api.github.com/gists/f7bdfc2628cbd55eefbf6d61254f49f3
#owner: https://api.github.com/users/glebsexy

#!/bin/zsh

json_file=".glitch-assets"
output_dir="downloaded_assets"
mkdir -p "$output_dir"

# Step 1: Extract deleted UUIDs
deleted_uuids=()
while read -r line; do
  if [[ "$line" == *'"deleted":true'* ]]; then
    uuid=$(echo "$line" | sed -n 's/.*"uuid":"\([^"]*\)".*/\1/p')
    deleted_uuids+=("$uuid")
  fi
done < "$json_file"

# Step 2: Download non-deleted assets
while read -r line; do
  [[ "$line" != *'"url":'* ]] && continue

  uuid=$(echo "$line" | sed -n 's/.*"uuid":"\([^"]*\)".*/\1/p')

  # Skip if in deleted_uuids
  skip=0
  for del in "${deleted_uuids[@]}"; do
    [[ "$uuid" == "$del" ]] && skip=1 && break
  done
  [[ "$skip" -eq 1 ]] && continue

  url=$(echo "$line" | sed -n 's/.*"url":"\([^"]*\)".*/\1/p')
  name=$(echo "$line" | sed -n 's/.*"name":"\([^"]*\)".*/\1/p')
  [[ -z "$name" ]] && name="$uuid"

  echo "Downloading $name ..."
  curl -s -L "$url" -o "$output_dir/$name"
done < "$json_file"

echo "All done."
