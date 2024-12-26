#date: 2024-12-26T16:49:18Z
#url: https://api.github.com/gists/3b95de42d2396ee5bd09fcf3311d0f79
#owner: https://api.github.com/users/audacioustux

#!/bin/bash

EXT_DRIVE="/Volumes/Untitled/important"
MAIN_DRIVE="/Users/tanjimhossain/Pictures"

# Ensure fd is installed
if ! command -v fd &>/dev/null; then
  echo "Error: 'fd' is not installed. Please install it before running this script."
  exit 1
fi

MAIN_INDEX=$(mktemp)

echo "Indexing files in MAIN_DRIVE..."
fd -t f . "$MAIN_DRIVE" > "$MAIN_INDEX"

# base64 each filename to avoid issues with special characters

process_file() {
  local log=""
  local ext_file="$1"
  
  # Extract the filename from EXT_DRIVE
  local filename
  filename=$(basename "$ext_file")

  log="$log$filename"
  
  # Locate the corresponding file in MAIN_DRIVE
  local main_file
  main_file=$(grep -F -m 1 "$filename" "$MAIN_INDEX")
  
  # Check if the main file exists
  if [[ -n "$main_file" && -f "$main_file" ]]; then
    # Compare checksums to detect corruption
    if ! cmp -s "$ext_file" "$main_file"; then
      # If files differ, copy from MAIN_DRIVE to EXT_DRIVE
      cp "$main_file" "$ext_file"
      log=$'\e[1;33m'"$log: Recovered: $main_file"$'\e[0m'
    else
      log=$'\e[1;32m'"$log: OK"$'\e[0m'
    fi
  else
    log=$'\e[1;31m'"$log: Not found"$'\e[0m'
  fi

  echo "$log"
}

export -f process_file  # Export function for xargs
export MAIN_INDEX  # Export variables for the function

fd -t f . "$EXT_DRIVE" -x bash -c 'process_file "$@"' _

# Clean up temporary index files
rm "$MAIN_INDEX"

