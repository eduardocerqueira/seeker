#date: 2025-01-31T16:40:15Z
#url: https://api.github.com/gists/cbc948c9e69cc856582cfd14d5753ec5
#owner: https://api.github.com/users/srtfisher

#!/bin/bash

# vip-index.sh:
#
# Utility script to index a WordPress VIP site using the VIP CLI with the
# ability to resume from the last processed object ID.
#
# Usage:
#   ./vip-index.sh <SITE_ALIAS>
#
# It will forward all additional arguments to the wp-cli command.

# Ensure SITE_ALIAS is provided
if [ -z "$1" ]; then
    echo "Usage: $0 <SITE_ALIAS> [extra WP-CLI arguments...]"
    exit 1
fi

# Set site alias from the first argument
SITE_ALIAS="$1"

# Shift arguments so that $@ contains only extra WP-CLI arguments
shift 1

# Function to get the last indexed Object ID
get_last_indexed_id() {
  local get_last_id_cmd="vip wp $SITE_ALIAS -- vip-search get-last-indexed-post-id $*"
  local last_id_json
  last_id_json=$($get_last_id_cmd 2>/dev/null)

  if [ $? -ne 0 ]; then
    echo ""
    return
  fi

  # Extract post_id using jq
  local post_id
  post_id=$(echo "$last_id_json" | jq -r '.post_id' 2>/dev/null)

  # Ensure it's a valid numeric ID
  if [[ "$post_id" =~ ^[0-9]+$ ]]; then
    echo "$post_id"
  else
    echo ""
  fi
}

# Initialize last indexed ID
LAST_OBJECT_ID=$(get_last_indexed_id "$@")

echo "Starting last object ID: $LAST_OBJECT_ID"

while true; do
  if [ -z "$LAST_OBJECT_ID" ]; then
    # Start indexing from the beginning
    CMD="vip wp $SITE_ALIAS -- vip-search index --skip-confirm $@"
  else
    echo "Deleting transient lock..."
    vip wp $SITE_ALIAS -- vip-search delete-transient $@

    # Resume indexing from the last known object ID
    CMD="vip wp $SITE_ALIAS -- vip-search index --skip-confirm --upper-limit-object-id=$LAST_OBJECT_ID $@"
  fi

  echo "Running command: $CMD"

  # Run the command, capturing output and exit code
  OUTPUT=$($CMD 2>&1)
  EXIT_CODE=$?

  # Log output for debugging
  echo "$OUTPUT"

  # Check exit code
  if [[ $EXIT_CODE -eq 101 ]]; then # panic error
    echo "Command exited with code 101 (deployment detected). Sleeping..."

    sleep 30

    echo "Fetching last indexed post ID..."
    LAST_OBJECT_ID=$(get_last_indexed_id "$@")

    if [ -z "$LAST_OBJECT_ID" ]; then
      echo "Failed to retrieve last indexed post ID. Restarting from scratch."
    else
      echo "Resuming from Object ID: $LAST_OBJECT_ID"
    fi

  elif [[ $EXIT_CODE -ne 0 ]]; then
    echo "Unexpected error occurred. Exit code: $EXIT_CODE"
    exit $EXIT_CODE  # Stop execution if it's an unknown failure
  else
    echo "Indexing completed successfully."
    break
  fi
done