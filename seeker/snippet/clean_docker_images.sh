#date: 2025-07-01T16:54:56Z
#url: https://api.github.com/gists/fe3e864ead15bdf47c1753f2ddf6afcf
#owner: https://api.github.com/users/lucas-kinisi

#!/bin/bash

# A script to remove Docker images based on a regex match of the repository or tag.
# This version computes the intersection of repo and tag matches when both are provided.

set -eo pipefail

usage() {
  echo "Usage: $0 [--repo <repo_regex>] [--tag <tag_regex>]"
  echo "  --repo <repo_regex>  - Regular expression to match anywhere in the image repository."
  echo "  --tag <tag_regex>   - Regular expression to match the image tag."
  echo
  echo "At least one of --repo or --tag must be provided."
  exit 1
}

REPO_REGEX=""
TAG_REGEX=""

# Parse command-line arguments
while [[ "$#" -gt 0 ]]; do
  case $1 in
    --repo) REPO_REGEX="$2"; shift ;;
    --tag) TAG_REGEX="$2"; shift ;;
    *) echo "Unknown parameter passed: $1"; usage ;;
  esac
  shift
done

# Validate that at least one regex is provided
if [ -z "$REPO_REGEX" ] && [ -z "$TAG_REGEX" ]; then
  echo "Error: At least one of --repo or --tag is required."
  usage
fi

REPO_MATCH_IDS=""
TAG_MATCH_IDS=""

# Filter by repository regex, if provided
if [ -n "$REPO_REGEX" ]; then
  REPO_MATCH_IDS=$(docker images --format "{{.Repository}} {{.ID}}" | grep -E "$REPO_REGEX" | awk '{print $NF}')
fi

# Filter by tag regex, if provided
if [ -n "$TAG_REGEX" ]; then
  TAG_MATCH_IDS=$(docker images --format "{{.Tag}} {{.ID}}" | grep -E "^$TAG_REGEX$" | awk '{print $NF}')
fi

IMAGE_IDS=""

# Determine the final set of image IDs
if [ -n "$REPO_REGEX" ] && [ -n "$TAG_REGEX" ]; then
  # Both filters are present, find the intersection
  IMAGE_IDS=$(comm -12 <(echo "$REPO_MATCH_IDS" | sort) <(echo "$TAG_MATCH_IDS" | sort))
elif [ -n "$REPO_REGEX" ]; then
  # Only repo filter is present
  IMAGE_IDS="$REPO_MATCH_IDS"
else
  # Only tag filter is present
  IMAGE_IDS="$TAG_MATCH_IDS"
fi

# Get unique image IDs
IMAGE_IDS=$(echo "$IMAGE_IDS" | sort -u)

if [ -z "$IMAGE_IDS" ]; then
  echo "No images found matching the specified criteria."
  exit 0
fi

echo "The following images are targeted for deletion:"
echo "----------------------------------------------"
docker images | head -n 1 # Print header
# Use a temporary file to safely pass IDs to grep
ID_FILE=$(mktemp)
echo "$IMAGE_IDS" > "$ID_FILE"
docker images | grep -Ff "$ID_FILE"
rm "$ID_FILE"
echo "----------------------------------------------"

read -p "Are you sure you want to delete these images? (yes/no): " CONFIRMATION

if [[ "$CONFIRMATION" =~ ^(yes|y)$ ]]; then
  echo "Deleting images..."
  if ! echo "$IMAGE_IDS" | xargs --no-run-if-empty docker rmi; then
    echo "An error occurred during image deletion. Some images may not have been deleted."
    exit 1
  fi
  echo "Deletion complete."
else
  echo "Image deletion cancelled."
fi
