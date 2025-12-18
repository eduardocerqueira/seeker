#date: 2025-12-18T17:02:00Z
#url: https://api.github.com/gists/8a06b66b769030dd91fa42b24a9bd3cf
#owner: https://api.github.com/users/chaws

#!/bin/bash

set -xe

IMAGE_NAME="claude-code-docker"
DOCKERFILE_PATH="$(dirname "$0")/Dockerfile.claude"

# Build the image if it doesn't exist
if ! docker image inspect "$IMAGE_NAME" &> /dev/null; then
    echo "Image $IMAGE_NAME not found. Building..."
    docker build -f "$DOCKERFILE_PATH" -t "$IMAGE_NAME" "$(dirname "$0")"
fi

# Ensure Claude config directory and file exist
mkdir -p "$HOME/.claude"
touch "$HOME/.claude.json"

# Run Claude Code in Docker container
# - Mount current directory to /workspace
# - Mount Claude config directory and file for authentication persistence
# - Pass current user UID/GID to maintain file permissions
# - Interactive with TTY for proper CLI experience
docker run -it --rm \
    -e HOST_UID="$(id -u)" \
    -e HOST_GID="$(id -g)" \
    -e HOST_USER="$USER" \
    -e HOST_HOME="/home/$USER" \
    -v "$PWD:/workspace" \
    -v "$HOME/.claude:/home/$USER/.claude" \
    -v "$HOME/.claude.json:/home/$USER/.claude.json" \
    -w /workspace \
    "$IMAGE_NAME" "$@"