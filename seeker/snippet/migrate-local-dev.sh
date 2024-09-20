#date: 2024-09-20T16:50:06Z
#url: https://api.github.com/gists/e89a274a3c87610cfc415e5969f63f05
#owner: https://api.github.com/users/jeroenvervaeke

#!/bin/bash

# Check if required arguments are provided
if [ "$#" -ne 2 ]; then
    echo "Usage: $0 <container_name> <new_image_name>"
    exit 1
fi

CONTAINER_NAME="$1"
NEW_IMAGE="$2"
OLD_CONTAINER_NAME="${CONTAINER_NAME}-old"

# Function to revert changes
revert() {
    echo "Error occurred. Reverting changes..."
    docker stop "$CONTAINER_NAME" 2>/dev/null
    docker rm "$CONTAINER_NAME" 2>/dev/null
    docker rename "$OLD_CONTAINER_NAME" "$CONTAINER_NAME" 2>/dev/null
    docker start "$CONTAINER_NAME"
    exit 1
}

# Step 1: Pull the new image
echo "Pulling new image: $NEW_IMAGE"
if ! docker pull "$NEW_IMAGE"; then
    echo "Failed to pull new image. Exiting."
    exit 1
fi

# Step 2: Take the settings from the running container
# Check if the container is running
if [ "$(docker inspect -f '{{.State.Running}}' "$CONTAINER_NAME" 2>/dev/null)" != "true" ]; then
    echo "Container $CONTAINER_NAME is not running. Exiting."
    exit 1
fi

PORT_MAPPINGS=$(docker inspect --format='{{range $p, $conf := .NetworkSettings.Ports}}{{if $conf}} -p {{(index $conf 0).HostIp}}:{{(index $conf 0).HostPort}}:{{$p}}{{end}}{{end}}' "$CONTAINER_NAME" | sed 's:/tcp::g')
ENV_VARIABLES=$(docker inspect --format='{{range $index, $value := .Config.Env}} -e "{{$value}}"{{end}}' "$CONTAINER_NAME")

# Step 3: Stop the old container
echo "Stopping container: $CONTAINER_NAME"
if ! docker stop "$CONTAINER_NAME"; then
    echo "Failed to stop container. Exiting."
    exit 1
fi

# Step 4: Rename the old container
echo "Renaming container to: $OLD_CONTAINER_NAME"
if ! docker rename "$CONTAINER_NAME" "$OLD_CONTAINER_NAME"; then
    echo "Failed to rename container. Reverting."
    docker start "$CONTAINER_NAME"
    exit 1
fi

# Step 5: Start the new container with volumes, port mappings and environment variables from the old container
echo "Starting new container"
if ! docker run -d --volumes-from "$OLD_CONTAINER_NAME" $PORT_MAPPINGS $ENV_VARIABLES --name "$CONTAINER_NAME" "$NEW_IMAGE"; then
    echo "Failed to start new container. Reverting."
    revert
fi

# Step 6: Remove the old container
echo "Removing old container"
if ! docker rm "$OLD_CONTAINER_NAME"; then
    echo "Failed to remove old container. Reverting."
    revert
fi

echo "Update completed successfully!"