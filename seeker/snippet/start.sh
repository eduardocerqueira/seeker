#date: 2025-01-31T17:06:43Z
#url: https://api.github.com/gists/a16b1951c91ee5c1ce770a351595c876
#owner: https://api.github.com/users/chand1012

# Name of the Docker container
CONTAINER_NAME="avalanche-node"

# Directory for persistent data
DATA_DIR="$(pwd)/avalanche"

# AvalancheGo Docker image
IMAGE="avaplatform/avalanchego:latest"

# Mainnet network ID
NETWORK_ID="1"

# Check if Docker is installed
if ! command -v docker &> /dev/null
then
    echo "Docker is not installed. Please install Docker and try again."
    exit 1
fi

# Create data directory if it doesn't exist
if [ ! -d "$DATA_DIR" ]; then
    echo "Creating data directory at $DATA_DIR"
    mkdir -p "$DATA_DIR"
fi

# Pull the latest AvalancheGo image
echo "Pulling the latest AvalancheGo Docker image..."
docker pull $IMAGE

# Check if the container is already running
if [ "$(docker ps -q -f name=$CONTAINER_NAME)" ]; then
    echo "Container '$CONTAINER_NAME' is already running."
    exit 0
fi

# Check if the container exists but is stopped
if [ "$(docker ps -aq -f status=exited -f name=$CONTAINER_NAME)" ]; then
    echo "Removing existing stopped container '$CONTAINER_NAME'..."
    docker rm $CONTAINER_NAME
fi

# Run the AvalancheGo container
echo "Starting AvalancheGo node..."
docker run -d \
    --restart unless-stopped \
    --name $CONTAINER_NAME \
    -v "$DATA_DIR":/root/.avalanchego \
    -p 9650:9650 \
    -p 9651:9651 \
    -e NETWORK_ID=$NETWORK_ID \
    $IMAGE \
    /avalanchego/build/avalanchego \
    --http-host=0.0.0.0 \
    --log-level=INFO \
    --partial-sync-primary-network

echo "AvalancheGo node started successfully."
echo "Access the API at http://localhost:9650"