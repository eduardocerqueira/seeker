#date: 2023-09-14T17:04:21Z
#url: https://api.github.com/gists/e9720c177c45f1908dce00d5bac5688c
#owner: https://api.github.com/users/Djaytan

#!/usr/bin/env bash

set -Eeuo pipefail
trap 'echo Error encountered while executing the script.' ERR

SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" &>/dev/null && pwd -P)

CLUSTER_NAME='<cluster-name>'
REGISTRY_NAME='<registry-name>-registry.localhost'

if [[ $(k3d registry list | grep "${REGISTRY_NAME}" | wc -w) == 0 ]]; then
  echo "Creating k3d registry '${REGISTRY_NAME}'..."
  k3d registry create "${REGISTRY_NAME}" --port 24659 --no-help
  echo "k3d registry '${REGISTRY_NAME}' created successfully."
else
  echo "k3d registry '${REGISTRY_NAME}' already exist."
fi

if [[ $(k3d cluster list | grep "${CLUSTER_NAME}" | wc -w) == 0 ]]; then
  echo "Creating k3d cluster '${CLUSTER_NAME}'..."
  k3d cluster create "${CLUSTER_NAME}" \
    --registry-use "k3d-${REGISTRY_NAME}:24659" \
    -p 31457:31457@server:0
  echo "k3d cluster '${CLUSTER_NAME}' created successfully."
else
  echo "k3d cluster '${CLUSTER_NAME}' already exist."
fi



# Then when having to push images locally:

OCI_REGISTRY_HOSTNAME='localhost:24659'
IMAGE_NAME='my/application'
IMAGE_TAG='latest'

# OCI image build logic here...

docker tag "${IMAGE_NAME}:${IMAGE_TAG}" "${OCI_REGISTRY_HOSTNAME}/${IMAGE_NAME}:${IMAGE_TAG}"
docker push "${OCI_REGISTRY_HOSTNAME}/${IMAGE_NAME}:${IMAGE_TAG}"
