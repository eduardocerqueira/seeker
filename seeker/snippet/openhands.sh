#date: 2025-06-16T16:51:34Z
#url: https://api.github.com/gists/13640faf99971caeeb7361853e9f1a30
#owner: https://api.github.com/users/jmizell

#!/bin/bash

export SANDBOX_VOLUMES=/var/run/docker.sock:/var/run/docker.sock

docker pull docker.all-hands.dev/all-hands-ai/runtime:0.42-nikolaik
docker kill openhands-app
docker rum openhands-app
docker run -d \
    --name openhands-app \
    -e SANDBOX_BASE_CONTAINER_IMAGE=docker.io/library/openhands-runtime-sanbox-go-docker:0.42-nikolaik \
    -e SANDBOX_VOLUMES=$SANDBOX_VOLUMES \
    -e LOG_ALL_EVENTS=true \
    -v /var/run/docker.sock:/var/run/docker.sock \
    -v ${PWD}/state:/.openhands-state \
    -p 3000:3000 \
    --add-host host.docker.internal:host-gateway \
    docker.all-hands.dev/all-hands-ai/openhands:0.42
docker logs -f openhands-app
