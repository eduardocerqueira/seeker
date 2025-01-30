#date: 2025-01-30T17:08:19Z
#url: https://api.github.com/gists/abe7d6d635806c0e57dc7570baf64542
#owner: https://api.github.com/users/maurobrandoni

#!/usr/bin/env bash

# Stop all running containers
docker ps -q | xargs -I {} docker stop {}

# Remove all containers
docker ps -a -q | xargs -I {} docker rm {}

# Remove all images
docker images -q | xargs -I {} docker rmi --force {}

# Remove all unused volumes
docker volume prune -f

# Remove all unused networks
docker network prune -f

# Remove all unused secrets
docker secret prune -f

# Remove all unused plugins
docker plugin prune -f

# Remove all unused configs
docker config prune -f

# Optionally, remove all unused builds
docker builder prune -f