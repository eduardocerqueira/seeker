#date: 2021-11-26T17:00:01Z
#url: https://api.github.com/gists/cb891db44d69346f3fe44e271dc56e3b
#owner: https://api.github.com/users/DevDavido

#!/bin/bash
# Download latest Docker Compose from GitHub
# Requires binaries: echo, rm, mv, chmod, uname, tr, curl, jq, sha256sum

DOCKER_COMPOSE_GITHUB_REPOSITORY="docker/compose"; \
DOCKER_COMPOSE_BINARY=$(echo "docker-compose-$(uname -s)-$(uname -m)" | tr '[:upper:]' '[:lower:]'); \
DOCKER_COMPOSE_VERSION=$(curl -fsSL -H "Accept: application/vnd.github.v3+json" "https://api.github.com/repos/${DOCKER_COMPOSE_GITHUB_REPOSITORY}/releases/latest" | jq -r ".tag_name"); \
curl -fsSLO "https://github.com/${DOCKER_COMPOSE_GITHUB_REPOSITORY}/releases/download/${DOCKER_COMPOSE_VERSION}/${DOCKER_COMPOSE_BINARY}{,.sha256}" && \
sha256sum --status -c ${DOCKER_COMPOSE_BINARY}.sha256 && rm ${DOCKER_COMPOSE_BINARY}.sha256 && \
mv ${DOCKER_COMPOSE_BINARY} /usr/bin/docker-compose && \
chmod +x /usr/bin/docker-compose