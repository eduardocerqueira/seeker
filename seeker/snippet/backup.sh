#date: 2025-02-14T17:11:53Z
#url: https://api.github.com/gists/0a9f943823d0256dc14664d7f6a24377
#owner: https://api.github.com/users/iamnimnul

#!/bin/bash -e
set -euxo pipefail

CONTAINER_NAME="uptime-kuma"
DATETIME="$(date -Ins)"
FILENAME="uptime-kuma-${DATETIME}.zip"
mkdir -p backup
docker exec ${CONTAINER_NAME} sh -c "apt update && apt install zip && mkdir -p /root/backup"
docker exec ${CONTAINER_NAME} sh -c "zip -r /root/backup/${FILENAME} /app/data"
docker cp ${CONTAINER_NAME}:/root/backup/${FILENAME} $(pwd)/backup/
docker exec ${CONTAINER_NAME} sh -c "rm /root/backup/${FILENAME}"