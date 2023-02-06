#date: 2023-02-06T17:01:07Z
#url: https://api.github.com/gists/50cbd2618d7858a61fd474d536d03ca2
#owner: https://api.github.com/users/socheatsok78

#!/usr/bin/env bash
set -e
docker_api() { curl --silent --fail --no-buffer --unix-socket /var/run/docker.sock "http://localhost$*" > /dev/null; }
echo -n "Waiting for Docker to be ready:"
until docker_api "/_ping"; do printf "."; sleep 1; done
for((i=0; i<5; i++)); do printf "."; sleep 1; done
echo "[READY]"
