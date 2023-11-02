#date: 2023-11-02T16:34:11Z
#url: https://api.github.com/gists/fc2f521160b067eca9b4b40efb1afb3e
#owner: https://api.github.com/users/ArloL

#!/bin/sh

set -o errexit
set -o nounset
set -o xtrace

cd "$(dirname "$0")/.." || exit 1

docker run \
  --rm \
  --interactive \
  --tty \
  --entrypoint "/bin/sh" \
  --volume "${PWD}:${PWD}" \
  --workdir "${PWD}" \
  --name eclipse \
  --platform=linux/amd64 \
  eclipse-temurin:17
