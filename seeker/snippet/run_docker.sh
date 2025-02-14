#date: 2025-02-14T17:11:53Z
#url: https://api.github.com/gists/0a9f943823d0256dc14664d7f6a24377
#owner: https://api.github.com/users/iamnimnul

#!/bin/bash -e
set -euxo pipefail

docker run \
	--detach \
	--restart=unless-stopped \
	--publish 3001:3001 \
	--volume uptime-kuma:/app/data \
	--name uptime-kuma \
	louislam/uptime-kuma:debian
  