#date: 2023-06-20T16:39:04Z
#url: https://api.github.com/gists/9470462b236ab959104936b0b38db198
#owner: https://api.github.com/users/CyberHoward

#!/usr/bin/env bash

if [[ $(arch) == "arm64" ]]; then
  image="cosmwasm/workspace-optimizer-arm64"
else
  image="cosmwasm/workspace-optimizer"
fi

# Optimized builds
docker run --rm -v "$(pwd)":/code \
  --mount type=volume,source="$(basename "$(pwd)")_cache",target=/code/target \
  --mount type=volume,source=registry_cache,target=/usr/local/cargo/registry \
  ${image}:0.12.13