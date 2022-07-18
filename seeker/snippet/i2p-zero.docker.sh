#date: 2022-07-18T17:11:34Z
#url: https://api.github.com/gists/6e8cc118c104a5acf27a7946b85e06d0
#owner: https://api.github.com/users/santaklouse

#!/usr/bin/env bash

containerId=$(docker run -td --rm ubuntu)
docker exec -ti $containerId bash -c  '\
  apt-get update \
  && apt-get -y install git wget zip unzip \
  && git clone https://github.com/i2p-zero/i2p-zero.git --depth 1 \
  && cd i2p-zero && bash bin/build-all-and-zip.sh'
docker cp $containerId:/i2p-zero/dist-zip ./
docker container stop $containerId