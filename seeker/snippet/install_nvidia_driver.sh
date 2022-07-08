#date: 2022-07-08T16:57:18Z
#url: https://api.github.com/gists/5e2213392b6304d18ac72f0c62675eef
#owner: https://api.github.com/users/yambottle

#!/bin/bash
export DRIVER_VERSION=$1
export BASE_URL=https://us.download.nvidia.com/tesla
curl -fSsl -O $BASE_URL/$DRIVER_VERSION/NVIDIA-Linux-x86_64-$DRIVER_VERSION.run
sudo sh NVIDIA-Linux-x86_64-$DRIVER_VERSION.run