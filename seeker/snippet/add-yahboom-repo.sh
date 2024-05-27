#date: 2024-05-27T16:38:48Z
#url: https://api.github.com/gists/a798281c8cfb65d8db66a610cb941bc4
#owner: https://api.github.com/users/Sakura286

#!/bin/bash

set -xe

sudo mkdir -p /etc/apt/keyrings
sudo mkdir -p /etc/apt/sources.list.d

curl -fsSL https://gist.githubusercontent.com/Sakura286/0592673ec3248241c80e0ed515c2cead/raw/a61918098b13c890ddd738650c1956cd27bfde04/public.key | sudo gpg --dearmor -o /etc/apt/keyrings/yahboom-car-revyos.gpg

cat << EOF | sudo tee /etc/apt/sources.list.d/10-yahboom-repo.sources > /dev/null
Types: deb
URIs: http://yahboom-repo.sakura286.ink/public
Suites: bullseye
Components: main
Signed-By: /etc/apt/keyrings/yahboom-car-revyos.gpg
EOF

sudo apt update