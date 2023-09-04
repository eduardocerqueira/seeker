#date: 2023-09-04T17:08:25Z
#url: https://api.github.com/gists/e9982ac24864f775abb53ed2faf470a5
#owner: https://api.github.com/users/random-tm

#!/bin/bash
wget https://nodejs.org/dist/index.json -O /tmp/node_versions.json
NODE_VERSION=$(jq -c '[ .[] | select( .security==true and .lts!=false) ][0].version' /tmp/node_versions.json)
NODE_VERSION=$(echo "$NODE_VERSION" | tr -d '"')
rm /tmp/node_versions.json
cd /mnt/md0/custom-software/runtime-versions
rm -rf /mnt/md0/custom-software/runtime-versions/node
wget https://nodejs.org/dist/$NODE_VERSION/node-$NODE_VERSION-linux-x64.tar.xz .
mkdir node
mv *.tar.xz node/
cd node
tar -xf *.tar.xz
rm *.tar.xz
mv node-* node
cp -r node ../
rm -rf /mnt/md0/custom-software/runtime-versions/node/node