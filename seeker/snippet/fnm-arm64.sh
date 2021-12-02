#date: 2021-12-02T16:58:30Z
#url: https://api.github.com/gists/eecd852d74b7eedba1bf4aed49bdbcba
#owner: https://api.github.com/users/tw-martijn

#!/bin/bash

# assume that you have installed homebrew, wget and fx(https://github.com/antonmedv/fx)

# do all things in /tmp
mkdir -p /tmp/fnm-node;
pushd /tmp/fnm-node;
# clean tmp
rm -fr ./*;
mkdir fnm;
# save info of node from homebrew
NODE_INFO=./tmp_node_info.json;
brew info node --json > ${NODE_INFO};
# get latest version of node
NODE_VERSION=$(fx $NODE_INFO .[0].linked_keg);
echo "nodejs version: ${NODE_VERSION;}";
# create the same folder structure of fnm
mkdir -p ./fnm/node-versions/v${NODE_VERSION}/installation;
# get the file on bottles
NODE_TAR=$(fx $NODE_INFO .[0].bottle.stable.files.arm64_big_sur.url)
# sha256
TAR_SHA256=$(fx $NODE_INFO .[0].bottle.stable.files.arm64_big_sur.sha256)
# local file name
NODE_TAR_FILE="node-${NODE_VERSION}.arm64_big_sur.bottle.tar.gz"
# download file
wget ${NODE_TAR};
# check sha256
FILE_SHA=$(shasum -a 256 ${NODE_TAR_FILE} | head -c 64);
if [[ "$FILE_SHA" != "$TAR_SHA256" ]]; then
    echo "sha256 check failed";
    exit 1;
fi
# extract files
tar xzf ${NODE_TAR_FILE};
# mv all files to mirror folder of fnm
mv node/${NODE_VERSION}/* ./fnm/node-versions/v${NODE_VERSION}/installation;
# install npm and npx
pushd ./fnm/node-versions/v${NODE_VERSION}/installation;                                                                                       
cp -r libexec/lib/node_modules ./lib;                                                                                                          
ln -sf ../lib/node_modules/npm/bin/npm-cli.js ./bin/npm;                                                                                       
ln -sf ../lib/node_modules/npm/bin/npx-cli.js ./bin/npx;                                                                                       
rm -fr AUTHORS etc INSTALL_RECEIPT.json libexec share;                                                                                         
popd; 
# create needed folder in fnm's folder
mkdir -p ~/.fnm/node-versions;
# mv files to fnm's folder
mv ./fnm/node-versions/v${NODE_VERSION} ~/.fnm/node-versions;
popd;
echo "all things done";
fnm ls;