#date: 2022-02-14T16:55:51Z
#url: https://api.github.com/gists/a3a32c5a956a69d13269363e753571af
#owner: https://api.github.com/users/Cyclenerd

#!/bin/sh

MY_LAST_VERSION_URL="$(curl -fsSLI -o /dev/null -w "%{url_effective}" https://github.com/gitpod-io/openvscode-server/releases/latest)"
echo $MY_LAST_VERSION_URL

MY_LAST_VERSION="${MY_LAST_VERSION_URL#https://github.com/gitpod-io/openvscode-server/releases/tag/openvscode-server-v}"
echo $MY_LAST_VERSION

MY_DOWNLOAD_URL="https://github.com/gitpod-io/openvscode-server/releases/download/openvscode-server-v${MY_LAST_VERSION}/openvscode-server-v${MY_LAST_VERSION}-linux-x64.tar.gz"
echo $MY_DOWNLOAD_URL

curl -LO "$MY_DOWNLOAD_URL"

tar -xzf "openvscode-server-v${MY_LAST_VERSION}-linux-x64.tar.gz"
ln -sf "openvscode-server-v${MY_LAST_VERSION}-linux-x64" "openvscode-server"

./openvscode-server/bin/openvscode-server --without-connection-token --host "0.0.0.0"