#date: 2023-07-21T16:59:07Z
#url: https://api.github.com/gists/fbaa41a40494fd43435b5bd11f7d0315
#owner: https://api.github.com/users/LOCNNIL

#!/bin/bash

set -e

if [[ "$EUID" -ne 0 ]]; then
  echo "Error, run as root:"
  echo "sudo update_go"
  exit 1
fi

GOVERSION=""
GOARCH=""

# Check the number of arguments
if [ $# -ge 1 ]; then
    # If the first argument exists, print "ok"
    GOVERSION=$1
else
    # If the first argument does not exist, print a helper message
    echo "[ERROR] Please provide the version of the software to be downloaded as the first argument."
    exit 1
fi

if [ -n "$2" ]; then
    GOARCH=$2
    echo "[INFO] Chossen architecture: $2"
else
    GOARCH="amd64"
    echo "[INFO] No Architechure provide, using the default arch: $GOARCH"
fi

pushd /tmp

echo "[INFO] Selected version: $GOVERSION"

echo "[INFO] Downloading GO $GOVERSION package..."
wget https://go.dev/dl/go$GOVERSION.linux-$GOARCH.tar.gz

echo "[INFO] Removing current version of GO..."
rm -rf /usr/local/go

echo "[INFO] Installing GO $GOVERSION"

tar -C /usr/local -xzf go$GOVERSION.linux-$GOARCH.tar.gz

popd

export PATH=$PATH:/usr/local/go/bin
echo "[INFO] Testing GO $GOVERSION installation"
go version