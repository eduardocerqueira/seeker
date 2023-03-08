#date: 2023-03-08T16:56:10Z
#url: https://api.github.com/gists/fa8a63e5c856d1e6f9f58a164f413fe3
#owner: https://api.github.com/users/mohamedattahri

#!/usr/bin/env bash
 
# Simple script to upgrade an existing Go installation to the latest version.
#
# Download the latest:
#   upgrade-go
#
# Download a specific version:
#   upgrade-go VERSION [CHECKSUM] [GOOS] [GOARCH]
#
# Dependencies:
#   curl, jq
#
# (c) Mohamed Attahri – https://github.com/mohamedattahri

set -e

NUMBER=$1
CHECKSUM=$2

GOROOT=/usr/local/go
GOOS=$3
GOARCH=$4


if type "go" > /dev/null 2>&1
then
    GOOS=`go env GOOS`
    GOARCH=`go env GOARCH`
    GOROOT=`go env GOROOT`
else
    if [ -z "${GOOS}" ]
    then
        read -p "GOOS:" GOOS
    fi
    if [ -z "${GOARCH}" ]
    then
        read -p "GOARCH:" GOARCH
    fi
fi    

# Retrieve the latest version number and checksum
if [ -z "$1" ]
then
    RAW=`curl -LsS "https://go.dev/dl/?mode=json"`
    NUMBER=`echo ${RAW} | jq '.[0].version' | tr -d \" | tr -d 'go'`
    CHECKSUM=`echo ${RAW} | jq ".[0].files[] | select(.filename | test(\"go${NUMBER}.${GOOS}-${GOARCH}.tar.gz\")).sha256" | tr -d \"`
fi
VERSION=${NUMBER}.${GOOS}-${GOARCH}
TMP=/tmp/go${VERSION}.tar.gz

# Check if already installed.
if type "go" > /dev/null 2>&1
then
    if echo "go version go${NUMBER} ${GOOS}/${GOARCH}" | grep -q "$(go version)"
    then
        echo "Go ${NUMBER} ${GOOS}/${GOARCH} is already installed."
        exit 0
    fi
fi    

# Download if necessary
if [ ! -f ${TMP} ]
then
    echo "Downloading ${VERSION}.tar.gz..."
    curl -LsS "https://go.dev/dl/go${VERSION}.tar.gz" > ${TMP}
fi

# Verify checksum
if [ ! -z "$CHECKSUM" ]
then
    echo "Verifying checksum..."
    echo "${CHECKSUM} *${TMP}" | shasum -a256 -c -s
fi

sudo rm -rf ${GOROOT}
sudo tar -xf ${TMP} -C `dirname ${GOROOT}`
rm ${TMP}
echo "Go ${NUMBER} ${GOOS}/${GOARCH} installed."
