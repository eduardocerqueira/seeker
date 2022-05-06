#date: 2022-05-06T17:15:12Z
#url: https://api.github.com/gists/d9f1b89092a171a0c06f8bc824faefb5
#owner: https://api.github.com/users/gsrai

#!/usr/bin/env bash

GO_FILE_NAME="go1.18.1.darwin-arm64.tar.gz" # find filename on https://go.dev/dl/

# usage:
# chmod u+x upgrade_go.sh
# sudo ./upgrade_go.sh

sudo mv /usr/local/go /usr/local/_go_old
mkdir /tmp/downloads
sudo wget https://golang.org/dl/$GO_FILE_NAME -P /tmp/downloads
sudo tar -C /usr/local -xzf /tmp/downloads/$GO_FILE_NAME
sudo rm -rf /usr/local/_go_old
go version
