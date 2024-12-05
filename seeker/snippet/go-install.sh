#date: 2024-12-05T17:03:46Z
#url: https://api.github.com/gists/d777a3533c5a9ee0a7d3c38c10225f38
#owner: https://api.github.com/users/hyhecor

#!/bin/bash
GOVERSION=$( curl https://go.dev/VERSION?m=text | head -n 1 )
GOROOT=/usr/local/go/bin

rm -rf /usr/local/go && \
    wget -O - https://go.dev/dl/${GOVERSION}.linux-amd64.tar.gz | \
    tar xz -C /usr/local

export PATH=$PATH:$GOROOT

go version