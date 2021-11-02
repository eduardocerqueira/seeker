#date: 2021-11-02T17:00:18Z
#url: https://api.github.com/gists/7e7d9fb5f5ed75b9e14b570b4e5fe687
#owner: https://api.github.com/users/jac18281828

#!/usr/bin/env bash

VERSION=$(date +%m%d%y)

docker build . -t subscribe:${VERSION} && docker run --rm -i -t subscribe:${VERSION}
