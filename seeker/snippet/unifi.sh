#date: 2024-09-20T16:43:07Z
#url: https://api.github.com/gists/c42e8104d2bcc5d2a801df7439f58433
#owner: https://api.github.com/users/mbc3k

#!/bin/sh
# works on SmartOS LX instances, maybe elsewhere?

# set up repos
echo 'deb [ arch=amd64,arm64 ] https://www.ui.com/downloads/unifi/debian stable ubiquiti' > /etc/apt/sources.list.d/100-ubnt-unifi.list
echo 'deb [trusted=yes] https://repo.mongodb.org/apt/ubuntu bionic/mongodb-org/3.6 multiverse' > /etc/apt/sources.list.d/mongodb-org-3.6.list

# for libssl1
echo 'deb http://security.ubuntu.com/ubuntu focal-security main' > /etc/apt/sources.list.d/old-ubuntu.list

curl -sO https://dl.ui.com/unifi/unifi-repo.gpg --output-dir /etc/apt/trusted.gpg.d/
curl -sO https://www.mongodb.org/static/pgp/server-3.6.asc --output-dir /etc/apt/trusted.gpg.d/

apt update
apt install -y unifi
