#date: 2022-03-11T17:03:03Z
#url: https://api.github.com/gists/32d0d9e01a88c5f6da80d8a91685111e
#owner: https://api.github.com/users/vitalyford

#!/bin/bash

wget -O rules.v4 https://gist.githubusercontent.com/vitalyford/6ddb2ba24a072f2d442ee4d5ee62c006/raw/f8049ee12164bd608c8a2b2f7288897cd1f686df/rules.v4
IP=$(curl https://ifconfig.me/ip)
sed -i "s/35.208.62.165/${IP}/g" rules.v4

cp rules.v4 /etc/iptables/rules.v4
