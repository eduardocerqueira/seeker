#date: 2022-05-17T16:59:20Z
#url: https://api.github.com/gists/228d42347ed93244050737f531121558
#owner: https://api.github.com/users/absane

#!/bin/bash

# Source: /usr/share/ubios-udapi-server/ips/bin/getsig.sh 
#
# These are the URLs where the source data is pulled from.
# https://assets.unifi-ai.com/idsips/5.0.5/rules.tar.gz
# https://assets.unifi-ai.com/reputation/alien.list.gz
# https://assets.unifi-ai.com/reputation/tor.list.gz

UPDATEURL="https://assets.unifi-ai.com"

for TYPE in rules alien tor; do
    echo "${UPDATEURL}/reputation/${TYPE}.list.gz"
    echo "${UPDATEURL}/idsips/5.0.5/${TYPE}.tar.gz"
done