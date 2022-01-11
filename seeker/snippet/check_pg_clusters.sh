#date: 2022-01-11T16:54:36Z
#url: https://api.github.com/gists/f21b0c0fdc158d1cf5c39478ee350eca
#owner: https://api.github.com/users/FerroEduardo

#!/bin/bash
OUTPUT=$(pg_lsclusters -h)
IFS=$'\n' read -r -a LINES <<< ${OUTPUT}
for line in "${LINES[@]}"
do
    IFS=' ' read -r -a array <<< ${line}
    CLUSTER_NAME=${array[1]}
    CLUSTER_VERSION=${array[0]}
    CLUSTER_STATUS=${array[3]}
    if [ $CLUSTER_STATUS != "online" ]; then
        echo Cluster \"$CLUSTER_NAME\" is not online, starting
        /usr/bin/pg_ctlcluster $CLUSTER_VERSION $CLUSTER_NAME start
    fi
done