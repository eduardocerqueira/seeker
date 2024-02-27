#date: 2024-02-27T17:01:56Z
#url: https://api.github.com/gists/b50b73c91a65475a12eb726caa9db176
#owner: https://api.github.com/users/meliber

#!/bin/bash

# sample name of zfs snapshot
# pool/data/home@20240220
# pool/sys/root@20240220

zfsd() {
start=$1
end=$2
connector=$3
for dataset in $(zfs list -H -o name);
do
    if [ -z $end ] && [ -z $connector ];then
        # delete snapshots of today if no args are given
        if [ -z $start ];then
            sudo zfs destroy "${dateset}@$(date +%Y%m%d)" 2>/dev/null
        else
            sudo zfs destroy "${dataset}@${start}" 2>/dev/null
        fi
    else
        for ((i=$start; i<=$end; i++));
        do
            sudo zfs destroy "${dataset}@${connector}${i}" 2>/dev/null
        done
    fi
done
}
