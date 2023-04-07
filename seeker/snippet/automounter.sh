#date: 2023-04-07T17:04:01Z
#url: https://api.github.com/gists/8d2697b4ef05b6369d137fddd6d8d52d
#owner: https://api.github.com/users/orneo1212

#!/bin/bash
DIR="/mnt"

if [[ ! -f "/etc/automount" ]]; then 
    echo "/etc/automount not exists"
    exit 1
fi

# Parse file
# File content:
# <name> <mount_options> <UUID>
declare -A mappings
while read line; do
    #skip comments
    if [[ $line == "#"* ]]; then
        continue;
    fi
    IFS=' ' read -ra DATA <<< "$line"
    DATA=($DATA)
    if [[ "${#DATA[@]}" == 3 ]]; then
        mappings[${DATA[2]}]="${DATA[0]} ${DATA[1]}"
    fi
done < /etc/automount

DEVICES=$(find /dev/ -maxdepth 1 -regextype posix-extended -regex '.*/sd[a-zA-Z]+[0-9]{1,3}')
DEVICES=(${DEVICES//$'\n'/ })
for dev in "${DEVICES[@]}" ; do
    UUID=$(blkid $dev -s UUID -o value)
    if [[ -z "$UUID" ]]; then 
        continue
    fi
    if [[ ! -z "$(mount|grep $dev)" ]]; then
        continue
    fi
    
    if [[ ! -z "${mappings[$UUID]}" ]]; then
        data=(${mappings[$UUID]})
        path="${DIR}/${data[0]}"
        mkdir -p "$path"
        mount "UUID=\"$UUID\"" -o "${data[1]}" "$path"
    fi
done