#date: 2022-07-20T17:13:03Z
#url: https://api.github.com/gists/593235bb93b7dbf60efb944f5b49838e
#owner: https://api.github.com/users/aboron

#!/usr/bin/bash

if [ "$#" -ne 1 ]; then
    echo " *** Illegal number of parameters"
    echo
    echo "Usage: vm-backup <UUID>"
    echo
    echo "Note: Backups will dump into the current directory"
    echo
    exit 0
fi

UUID=$1

vmadm get ${UUID} > ${UUID}.json

zfs snapshot zones/${UUID}@decom

zfs send -p zones/${UUID}@decom | pbzip2 > ${UUID}.zfs.bz

zfs destroy zones/${UUID}@decom
