#date: 2023-07-18T17:07:44Z
#url: https://api.github.com/gists/c78c43ad5cf79b96ab8f2fa6c6ce1d96
#owner: https://api.github.com/users/maximko

#!/usr/local/bin/bash

export BORG_REPO="ssh://user@server/borg-repo"

if [[ ! "${USER}" == "root" ]]; then
    echo Start this script as root
    exit 1
fi

echo === Present snapshots ===
tmutil listlocalsnapshots /

echo === Creating snapshot ===
name=$(tmutil localsnapshot | grep Created | egrep -o '[0-9-]*')
if [ ! -z "${name}" ]; then
    echo Created snapshot ${name}, mounting...
else
    echo Snapshot creation failed
    exit 1
fi

echo === Mounting snapshot ===
mkdir /tmp/borg-snapshot

# Using /Volumes instead of / because otherwise it says resource busy. Can be any existing dir in /
if ! mount_apfs -s com.apple.TimeMachine.${name}.local /Volumes /tmp/borg-snapshot/; then
    echo Failed to mount snapshot, exiting
    exit 1
fi

echo === BACKUP ===
borg create -C zstd --stats --progress ::$(gdate -Is) /tmp/borg-snapshot

echo === Umount and removing snapshot ===
umount /tmp/borg-snapshot && rm -rf /tmp/borg-snapshot
tmutil deletelocalsnapshots ${name}

echo === Done ===