#date: 2025-07-25T16:56:05Z
#url: https://api.github.com/gists/e7631943b7e5d4304b7e5aa0b11654ea
#owner: https://api.github.com/users/raianmr

#!/bin/bash

# $BASE_DIR needs to be owned by the current user for this script to work without sudo
BASE_DIR="/mnt/rclone"

UNMOUNT=false
for arg in "$@"; do
    if [[ "$arg" == "--unmount" ]]; then
        UNMOUNT=true
    fi
done

mount_all() {
    echo ">--trying to mount all rclone remotes-->"

    rclone listremotes | while read -r REMOTE; do
        MOUNT_POINT="$BASE_DIR/${REMOTE%:}"

        mkdir -p "$MOUNT_POINT"
        chmod 777 "$MOUNT_POINT"

        rclone mount "$REMOTE" "$MOUNT_POINT" \
            --allow-other \
            --attr-timeout 1w \
            --buffer-size 1G \
            --transfers 8 \
            --read-only \
            --use-mmap \
            --daemon \
            --log-file rclone.log \
            -vv

        sleep 1

        if mountpoint -q "$MOUNT_POINT"; then
            echo "mounted $REMOTE at $MOUNT_POINT"
        fi

    done

    echo
}

unmount_all() {
    echo ">--trying to unmount all rclone remotes-->"

    rclone listremotes | while read -r REMOTE; do
        MOUNT_POINT="$BASE_DIR/${REMOTE%:}"
        fusermount -uz "$MOUNT_POINT"
        rm -r "$MOUNT_POINT"
    done

    echo
}

clean_mount() {
    unmount_all # TODO run this silently
    mount_all
}

if $UNMOUNT; then
    unmount_all
else
    clean_mount
fi

