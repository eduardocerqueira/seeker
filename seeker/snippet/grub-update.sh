#date: 2025-06-06T17:01:49Z
#url: https://api.github.com/gists/b2c0fb1699b72f96840c6043aad7c4c6
#owner: https://api.github.com/users/Bonveio

#!/bin/bash
# Auto-detect and mount the Windows partition on /dev/sda, update GRUB, and run cleanup.
# Run as sudo.

DISK="/dev/sda"
TEMP_MNT="/mnt/temp"
TARGET_MNT="/mnt"

[ ! -d "$TEMP_MNT" ] && mkdir "$TEMP_MNT"
WIN_PART=""

for part in $(ls ${DISK}[0-9]* 2>/dev/null); do
  TYPE=$(blkid -o value -s TYPE "$part")
  if [ "$TYPE" == "ntfs" ]; then
    echo "Checking $part..."
    mount -t ntfs-3g -o ro "$part" "$TEMP_MNT" 2>/dev/null
    if [ $? -eq 0 ]; then
      if [ -d "$TEMP_MNT/Windows" ] || [ -f "$TEMP_MNT/bootmgr" ]; then
        WIN_PART="$part"
        umount "$TEMP_MNT"
        break
      fi
      umount "$TEMP_MNT"
    fi
  fi
done

if [ -z "$WIN_PART" ]; then
  echo "Windows partition not found."
  exit 1
fi

echo "Found Windows partition: $WIN_PART"
mount -t ntfs-3g "$WIN_PART" "$TARGET_MNT"
if [ $? -ne 0 ]; then
  echo "Failed to mount $WIN_PART."
  exit 1
fi

echo "Mounted $WIN_PART. Updating GRUB..."
grub-mkconfig -o /boot/grub/grub.cfg
echo "GRUB configuration updated."

# Addon cleanup and defrag steps
umount -R "$TARGET_MNT" &>/dev/null
pacman -Scc --noconfirm &>/dev/null
truncate -s 0 ~/.bash_history
history -c; history -wc
btrfs filesystem defragment -rv /