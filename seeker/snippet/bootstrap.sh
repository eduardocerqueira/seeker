#date: 2023-11-23T16:41:27Z
#url: https://api.github.com/gists/f161ef1ee3efdff07f5ff3d7b036c664
#owner: https://api.github.com/users/fcoury

#!/bin/bash

set -euo pipefail

loadkeys us-acentos
setfont Lat2-Terminus16
timedatectl 

DEVICE="/dev/vda"

# Create partitions using sfdisk
{
echo label: gpt
echo size=512M, type=uefi
echo size=512M, type=swap
echo type=linux
} | sfdisk $DEVICE

echo "Partitioning on $DEVICE is done."

# Filesystems
mkfs.ext4 /dev/vda3
mkswap /dev/vda2
mkfs.fat -F 32 /dev/vda1

# Mount the new system
mount /dev/vda3 /mnt
mount --mkdir /dev/vda1 /mnt/boot
swapon /dev/vda2

# Install the base system
pacstrap -K /mnt base base-devel linux linux-firmware e2fsprogs dhcpcd networkmanager vim neovim man-db man-pages texinfo wget
genfstab -U /mnt >> /mnt/etc/fstab
arch-chroot /mnt
