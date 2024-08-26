#date: 2024-08-26T16:56:21Z
#url: https://api.github.com/gists/bfac6ef36f0a21d1349aa52b1f135ea2
#owner: https://api.github.com/users/bckelley

#!/bin/bash

fdisk -lu
pvscan
vgscan
vgchange -a y
lvscan
mount /dev/ubuntu-vg/root /mnt
mount --bind /dev /mnt/dev
mount --bind /proc /mnt/proc
mount --bind /sys /mnt/sys
mount /dev/sda2 /mnt/boot
mkdir /mnt/boot/efi || true
mount /dev/sda1 /mnt/boot/efi
chroot /mnt