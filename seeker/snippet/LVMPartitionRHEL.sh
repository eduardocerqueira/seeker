#date: 2022-12-29T16:55:01Z
#url: https://api.github.com/gists/d020640c231eeb010e65741decb9fcd2
#owner: https://api.github.com/users/lfelipe1501

#!/bin/bash

#
# Script para Ajustar partición servers con base RHEL
#
# @author   Luis Felipe <lfelipe1501@gmail.com>
# @website  https://www.lfsystems.com.co
# @version  1.0

PVNAME="$(pvs --noheadings -o name)"

(
echo p # List Partitions
echo d # Delete partition
echo   # Partition for LVM (Accept default)
echo p # List Partitions
echo n # New Partition
echo p # Primary Partition for LVM
echo   # Partition for LVM (Accept default)
echo   # First sector (Accept default)
echo   # Last sector (Accept default)
echo p # List Partitions
echo t # Assign the correct type of partition
echo   # Partition for LVM (Accept default)
echo 8e # LVM Type
echo w # Write changes
echo p # List Partitions
echo q # Quit fdisk
) | fdisk -c -u /dev/sda

pvresize $PVNAME

vgdisplay

lvextend -l +50%FREE /dev/mapper/ol-root
lvextend -l +100%FREE /dev/mapper/ol-home

xfs_growfs /dev/mapper/ol-root
xfs_growfs /dev/mapper/ol-home

df -Tlh

reboot