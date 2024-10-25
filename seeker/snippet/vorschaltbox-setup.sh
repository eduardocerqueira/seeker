#date: 2024-10-25T14:43:35Z
#url: https://api.github.com/gists/b3ad774b3f98c98e34466576fb8cdd9a
#owner: https://api.github.com/users/stefanbaur

#!/bin/bash -xv
# DO NOT RUN THIS IF YOU HAVE NO CLUE WHAT THIS IS ABOUT
# IT WILL SERIOUSLY MESS WITH YOUR HARD DISK PARTITIONS
# WARNING: DATA LOSS GUARANTEED!
# WARNUNG: Dieses Script ist dazu gedacht, in einem
# grundinstallierten Debian-Bookworm-Cloud-Image vom
# Typ "nocloud" ausgeführt zu werden, wie man es von
# https://cloud.debian.org/images/cloud/bookworm/latest/
# herunterladen kann. Die Ausführung auf anderen Systemen
# WIRD MIT HOHER WAHRSCHEINLICHKEIT ZU DATENVERLUST FÜHREN!
# Version: 1.0

apt update && apt upgrade -y
apt install busybox-static overlayroot tasksel fdisk console-setup-linux locales parted -y
dpkg-reconfigure locales
dpkg-reconfigure tzdata
tasksel install ssh-server || tasksel install ssh-server || dpkg --configure -a
DEVICENAME="$(df / | tail -n 1 | awk -F'[0-9]' '{ print $1 }')
parted --fix -s /dev/${DEVICENAME} align-check optimal 1
#cfdisk /dev/${DEVICENAME}
parted -s /dev/${DEVICENAME} mkpart primary ext4 $(parted -s /dev/${DEVICENAME} unit MB print free | awk '$4=="Free" && $5=="Space" { print $1 " " $2 }' | tr -d '[:alpha:]')
mkfs.ext4 -L homefs /dev/${DEVICENAME}2
echo -e "LABEL=homefs\t/home\text4\trw,sync,noexec,defaults\t0\t2" >> /etc/fstab
mount -a
apt purge unattended-upgrades -y
sed -e's/="console=/="net.ifnames=0 biosdevname=0 console=/' -i /etc/default/grub && update-grub
apt install net-tools -y
useradd -G users,sudo -m -s /bin/bash apu && passwd apu
useradd -G users -m -s /bin/bash userle && passwd userle
sed -e 's/^#PermitRootLogin/PermitRootLogin/‘ -e 's/^#PasswordAuthentication/PasswordAuthentication/‘ -i /etc/ssh/sshd_config
passwd -l root
