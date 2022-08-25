#date: 2022-08-25T15:41:10Z
#url: https://api.github.com/gists/6a770fed441e4a724d63d638c15678f6
#owner: https://api.github.com/users/patmaddox

#!/bin/sh

# Adapted from https://www.daemonology.net/blog/2019-02-16-FreeBSD-ZFS-AMIs-now-available.html

if [ ! $# -eq 2 ]; then
  echo "Usage: configure.sh <cloud> <disk>"
  echo "    cloud: aws|gcp"
  echo "    disk: e.g. da1, nda1"
  exit 1
fi

cloud=$1
disk=$2

if [ $cloud != "aws" ] && [ $cloud != "gcp" ]; then
  echo "cloud must be aws|gcp"
  exit 1
fi

# error out if disk doesn't exist
geom disk list $disk > /dev/null

# boot
gpart create -s gpt $disk
gpart add -a 4k -s 40M -t efi $disk
newfs_msdos -F 32 -c 1 /dev/${disk}p1
mount -t msdosfs -o longnames /dev/${disk}p1 /mnt
mkdir -p /mnt/EFI/BOOT
cp /boot/loader.efi /mnt/EFI/BOOT/BOOTX64.efi
umount /mnt

# root
gpart add -a 1m -t freebsd-zfs -l disk0 $disk
zpool create -o altroot=/mnt -o autoexpand=on -O compress=lz4 -O atime=off -m none -f zroot ${disk}p2
zfs create -o mountpoint=none zroot/ROOT
zfs create -o mountpoint=/ -o canmount=noauto zroot/ROOT/default
mount -t zfs zroot/ROOT/default /mnt
zpool set bootfs=zroot/ROOT/default zroot

# data
zfs create -o mountpoint=none zroot/DATA
zfs create -o mountpoint=/tmp -o exec=on -o setuid=off zroot/DATA/tmp
zfs create -o mountpoint=/usr -o canmount=off zroot/DATA/usr
zfs create zroot/DATA/usr/home
zfs create -o mountpoint=/var zroot/DATA/var
zfs create -o exec=off -o setuid=off zroot/DATA/var/audit
zfs create -o exec=off -o setuid=off zroot/DATA/var/crash
zfs create -o exec=off -o setuid=off zroot/DATA/var/log
zfs create -o atime=on zroot/DATA/var/mail
zfs create -o setuid=off zroot/DATA/var/tmp
zfs create -o canmount=off zroot/DATA/var/db

# configure
if [ ! -f /tmp/base.txz ]; then
  fetch -o /tmp/base.txz https://download.freebsd.org/ftp/releases/amd64/13.1-RELEASE/base.txz
fi
tar -xf /tmp/base.txz -C /mnt

if [ ! -f /tmp/kernel.txz ]; then
  fetch -o /tmp/kernel.txz https://download.freebsd.org/ftp/releases/amd64/13.1-RELEASE/kernel.txz
fi
tar -xf /tmp/kernel.txz -C /mnt

: > /mnt/etc/fstab

## copy cloud-provided conf files, then edit them
cp /etc/rc.conf /mnt/etc/
cp /etc/sysctl.conf /mnt/etc/
cp /boot/loader.conf /mnt/etc/
cp /etc/ssh/sshd_config /mnt/etc/ssh/
cp /etc/ntp.conf /mnt/

sysrc -f /mnt/etc/rc.conf zfs_enable="YES"

## these assume lines aren't already in there / last line wins
## one day there will be a `sysrc -f` for tunables
echo 'zfs_load="YES"' >> /mnt/boot/loader.conf
echo 'kern.geom.label.disk_ident.enable="0"' >> /mnt/boot/loader.conf
echo 'kern.geom.label.gptid.enable="0"' >> /mnt/boot/loader.conf
echo 'vfs.zfs.min_auto_ashift=12' >> /mnt/etc/sysctl.conf

## cloud-specific config
## taken from release/tools dir of freebsd-src

if [ $cloud = "aws" ]; then
  cp /boot.config /mnt/
  touch /mnt/firstboot
fi

if [ $cloud = "gcp" ]; then
  # cp /etc/resolv.conf /mnt/etc/
  cp /etc/rc.d/growfs /mnt/etc/rc.d/
  cp /etc/hosts /mnt/etc/
  cp /etc/ntp.conf /mnt/etc/
  cp /etc/syslog.conf /mnt/etc/
  cp /etc/crontab /mnt/etc/
fi

# packages
## image package config
mkdir -p /mnt/usr/local/etc/pkg/repos/
cat /etc/pkg/FreeBSD.conf | sed -e 's/quarterly/latest/' | > /mnt/usr/local/etc/pkg/repos/FreeBSD.conf
pkg -r /mnt install -y pkg
pkg -r /mnt update

if [ $cloud = "aws" ]; then
  pkg -r /mnt install -y ec2-scripts firstboot-freebsd-update firstboot-pkgs isc-dhcp44-client ebsnvme-id
fi

if [ $cloud = "gcp" ]; then
  pkg -r /mnt install -y firstboot-freebsd-update firstboot-pkgs \
	google-cloud-sdk panicmail sudo sysutils/py-google-compute-engine \
	lang/python lang/python2 lang/python3
fi

# snapshot
sync; sync; sync
zfs snapshot -r zroot@init
zpool export zroot

echo "Done. You must detach this volume and create the disk image manually."
echo "Be sure to use UEFI!"
