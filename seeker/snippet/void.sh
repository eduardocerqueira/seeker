#date: 2022-07-15T17:08:19Z
#url: https://api.github.com/gists/8a4402e470cbd94a3843f7ff88e9a997
#owner: https://api.github.com/users/oxSleep

#!/usr/bin/env -S bash -xe
https://gist.github.com/oxSleep/8a4402e470cbd94a3843f7ff88e9a997
export USER = "sleepy"
export HOSTNAME = "void"

fsdisk /dev/vda < e
mkfs.ext4 -L voidlinux /dev/vda2 
mkfs.fat -F 32 -n boot /dev/vda1
mount /dev/disk/by-label/voidlinux /mnt 
mkdir -p /mnt/boot/efi
mount /dev/disk/by-label/boot /mnt/boot/efi
mkdir -p /mnt/var/db/xbps/key
cp /var/db/xbps/keys/* /mnt/var/db/xbps/keys/
mount --rbind /sys /mnt/sys && mount --make-rslave /mnt/sys
mount --rbind /dev /mnt/dev && mount --make-rslave /mnt/dev
mount --rbind /proc /mnt/proc && mount --make-rslave /mnt/proc

export REPO=https://repo-fi.voidlinux.org/current
export ARCH=x86_64
XBPS_ARCH=$ARCH xbps-install -S -r /mnt -R $REPO/current \
    base-minimal file bash dhcpcd ncurse grub-x86_64-efi usbutils pciutils \
    libgcc less man-pages kbd sudo udevd kmod acpid \
    curl wget git zip unzip intel-ucode neovim \
    dhcpcd dbus virt-manager libvirt qemu \
    xorg-minimal dejavu-fonts-ttf xclip xinit \
    kitty bspwm sxhkd