#date: 2022-02-18T16:53:33Z
#url: https://api.github.com/gists/c0aeb3b7b669ac242db8b175f9937fbf
#owner: https://api.github.com/users/ovik41

#!/bin/sh

sgdisk -Z -n 1:0:+512M -t 1:ef00 -n 2:0:0 /dev/sda

mkfs.fat -F 32 /dev/sda1
mkfs.ext4 -F /dev/sda2

mount /dev/sda2 /mnt
mkdir /mnt/boot
mount /dev/sda1 /mnt/boot

cd /mnt
curl -fLO https://github.com/gkisslinux/grepo/releases/download/2022.2-1/gkiss-chroot-2022.2-1.tar.xz
tar xf gkiss-chroot-2022.2-1.tar.xz

echo '/dev/sda2 / ext4 defaults,noatime 0 1
/dev/sda1 /boot vfat defaults,noatime 0 2' > /mnt/etc/fstab

echo '#!/bin/sh

export CFLAGS="-O3 -pipe -march=native"
export CXXFLAGS="$CFLAGS"
export MAKEFLAGS="-j6"
export KISS_PATH=

KISS_PATH="$KISS_PATH:$HOME/kiss/grepo/core"
KISS_PATH="$KISS_PATH:$HOME/kiss/grepo/extra"
KISS_PATH="$KISS_PATH:$HOME/kiss/grepo/nvidia"
KISS_PATH="$KISS_PATH:$HOME/kiss/grepo/wayland"
KISS_PATH="$KISS_PATH:$HOME/kiss/community/community"

export XDG_RUNTIME_DIR=/run/user/$(id -u)' > /mnt/root/.profile

/mnt/bin/kiss-chroot /mnt

mkdir /root/kiss && cd $_
git clone https://github.com/gkisslinux/grepo
git clone https://github.com/kiss-community/community

kiss u && kiss u
cd /var/db/kiss/installed && kiss b *
kiss b baseinit dosfstools e2fsprogs efibootmgr libelf ncurses util-linux

mkdir /usr/src && cd $_
curl -fLO https://cdn.kernel.org/pub/linux/kernel/v5.x/linux-5.16.9.tar.xz
tar xf linux-5.16.9.tar.xz
cd linux-5.16.9
make defconfig

# CONFIG_SYSFB_SIMPLEFB, CONFIG_FB, CONFIG_FB_SIMPLE
make menuconfig

make
make INSTALL_MOD_STRIP=1 modules_install

mkdir -p /boot/efi/boot
cp arch/x86/boot/bzImage /boot/efi/boot/bootx64.efi
efibootmgr -c -d /dev/sda -L 'Kiss Linux' -l '\efi\boot\bootx64.efi' -u 'root=/dev/sda2 rw'

kiss b libglvnd
kiss b mesa
kiss b nvidia
export KERNEL_UNAME=5.16.9
kiss b nvidia
depmod "$KERNEL_UNAME"
echo '::once:/bin/modprobe nvidia-drm modeset=1' >> /etc/inittab

locale-gen
passwd
kiss b sway
ln -s /etc/sv/seatd /var/service
adduser user
adduser user video
mv /root/.profile /root/kiss /home/user
chown -R user:user /home/user/.profile /home/user/kiss