#date: 2023-01-19T16:43:15Z
#url: https://api.github.com/gists/03d3e1b662d68cb60dc33ff24a0cf1b5
#owner: https://api.github.com/users/tmaillart

#!/bin/sh

set -xe

mkdir -p mkarchlinux
cd mkarchlinux

IMG=arch.img

[ -e $IMG ] && echo "$IMG already exists" >&2 && exit 2

ROOTFS=ArchLinuxARM-aarch64-latest.tar.gz
ARCH_MIRROR=http://eu.mirror.archlinuxarm.org/aarch64/core/
GRUB_TAR=$(curl -s $ARCH_MIRROR | sed -nE 's,.*href="./(grub[^"]*.tar.xz)".*,\1,p')

[ -e "$GRUB_TAR" ] || wget "$ARCH_MIRROR$GRUB_TAR"

tar -xvf "./$GRUB_TAR"

[ -e $ROOTFS ] || wget http://os.archlinuxarm.org/os/$ROOTFS

truncate -s ${1:-4G} arch.img

DEV=$(sudo losetup -fP --show arch.img)

sudo -i -u root /bin/sh<<EOF
cd "$PWD"
sed -e 's/\s*\([\+0-9a-zA-Z]*\).*/\1/' << DIS | fdisk $DEV
  o # clear the in memory partition table
  g # go GPT
  n # new partition
  p # primary partition
  1 # partition number 1
    # default - start at beginning of disk 
  +512M # boot parttion
  n # new partition
  p # primary partition
  2 # partion number 2
    # default, start immediately after preceding partition
    # default, extend partition to end of disk
  t # change type
  1 # partition number 1
  1 # efi system
  p # print the in-memory partition table
  w # write the partition table
  q # and we're done
DIS

losetup -d $DEV
EOF

MNT=$(mktemp -d)

DEV=$(sudo losetup -fP --show arch.img)

sudo -i -u root /bin/sh<<EOF
cd "$PWD"
mkfs.fat -F 16 ${DEV}p1
mkfs.ext4 ${DEV}p2
mount ${DEV}p2 $MNT
mkdir -p $MNT/boot
mount ${DEV}p1 $MNT/boot
mkdir -p $MNT/boot/grub $MNT/boot/EFI/BOOT
bsdtar -xpf $ROOTFS -C $MNT
grub-install -d usr/lib/grub/arm64-efi --target arm64-efi --efi-directory $MNT/boot -v --boot-directory $MNT/boot --force --no-nvram
cp $MNT/boot/EFI/arch/*.efi $MNT/boot/EFI/BOOT/BOOTAA64.EFI
EOF

GRUB_CFG='H4sIAFpgyWMAA7VXW2vjRhR+tn7FwQ74pbKd7C59KKJkidMawm5pHCiURYylI3mINKPOxbE3+L/3jC62ZMtJKe3LrnXmO/drRt4I7r7Cl69LmN8tlrD8dfEI94uHuTeil4UBroFZI3NmeMSybAcpClTMYAwr+lB25efPkRQJT8FqLlIwmBcZATQJSJTMYYommjrkJAYmYtBoDAF16zXGhNnMlCjS7I1GI/g8/2Xxpc08nc3CNbIYFdC7x4XOZQwFUyZMC9P5znUstccT+BN8DVeFwoRvSzEoNvDtJzBrFB5AJlkcEslLeIUeXr0K3BqiGbXbD+EbHLDObqgNDU5wzfORFpQktkEnvkVvkCspTShFhIFRFj3MNJ7pmA2dXZVhW9KYIDNWYZijsKWwkMdkYwDbXcun9msoC8OlCIa+z+Nho6UXUenCbSGV6UN4hwBRNDehcy0+humg3TnQegt68V4rNK2Hmv0UH7Thp49ebzidK4kVkTO80lDFFF4JX5XFd+fKga3jA5x6EK2lRlGl+ZLlpHLfUlpW1obHKI86t1dNBqmTqseQStZmeJpDYqiq+QAkYlMjzRvVdJjK4pRkU9YmcUS8vvnxU5is2uTNqiNr02WqbFvJaK3PyRFXyuqDyyeu1ZEOEylMWDCzrn1rNZJ7CqzgkYzryr/Qy9Q1N55GpqI1+L6QfpLJotjR70T71vKYflH2A0VphNfX+R/Lj/7T0+Juvy+tLvUMp1arqV4zheUAmNaKJ0Vyc+gvly2Hhqvy3041p8mWbMHAjcFmZDQpqe0kiEGVO2nufy5YFnJRWAM0GbXM8EiW1jh6w3ESPMNzJESoze5QFF1rOojA9Wn3IfhEY/eeymbFomcQUuUsa97A+U02Q8Q0OqFdaVBbQTOfZFjBNoxnbJXhpBkcXT3krZvT8y93b0zpC6P8ehZmXNhtiTlMGxjfulQ/uJcx5TbKmKb9U6W/+kiF9SvOFuXwW9Kw75ldMCZQyeVrTtsJ/U6xjKlFB628Duq0F2znqMEzYuENmlx/5/LwcSjYhpAw49j/acXe3y6bgh0gTZnB+IEUui1axgAWOUsRJpPJmAx0lMG0Ijn+4Lfb35eOO6i9ab73e1AvrlLTDDeYBR/gL8vRULNqu3LhoTjHG0ZjL4YqQLSLpYJ28N8JI6v5ewLZm80f4IWbddur/ym/pey3zOskuj/T3VSf57qT7H+Z7XfT/R/ke7B387mvRzvtd6FFb2pMuEVR4voEnYEuCPswC6X2CyVX9VzoE3YGuizM0qbzE67yF5rr1TlYXScOErr7kwo6d7fRkJDH5T5IXightqDEcO1rW7hzB2Pq34r9Z8cyO8Lb1fw0v1/AfaPzsb5iL7VKx8Kq8hrl9JMqgMbnhQna69+FWHychZHVdKBXUYXlmi72hNM0pzi6Kqf+EoBM7+CF7cBIYHEMNUs5C5zlHPUE4NHNRsLsinI7kLQ2AHbSkgxaj7UQltAKIyApjGROUEMyPiNtFzq3bUbLp4RGayZSJ8wtnDFuMQJDi2UMVDgIbCU3tGD6wtB17ZL/121QddolAHSzlX+ThDFXGBlJV+e0wk2iJG3fq9KqCN/B0/5rHY1nyCH4rNJa/6HxpqYzzKUy6Lr2N2J6DN2qDQAA'

FAT_UUID=$(sudo blkid -s UUID -o value ${DEV}p1)
EXT4_UUID=$(sudo blkid -s UUID -o value ${DEV}p2)
EXT4_PARTUUID=$(sudo blkid -s PARTUUID -o value ${DEV}p2)

sudo -i -u root /bin/sh<<EOF
cd "$PWD"
echo $GRUB_CFG | base64 -d | gunzip | sed "s,{{FAT-UUID}},$FAT_UUID, ; s,{{EXT4-PARTUUID}},$EXT4_PARTUUID, ; s,{{EXT4-UUID}},$EXT4_UUID," > $MNT/boot/grub/grub.cfg
# fallback
echo "Image root=UUID=$EXT4_UUID rw initrd=\initramfs-linux.img" > $MNT/boot/startup.nsh
echo "	/dev/disk/by-uuid/$FAT_UUID /boot vfat defaults 0 0" >> $MNT/etc/fstab
sync
umount -R $MNT
losetup -d $DEV
rmdir $MNT
EOF