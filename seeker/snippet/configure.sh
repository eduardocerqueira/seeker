#date: 2023-11-23T16:41:27Z
#url: https://api.github.com/gists/f161ef1ee3efdff07f5ff3d7b036c664
#owner: https://api.github.com/users/fcoury

#!/bin/bash

set -euo pipefail

ln -sf /usr/share/zoneinfo/America/Sao_Paulo /etc/localtime
hwclock --systohc

# Locale
sed -i '/en_US.UTF-8/s/^#//g' /etc/locale.gen
locale-gen
echo LANG=en_US.UTF-8 > /etc/locale.conf
echo KEYMAP=us-acentos > /etc/vconsole.conf
echo FONT=Lat2-Terminus16 >> /etc/vconsole.conf
echo archie > /etc/hostname
echo "127.0.0.1  localhost" >> /etc/hosts
echo "::1        localhost" >> /etc/hosts
echo "127.0.0.1  archie.localdomain archie" >> /etc/hosts

mkinitcpio -P
passwd
sed -i '/^SigLevel/s/^/#/' /etc/pacman.conf
sed -i '/^#SigLevel/a SigLevel = Never' /etc/pacman.conf

pacman -S archlinuxarm-keyring
pacman-key --populate

sed -i '/^SigLevel = Never/s/^/#/' /etc/pacman.conf
sed -i '/^#SigLevel/s/^#//' /etc/pacman.conf

pacman -S grub efibootmgr
grub-install --efi-directory=/boot --bootloader-id=GRUB
grub-mkconfig -o /boot/grub/grub.cfg

echo "Type poweroff to shut the VM down, remove the CD-ROM and start it again."
