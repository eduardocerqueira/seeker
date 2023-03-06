#date: 2023-03-06T17:07:57Z
#url: https://api.github.com/gists/44e8a4405d7cdbaf234a826eeaa3885b
#owner: https://api.github.com/users/grubhub101

#!/bin/bash

# Update system clock
timedatectl set-ntp true

# Partition the disk
# TODO: replace sda with the name of your drive
sgdisk -Z /dev/sda
sgdisk -n 1::+512M -t 1:ef00 -c 1:"EFI System Partition" /dev/sda
sgdisk -n 2::+55G -t 2:8200 -c 2:"Hibernation Partition" /dev/sda
sgdisk -n 3:: -t 3:8300 -c 3:"Arch Linux" /dev/sda

# Format the partitions
mkfs.fat -F32 /dev/sda1
mkswap /dev/sda2
mkfs.ext4 /dev/sda3

# Mount the partitions
swapon /dev/sda2
mount /dev/sda3 /mnt
mkdir /mnt/boot
mount /dev/sda1 /mnt/boot

# Install essential packages
pacstrap /mnt base linux linux-firmware

# Generate filesystem table
genfstab -U /mnt >> /mnt/etc/fstab

# Change root into new system
arch-chroot /mnt /bin/bash <<EOF

# Set system clock
ln -sf /usr/share/zoneinfo/America/New_York /etc/localtime
hwclock --systohc

# Uncomment your preferred locale in /etc/locale.gen, then generate it
sed -i '/en_US.UTF-8/s/^#//g' /etc/locale.gen
locale-gen
echo "LANG=en_US.UTF-8" > /etc/locale.conf

# Set the hostname
echo "myhostname" > /etc/hostname

# Configure network
echo "127.0.0.1    localhost" >> /etc/hosts
echo "::1          localhost" >> /etc/hosts
echo "127.0.1.1    myhostname.localdomain myhostname" >> /etc/hosts

# Install and configure bootloader
pacman -S grub efibootmgr
grub-install --target=x86_64-efi --efi-directory=/boot --bootloader-id=arch_grub --recheck
grub-mkconfig -o /boot/grub/grub.cfg

# Install GNOME and audio drivers
pacman -S gnome gnome-extra pulseaudio pulseaudio-alsa alsa-utils

# Install Python, Java, and C development tools
pacman -S python java jdk-openjdk gcc make

# Install additional packages
pacman -S yay gnome-software

# Enable hibernation
echo "resume=UUID=$(blkid -s UUID -o value /dev/sda2)     none    swap    defaults        0 0" >> /etc/fstab
sed -i 's/HOOKS="base udev/HOOKS="base udev resume/g' /etc/mkinitcpio.conf
mkinitcpio -p linux

# Enable GNOME login manager
systemctl enable gdm.service

EOF

# Unmount partitions and reboot
umount -R /mnt
reboot
