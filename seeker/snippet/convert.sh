#date: 2022-08-30T17:14:08Z
#url: https://api.github.com/gists/0ba14028cde77e1adc25b953ba6acf9d
#owner: https://api.github.com/users/nstgc

Refs:
1. http://mirror.cs.pitt.edu/archlinux/iso/2021.02.01/archlinux-bootstrap-2021.02.01-x86_64.tar.gz
2. https://dl-cdn.alpinelinux.org/alpine/v3.13/releases/x86_64/alpine-virt-3.13.1-x86_64.iso
3. https://wiki.alpinelinux.org/wiki/Replacing_non-Alpine_Linux_with_Alpine_remotely
4. https://wiki.archlinux.org/index.php/installation_guide#Configure_the_system

Requirement:
  Console access.

1. In Ubuntu
  cd /
  wget https://dl-cdn.alpinelinux.org/alpine/v3.13/releases/x86_64/alpine-virt-3.13.1-x86_64.iso
  dd if=alpine-virt-3.13.1-x86_64.iso of=/dev/sda
  sync
  reboot
2. In Alpine
# [Bring up networking]
vi /etc/network/interfaces, add:
  auto eth0
  iface eth0 inet dhcp
ifup eth0
# [Setup SSH]
setup-sshd
adduser tempuser
passwd
# [At this point it's easier to use SSH to copy & paste]
# [Per Ref #3]
mkdir /media/setup
cp -a /media/sda/* /media/setup
mkdir /lib/setup
cp -a /.modloop/* /lib/setup
/etc/init.d/modloop stop
umount /dev/sda
mv /media/setup/* /media/sda/
mv /lib/setup/* /.modloop/
# [Setup apk and bring in pacman]
setup-apkrepos
vi /etc/apk/repositories, enable community
apk update
apk add dosfstools e2fsprogs fdisk pacman arch-install-scripts
# [Disk partitioning & mounting]
fdisk /dev/sda(use gpt table, set esp partition 15 size 260M), set root partition 1 size remaining)
mkfs.vfat /dev/sda15
mkfs.ext4 /dev/sda1
mount /dev/sda1 /mnt
mkdir -p /mnt/boot/EFI
mount /dev/sda15 /mnt/boot/EFI
# [1G ram is not enough to hold arch bootstrap. Use HDD for now.]
mkdir /mnt/tmp
cd /mnt/tmp
wget http://mirror.cs.pitt.edu/archlinux/iso/2021.02.01/archlinux-bootstrap-2021.02.01-x86_64.tar.gz
tar xf archlinux-bootstrap-2021.02.01-x86_64.tar.gz
vi root.x86_64/etc/pacman.d/mirrorlist
arch-chroot root.x86_64/
pacman-key --init
pacman-key --populate archlinux
# [Any other way than mount again?]
mount /dev/sda1 /mnt
pacstrap /mnt base linux linux-firmware
genfstab -U /mnt >> /mnt/etc/fstab
follow https://wiki.archlinux.org/index.php/installation_guide#Configure_the_system
# [EFI boot]
pacman -S grub efibootmgr
grub-install --efi-directory=/boot --bootloader-id=GRUB
# [Any other way than fallback?]
mv /boot/EFI/grub /boot/EFI/BOOT
mv /boot/EFI/BOOT/grubx64.efi esp/EFI/BOOT/BOOTX64.EFI