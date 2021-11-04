#date: 2021-11-04T17:04:04Z
#url: https://api.github.com/gists/84a46fedeedad065f9e1aed04236ec35
#owner: https://api.github.com/users/ReturnRei

# http://askubuntu.com/questions/505446/how-to-install-ubuntu-14-04-with-raid-1-using-desktop-installer
# http://askubuntu.com/questions/660023/how-to-install-ubuntu-14-04-64-bit-with-a-dual-boot-raid-1-partition-on-an-uefi%5D

sudo -s
apt-get -y install mdadm
apt-get -y install grub-efi-amd64
sgdisk -z /dev/sda
sgdisk -z /dev/sdb
sgdisk -n 1:0:+100M -t 1:ef00 -c 1:"EFI System" /dev/sda
sgdisk -n 2:0:+8G -t 2:fd00 -c 2:"Linux RAID" /dev/sda
sgdisk -n 3:0:0 -t 3:fd00 -c 3:"Linux RAID" /dev/sda
sgdisk /dev/sda -R /dev/sdb -G
mkfs.fat -F 32 /dev/sda1
mkdir /tmp/sda1
mount /dev/sda1 /tmp/sda1
mkdir /tmp/sda1/EFI
umount /dev/sda1

mdadm --create /dev/md0 --level=0 --raid-disks=2 /dev/sd[ab]2
mdadm --create /dev/md1 --level=0 --raid-disks=2 /dev/sd[ab]3

sgdisk -z /dev/md0
sgdisk -z /dev/md1
sgdisk -N 1 -t 1:8200 -c 1:"Linux swap" /dev/md0
sgdisk -N 1 -t 1:8300 -c 1:"Linux filesystem" /dev/md1

ubiquity -b

mount /dev/md1p1 /mnt
mount -o bind /dev /mnt/dev
mount -o bind /dev/pts /mnt/dev/pts
mount -o bind /sys /mnt/sys
mount -o bind /proc /mnt/proc
cat /etc/resolv.conf >> /mnt/etc/resolv.conf
chroot /mnt

nano /etc/grub.d/10_linux
# change quick_boot and quiet_boot to 0

apt-get install -y grub-efi-amd64
apt-get install -y mdadm

nano /etc/mdadm/mdadm.conf 
# remove metadata and name

update-grub

mount /dev/sda1 /boot/efi
grub-install --boot-directory=/boot --bootloader-id=Ubuntu --target=x86_64-efi --efi-directory=/boot/efi --recheck
update-grub
umount /dev/sda1

dd if=/dev/sda1 of=/dev/sdb1

efibootmgr -c -g -d /dev/sdb -p 1 -L "Ubuntu #2" -l '\EFI\Ubuntu\grubx64.efi'

exit # from chroot
exit # from sudo -s
reboot