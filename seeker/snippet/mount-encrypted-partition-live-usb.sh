#date: 2024-04-04T16:52:51Z
#url: https://api.github.com/gists/9c94b8d9f1cd6eca5a24b85febd743be
#owner: https://api.github.com/users/newtonmc

# make sure crypt module in use
sudo modprobe dm-crypt

# Find out which drive it was with the following command:
sudo fdisk -l

# You must mount /dev/sda3 myvolume
# use cryptsetup, device is accessible under /dev/mapper/myvolume
sudo cryptsetup luksOpen /dev/sde3 myvolume

# scan for LVM volumes and choose the right volume group name that you are looking for:
sudo vgscan

# list vgs by name, uuid and size
sudo vgs -o vg_name,vg_uuid,vg_size

# If it is eg. Fedora, activate it
sudo vgchange -ay system
# or
sudo vgchange -ay --select vg_uuid=<vg_uuid>

# find out root volume
sudo lvs

# mount it with the following command:
sudo mount /dev/system/root /mnt/

# to work in the volume use the following commands
sudo mount --bind /dev /mnt/dev 
sudo mount --bind /dev/pts /mnt/dev/pts
sudo mount --bind /proc /mnt/proc
sudo mount --bind /sys /mnt/sys
sudo chroot /mnt

# consider mounting /boot also
