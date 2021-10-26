#date: 2021-10-26T17:10:08Z
#url: https://api.github.com/gists/ef6bd033bd69719a13c54bcf7e1e2b2e
#owner: https://api.github.com/users/m4rc1a

mkfs.vfat $11
cryptsetup luksFormat $12
cryptsetup luksOpen $12 nixos
mkfs.ext4 /dev/mapper/nixos
mount /dev/mapper/nixos /mnt
mkdir /mnt/boot
mount $11 /mnt/boot
nixos-generate-config --root /mnt
nixos-install
reboot