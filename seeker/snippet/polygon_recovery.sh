#date: 2023-12-13T16:46:40Z
#url: https://api.github.com/gists/8984bbc0e65b68e23fb36cc3e16b5b08
#owner: https://api.github.com/users/david-sykora

#!/bin/bash

readonly normal_color=$(tput sgr0)

[ -n "${DEBUG}" ] && set -x

wait_for_reboot() {
	read -rsp $'Press any key to reboot...\n' -n1 key
	reboot
}

ERROR() {
	local msg="$*"
	echo "$(tput setaf 1)[ERROR]: ${msg}${normal_color}" >&2
}

DIE() {
	ERROR "$*"
	wait_for_reboot
}

OK() {
	local msg="$*"
	echo "$(tput setaf 2)[OK] ${msg}${normal_color}" >&2
}

INFO() {
	local msg="$*"
	echo "$(tput setaf 3)[INFO]: ${msg}${normal_color}"
}

wipe_partition_table() {
	wipefs --all --force $1
}

create_partition_table() {
  cat <<EOF | sfdisk $1
  label: gpt
  label-id: 4E4120FD-45C2-44F9-9AFE-390754B2C3C1
  device: /dev/nvme0n1
  unit: sectors
  first-lba: 34
  last-lba: 1000215182
  sector-size: 512

  /dev/nvme0n1p1 : start=        2048, size=     1048576, type=C12A7328-F81F-11D2-BA4B-00A0C93EC93B
  /dev/nvme0n1p2 : start=     1050624, size=   997165056, type=0FC63DAF-8483-4772-8E79-3D69D8477DE4
  /dev/nvme0n1p3 : start=   998215680, size=     1998848, type=0FC63DAF-8483-4772-8E79-3D69D8477DE4
EOF
}

format_root_partition() {
	ERROR "Failed to mount ext4 root partition"
	INFO "Formatting root partition as ext4 FS"
	yes | mkfs.ext4 $1
	mount -t ext4 $1 /mnt
}

format_efi_partition() {
	ERROR "Failed to mount fat32 EFI partition"
	INFO "Formatting EFI partition as fat32 FS"
	yes | mkfs.vfat $1
	[ -d /mnt/boot/efi ] || mkdir /mnt/boot/efi
	mount $1 /mnt/boot/efi
}

chroot_install_bootloader() {
	[ -d /mnt/proc ] || mkdir /mnt/proc
	[ -d /mnt/sys ] || mkdir /mnt/sys
	[ -d /mnt/tmp ] || mkdir /mnt/tmp
	[ -d /mnt/run ] || mkdir /mnt/run
	[ -d /mnt/dev ] || mkdir /mnt/dev
	arch-chroot /mnt /bin/bash -c "export PATH=$PATH:/sbin;grub-install --target=x86_64-efi --efi-directory=/boot/efi $1"
	arch-chroot /mnt /bin/bash -c 'export PATH=$PATH:/sbin;update-grub'
}

INFO "Waitng for DHCP ack"
COUNTER=2
/bin/sh -c 'until ping -c1 192.168.1.253; do if [[ $(( COUNTER % 30 )) == 0 ]]; then dhclient -v; fi;sleep 1;COUNTER=$((COUNTER+1)); done;'

DISK_PATH=/dev/nvme0n1

INFO "Wiping all MBR&GPT signatures on ${DISK_PATH}"
wipe_partition_table ${DISK_PATH}

INFO "Creating new partition table"
create_partition_table ${DISK_PATH}

INFO "Trying to mount Linux root partition"
mount -t ext4 "${DISK_PATH}p2" /mnt || format_root_partition "${DISK_PATH}p2"
rm -rf /mnt/*

INFO "Trying to mount Linux Boot partition"
[ -d /mnt/boot/ ] || mkdir /mnt/boot
[ -d /mnt/boot/efi ] || mkdir /mnt/boot/efi
mount "${DISK_PATH}p1" /mnt/boot/efi || format_efi_partition "${DISK_PATH}p1"

INFO "Rsyncing root partition"
export RSYNC_PASSWORD= "**********"
rsync -aAXH --delete --info=progress2 --exclude=.rsyncsums rsync://wizard@192.168.1.253:/data/image/student/root/ /mnt || DIE "Synchronization failed!"

INFO "Rsyncing boot partition"
rsync -aAXH --delete --exclude=.rsyncsums --info=progress2 rsync://wizard@192.168.1.253:/data/image/student/efi/EFI /mnt/boot/efi/ || DIE "Synchronization failed!"

INFO "Installing grub bootloader"
chroot_install_bootloader ${DISK_PATH}

OK "Installation was succesfull!"
reboot
