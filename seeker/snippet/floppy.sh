#date: 2022-08-31T16:47:59Z
#url: https://api.github.com/gists/160a6fc6b3d434915dc6cb9b65c03d46
#owner: https://api.github.com/users/piotrek94692

#!/bin/sh

# floppyOS main installation script
# status: in early development

### variables ###
efivars = "/sys/firmware/efi/efivars"
#################

if [ ! -d "$efivars" ]; then
  echo "${efivars} not found, the system didn't boot in UEFI mode."
  echo "floppyOS doesn't support legacy boot modes."
  exit 0
fi

echo "floppyOS installer has initialized successfully."

rmmod pcspkr # TODO: Blacklist the PC speaker module on the new installation.

echo "Tried to unload the (annoying) PC speaker kernel module, if you see any errors, it doesn't matter."

timedatectl set-ntp true

echo "Enabled NTP / network time synchronization."

echo "Welcome to the floppyOS installer!"
echo "Make sure you have set up a working network connection before launching the script."
echo "Ethernet should work out of the box, for Wi-Fi please use the iwctl command."
echo "If Wi-Fi doesn't work, try running 'rfkill unblock wifi'"
echo "And then also try running 'ip link set <interface> up'"
echo "To find your interface's name, please run 'ip -c a'"
echo "It's probably named 'wlan0' by default."
echo "DHCP should work out of the box too, thanks to systemd-networkd and systemd-resolved."
echo "You shouldn't use any additional networking tools if not necessary."
echo "Use 'ping archlinux.org' to test the network connection."
echo "You can press Ctrl+C to exit the installer."

echo "What's your time zone?"
echo "Examples: Europe/Warsaw | Europe/Paris"
read timezone

timedatectl set-timezone $timezone # TODO: Run "ln -sf /usr/share/zoneinfo/$timezone /etc/localtime" on the new installation.

echo "Installer's time zone set."

echo "Which keymap do you want to use by default?"
echo "Examples: us | uk | pl | fr-latin1"
echo "(United States, United Kingdom, Poland and France, respectively.)"
read keymap

loadkeys $keymap # TODO: Run "localectl set-keymap --no-convert $keymap" on the new installation.

echo "Installer's keymap set."

fdisk -l

echo "Which disk do you want to install floppyOS on?"
echo "The installer will open cgdisk, which should be used to partition the disk."
echo "If you have partitioned the disk already, you can exit cgdisk and continue the installation."
echo "Warning: floppyOS doesn't support swap yet."
read disk

cgdisk $disk

fdisk -l

echo "What's the name of the EFI partition?"
echo "Example: /dev/sda1"
read efip

echo "What's the name of the root partition?"
echo "Example: /dev/sda2"
read rootp

mkfs.fat -F 32 $efip
mkfs.ext4 $rootp

mount $rootp /mnt
mount --mkdir $efip /mnt/boot

echo "The partitions have been formatted and the file systems have been mounted."
