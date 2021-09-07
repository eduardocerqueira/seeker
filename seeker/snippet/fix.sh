#date: 2021-09-07T17:12:28Z
#url: https://api.github.com/gists/ce1d160dac1cf5c9c4845128e487004c
#owner: https://api.github.com/users/jakebrinkmann

YOUR_FLASH_DRIVE=$(/bin/ls /dev/disk/by-id/usb-*\:0)

sudo fdisk $YOUR_FLASH_DRIVE 
# o <CR> n <CR> p <CR> <CR> <CR> <CR> w <CR>

sudo mkfs.fat -F 32 $YOUR_FLASH_DRIVE-part1

sudo eject $YOUR_FLASH_DRIVE