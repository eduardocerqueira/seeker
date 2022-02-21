#date: 2022-02-21T16:55:20Z
#url: https://api.github.com/gists/b2e3647625ad588e425320b8df12759c
#owner: https://api.github.com/users/kamakazix

sudo modprobe nbd max_part=8
sudo qemu-nbd --connect=/dev/nbd0 /var/lib/libvirt/images/$1.qcow2

echo ""
echo "DO NOT EXIT! THE IMAGE IS CONNECTED AND YOU CAN NOW MOUNT THE PARTITIONS. AFTER YOU ARE DONE, MANUALLY UNMOUNT ALL THE PARTITIONS AND PRESS ANY KEY TO DISCONNECT THE IMAGE."
echo ""

read -n 1 -s

sudo qemu-nbd --disconnect /dev/nbd0
sudo rmmod nbd
