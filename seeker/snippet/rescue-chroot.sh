#date: 2023-05-12T16:44:53Z
#url: https://api.github.com/gists/5ea14e7ec6ac5687c27bb0e8dace3de2
#owner: https://api.github.com/users/vpnwall-services

#!/bin/bash
mkdir /mnt/chroot
mount /dev/sdb1 /mnt/chroot
sudo mount -t sysfs /sys /mnt/chroot/sys
sudo mount -t proc /proc /mnt/chroot/proc
sudo mount --bind /dev /mnt/chroot/dev
sudo mount -t devpts /dev/pts /mnt/chroot/dev/pts
sudo mount --bind /tmp /mnt/chroot/tmp
sudo chroot /mnt/chroot/ /bin/bash 