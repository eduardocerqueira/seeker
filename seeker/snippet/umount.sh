#date: 2022-08-29T17:13:35Z
#url: https://api.github.com/gists/f6d604e85d9b5969ab0e28f49dda7208
#owner: https://api.github.com/users/viti95

#!/bin/sh

sudo umount /dev/mapper/loop0p1
sudo kpartx -v -d hdddos.img
rm -d mnt
