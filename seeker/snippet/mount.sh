#date: 2022-08-29T17:12:54Z
#url: https://api.github.com/gists/870779712e755838d4f57452f513916e
#owner: https://api.github.com/users/viti95

#!/bin/sh

mkdir mnt
sudo kpartx -v -a hdddos.img
sudo mount -o rw,sync /dev/mapper/loop0p1 ./mnt
sudo thunar
