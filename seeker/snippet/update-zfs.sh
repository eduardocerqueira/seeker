#date: 2022-05-27T17:03:35Z
#url: https://api.github.com/gists/311c0ff31164d3cab4a38ea71cb4b01f
#owner: https://api.github.com/users/brettinternet

#!/bin/bash

# Source: https://github.com/Morganamilo/paru/issues/707#issuecomment-1116491359

# e.g. paru or yay
AUR_HELPER="paru"

$AUR_HELPER -Sy

g='/Version/{print $3}'
d1=$($AUR_HELPER -Qi zfs-dkms | gawk "$g")
d2=$($AUR_HELPER -Si zfs-dkms | gawk "$g")
u1=$($AUR_HELPER -Qi zfs-utils | gawk "$g")
u2=$($AUR_HELPER -Si zfs-utils | gawk "$g")

if [[ $d1 == $d2 || $u1 == $u2 ]]; then
	echo "zfs is up to date"
	exit 0
fi

$AUR_HELPER -Sy zfs-dkms zfs-utils \
 --assume-installed zfs-dkms=$d1 --assume-installed zfs-dkms=$d2 \
 --assume-installed zfs-utils=$u1 --assume-installed zfs-utils=$u2
