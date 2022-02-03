#date: 2022-02-03T17:12:50Z
#url: https://api.github.com/gists/e2f2e2a4f2ca9dde8358a5e26e3671a5
#owner: https://api.github.com/users/fenze

#!/bin/sh

CONFIG=$HOME/.config/suckless
SOFTWARE='dwm'\ 'st'\ 'dmenu'\ 'surf'
LIST=$(dirname $0)/pkglist.txt

[ "$1" = clean ] && (set -x; rm -rf $CONFIG)

[ $(id -u) = 0 ] || su -c "$(dirname $0)/install.sh"

pacman --noconfirm --needed -S - < $LIST

for SOFT in $SOFTWARE; do
	git clone https://git.suckless.org/$SOFT
done