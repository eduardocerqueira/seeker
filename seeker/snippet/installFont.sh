#date: 2023-07-14T16:47:23Z
#url: https://api.github.com/gists/49e3344eb4609e763b6598d1b8c77c0c
#owner: https://api.github.com/users/sangoon

#!/bin/bash

die () {
    echo >&2 -e "$@"
    exit 1
}

[ "$#" -eq 2 ] || die "Usage : installFont <FONTNAME> <FONTPATH>\n2 arguments required, $# provided"
echo $1 | grep -E -q '^[a-z|A-Z]+$' || die "Invalid FONTNAME ^[a-z|A-Z]+$ required , $1 provided"

FONTNAME=$1
FONTPATH=$2

trap 'die "Installation failure !"' ERR

echo "installing $1 font"

sudo mkdir -p /usr/local/share/fonts/$FONTNAME
sudo cp $FONTPATH /usr/local/share/fonts/$FONTNAME/
sudo chown -R root: /usr/local/share/fonts/$FONTNAME
sudo chmod 644 /usr/local/share/fonts/$FONTNAME/*
sudo restorecon -vFr /usr/local/share/fonts/$FONTNAME
sudo fc-cache -v

echo "Font $1 installed !"
