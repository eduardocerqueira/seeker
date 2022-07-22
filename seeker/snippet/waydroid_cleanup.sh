#date: 2022-07-22T17:06:44Z
#url: https://api.github.com/gists/c743e24e985d454cb9af04edb5d48ac9
#owner: https://api.github.com/users/GithubUser5462

#!/bin/bash

# https://docs.waydro.id/usage/install-on-desktops#reinstalling-waydroid

# chmod a+x waydroid_cleanup.sh;  sudo waydroid_cleanup.sh;

if [ "$EUID" -ne 0 ]
  then echo "Please run as root"
  exit
fi

rm -rf "$HOME/waydroid";
rm -rf "$HOME/.share/waydroid";
rm -rf "$HOME/.local/share/waydroid";

rm -fi "$HOME/.local/share/applications/"*"aydroid"* ;

rm -rf "/var/lib/waydroid";
rm -rf "/home/.waydroid";

echo Done;
