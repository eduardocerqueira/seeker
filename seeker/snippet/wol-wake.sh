#date: 2022-10-10T17:13:58Z
#url: https://api.github.com/gists/95528b099c7feff0c6795391a7bae599
#owner: https://api.github.com/users/Natetronn

#!/bin/bash
# https://wiki.archlinux.org/index.php/Wake-on-LAN

#!/bin/bash

# definition of MAC addresses
monster=01:12:46:82:ab:4f
ghost=01:1a:d2:56:6b:e6

echo "Which PC to wake?"
echo "m) monster"
echo "g) ghost"
echo "q) quit"
read input1
case $input1 in
  m)
    /usr/bin/wol $monster
    ;;
  g)
    # uses wol over the internet provided that port 9 is forwarded to ghost on ghost's router
    /usr/bin/wol --port=9 --host=ghost.mydomain.org $ghost
    ;;
  Q|q)
    break
    ;;
esac
