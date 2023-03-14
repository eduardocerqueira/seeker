#date: 2023-03-14T16:50:19Z
#url: https://api.github.com/gists/06b4fa1ac38ee0c5a366eb4484053f9e
#owner: https://api.github.com/users/chengtripp

#!/bin/sh
#Inspired by https://www.mythic-beasts.com/blog/2018/11/23/openwrt-install-to-ram-run-iftop-on-a-router-with-very-limited-flash/
#
# Tested on Sky SR102 running OpenWRT 21.02.5 (
# The script is designed to run on boot, it will install python3 and pip into the tmpfs ram drive as there is insufficient space in the flash. Installed in order to ensure that ram usage doesn't cause a crash.
# Reticulum and NomadNet both store their config files in /root/ so these are not lost on power cycles.


echo 'Check for python3 and pip'
if [ ! -f /tmp/usr/bin/python3 ] ; then
  opkg update --no-check-certificate
  opkg install -d ram python3-light
  opkg install -d ram python3-pip
  opkg install -d ram python3-cryptography
  opkg install -d ram python3-pyserial
  opkg install -d ram python3-netifaces
fi

echo 'Export paths'
export PATH=$PATH:/tmp/usr/bin/
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/tmp/usr/lib/

echo 'Install RNS'
/tmp/usr/bin/python3.9 -m pip install rns

echo 'Install NomadNet'
/tmp/usr/bin/python3.9 -m pip install urwid
/tmp/usr/bin/python3.9 -m pip install nomadnet

echo 'Script Complete, you now can run rnsd and nomadnet'