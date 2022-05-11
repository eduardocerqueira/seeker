#date: 2022-05-11T17:03:42Z
#url: https://api.github.com/gists/2df9faecc53e5f584f4313e3d2c6af22
#owner: https://api.github.com/users/Morgbn

# 1) Expand Omega storage with SD card
# see: https://docs.onion.io/omega2-docs/boot-from-external-storage.html

# 2) Extend the Omega’s Available Memory (swap)
dd if=/dev/zero of=/swap.page bs=1M count=512
chmod 0600 /swap.page
mkswap /swap.page
swapon /swap.page
# check that the Swap row is populated:
free

# 3) Expand /tmp folder
mkdir /overlay/tmp
rm -rf /overlay/tmp/*
cp -a /tmp/* /overlay/tmp/
umount /tmp
[ $? -ne 0 ] && {
umount -l /tmp
}
mount /overlay/tmp/ /tmp

# 4) Build Octoprint
opkg update
opkg install python3 python-dev python-pip3 git
pip3 install pip --upgrade
pip3 -v install virtualenv

virtualenv -p python3 OctoPrint
source OctoPrint/bin/activate
pip3 install octoprint

./OctoPrint/bin/octoprint serve --iknowwhatimdoing
# You can now access to octoprint from http://omega-****.local:5000

# 5) Auto start Octoprint at startup
ln -s ./OctoPrint/bin/octoprint /usr/bin/octoprint
sed -i '/^exit 0/i octoprint serve --iknowwhatimdoing &' /etc/rc.local
