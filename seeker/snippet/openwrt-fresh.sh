#date: 2022-09-06T17:01:33Z
#url: https://api.github.com/gists/1e55b74f718bdc37bfbe6ed74b3857d2
#owner: https://api.github.com/users/rajwanur

# EXTFS
# Set overlay to pendrive
opkg update && opkg install block-mount e2fsprogs kmod-fs-ext4 kmod-usb-storage kmod-usb2 kmod-usb3

DEVICE="$(sed -n -e "/\s\/overlay\s.*$/s///p" /etc/mtab)"
uci -q delete fstab.rwm
uci set fstab.rwm="mount"
uci set fstab.rwm.device="${DEVICE}"
uci set fstab.rwm.target="/rwm"
uci commit fstab

block info

DEVICE="/dev/sda1"
mkfs.ext4 ${DEVICE}

eval $(block info ${DEVICE} | grep -o -e "UUID=\S*")
uci -q delete fstab.overlay
uci set fstab.overlay="mount"
uci set fstab.overlay.uuid="${UUID}"
uci set fstab.overlay.target="/overlay"
uci commit fstab

#Transfer data
mkdir -p /tmp/cproot
mount --bind /overlay /tmp/cproot
mount ${DEVICE} /mnt
tar -C /tmp/cproot -cvf - . | tar -C /mnt -xf -
umount /tmp/cproot /mnt
reboot


# Install argon theme

mkdir -p /tmp/okibcn
cd /tmp/okibcn
wget --no-check-certificate https://github.com/jerrykuku/luci-theme-argon/releases/download/v2.2.9.4/luci-theme-argon-master_2.2.9.4_all.ipk -O luci-theme-argon-master_2.2.9.4_all.ipk

wget --no-check-certificate https://github.com/jerrykuku/luci-app-argon-config/releases/download/v0.8-beta/luci-app-argon-config_0.8-beta_all.ipk -O luci-app-argon-config_0.8-beta_all.ipk

opkg update && opkg install luci-lib-ipkg luci-compat
opkg install luci-theme-argon*.ipk luci-app-argon*.ipk


# Configure wireless radios with AP
opkg install nano
cd /etc/config 
nano wireless 

# set wifi passwords here
config wifi-iface 'default_radio0'
	option device 'radio0'
	option network 'lan'
	option mode 'ap'
	option ssid 'Thunderstorm'
	option encryption 'psk2'
	option key '###################'

config wifi-iface 'default_radio1'
	option device 'radio1'
	option network 'lan'
	option mode 'ap'
	option encryption 'sae'
	option key '####################'
	option ssid 'Thunderstorm_5z'


opkg install unbound luci-app-unbound
# https://gist.github.com/kevinoid/00656e6e4815e3ffe25dabe252e0f1e3

# Install adblock
opkg install adblock luci-app-adblock curl tcpdump

mkdir /overlay/adblock-Report
mkdir /overlay/adblock-backup
mkdir /overlay/adblock-tmp
