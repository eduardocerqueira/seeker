#date: 2024-11-19T17:12:06Z
#url: https://api.github.com/gists/88b0048d567fc5ef60be8f668ac67080
#owner: https://api.github.com/users/zichuan-li

# ---------------------------------------------------------------------------------
# Device - PogoPlug Mobile - Install OpenWrt on internal flash (128mb) and keep allowing boot to SD or USB
# ---------------------------------------------------------------------------------
http://blog.qnology.com/2015/02/openwrt-on-pogoplug-mobile.html

# Corrected URLs, because download.qnology.com is down:
http://ssl.pepas.com/pogo/mirrored/download.qnology.com/pogoplug/v4/

cd /tmp
#wget http://ssl.pepas.com/pogo/mirrored/download.qnology.com/pogoplug/v4/fw_printenv
#wget http://ssl.pepas.com/pogo/mirrored/download.qnology.com/pogoplug/v4/fw_setenv
#chmod +x fw_setenv

apt-get update
apt-get install u-boot-tools

# setup fw_env.config
echo "/dev/mtd0 0xc0000 0x20000 0x20000">/etc/fw_env.config

# 1MB for the uBoot, the rest for rootfs (~122MB)
/tmp/fw_setenv mtdparts 'mtdparts=orion_nand:1M(u-boot),-(rootfs)'

# This reboot is important
reboot

cat /proc/mtd
#dev: size erasesize name
#mtd0: 00100000 00020000 "u-boot"
#mtd1: 07f00000 00020000 "rootfs"

apt-get update
apt-get install mtd-utils
# with mtd-utils we will have ubiattach, ubimkvol, ubiformat:
#wget http://ssl.pepas.com/pogo/mirrored/download.qnology.com/pogoplug/v4/ubiattach
#wget http://ssl.pepas.com/pogo/mirrored/download.qnology.com/pogoplug/v4/ubimkvol
#wget http://ssl.pepas.com/pogo/mirrored/download.qnology.com/pogoplug/v4/ubiformat


ubiformat /dev/mtd1
#ubiformat: mtd1 (nand), size 133169152 bytes (127.0 MiB), 1016 eraseblocks of 131072 bytes (128.0 KiB), min. I/O size 2048 bytes
#libscan: scanning eraseblock 1015 -- 100 % complete
#ubiformat: 896 eraseblocks have valid erase counter, mean value is 0
#ubiformat: 86 eraseblocks are supposedly empty
#ubiformat: 2 bad eraseblocks found, numbers: 25, 26
#ubiformat: warning!: 32 of 1014 eraseblocks contain non-UBI data
#ubiformat: continue? (y/N)
y
#ubiformat: warning!: only 896 of 1014 eraseblocks have valid erase counter
#ubiformat: mean erase counter 0 will be used for the rest of eraseblock
#ubiformat: continue? (y/N)
y
#ubiformat: use erase counter 0 for all eraseblocks
#ubiformat: warning!: VID header and data offsets on flash are 2048 and 4096, which is different to requested offsets 512 and 2048
#ubiformat: use new offsets 512 and 2048? ubiformat: continue? (y/N)
y
#ubiformat: use offsets 512 and 2048
#ubiformat: formatting eraseblock 1015 -- 100 % complete
ubiattach -p /dev/mtd1
#UBI device number 0, total 1014 LEBs (130830336 bytes, 124.8 MiB), available 992 LEBs (127991808 bytes, 122.1 MiB), LEB size 129024 bytes (126.0 KiB)
ubimkvol /dev/ubi0 -m -N rootfs
#Set volume size to 127991808
# Volume ID 0, size 992 LEBs (127991808 bytes, 122.1 MiB), LEB size 129024 bytes (126.0 KiB), dynamic, name "rootfs", alignment 1
mkdir /tmp/ubi
mount -t ubifs ubi0:rootfs /tmp/ubi

# Corrected URLs, because 15.05.1 is newer than 15.05 (we could use LEDE too if it is available)
cd /tmp
wget http://downloads.openwrt.org/chaos_calmer/15.05.1/kirkwood/generic/openwrt-15.05.1-kirkwood-generic-rootfs.tar.gz
wget http://downloads.openwrt.org/chaos_calmer/15.05.1/kirkwood/generic/openwrt-15.05.1-kirkwood-zImage
wget http://downloads.openwrt.org/chaos_calmer/15.05.1/kirkwood/generic/md5sums

# check that zImage and rootfs.tar.gz are OK
md5sum -c md5sums 2>/dev/null | grep OK

# extract rootfs.tar.gz to ubi rootfs at /tmp/ubi
cd /tmp/ubi
tar xvzf /tmp/openwrt*rootfs.tar.gz

mkdir boot
mv /tmp/openwrt*kirkwood-zImage ./boot/zImage

#FDT from http://forum.doozan.com/read.php?2,12096

# This, instead of download it we already have it so we copy it :
# link down: http://download.qnology.com/pogoplug/v4/kirkwood-pogoplug_v4.dtb
#cd boot; wget http://download.qnology.com/pogoplug/v4/kirkwood-pogoplug_v4.dtb
cd boot
cp /boot/dts/kirkwood-pogoplug_v4.dtb ./kirkwood-pogoplug_v4.dtb

sync
cd /
umount /tmp/ubi

# Setup uBoot Environment
# Cut and Paste this whole section
fw_setenv fdt_file '/boot/kirkwood-pogoplug_v4.dtb'
fw_setenv loadaddr '0x800000'
fw_setenv zimage '/boot/zImage'
fw_setenv fdt_addr '0x1c00000'
fw_setenv loadubi 'echo Trying to boot from NAND ...;if run mountubi; then run loadubizimage;run loadubifdt;ubifsumount;run bootzubi;fi'
fw_setenv mountubi 'ubi part rootfs; ubifsmount ubi0:rootfs'
fw_setenv loadubizimage 'ubifsload ${loadaddr} ${zimage}'
fw_setenv loadubifdt 'ubifsload ${fdt_addr} ${fdt_file}'
fw_setenv bootzubi 'echo Booting from nand ...; run setargsubi; bootz ${loadaddr} - ${fdt_addr};'
fw_setenv setargsubi 'setenv bootargs console=${console},${baudrate} ${optargs} ubi.mtd=1 root=ubi0:rootfs rw rootfstype=ubifs rootwait ${mtdparts}'

# Setup boot order.
# USB->SD->SATA->UBI/NAND
fw_setenv bootcmd 'run bootcmd_usb; run bootcmd_mmc; run bootcmd_sata; run loadubi; reset'

#poweroff Pogoplug and remove SD Card and USB flash drive.
poweroff

# So far it works perfectly, without SD / USB it boots to OpenWrt and with Debian SD it boots to Debian
# Important: It will not work if you use an SD that was previusly used on another Pogo Device.
# the ethernet it wont work.

# Now OpenWRT will have as usual IP 192.168.1.1 with DHCP server enabled
# we will have to disable DHCP server, and configure to get automatic IP from DHCP, and assign a password
telnet 192.168.1.1

#####Initial Boot via Telnet######
#set passwd and enable ssh
passwd

# Reconfigure Network to DHCP Client
# disable dhcp server on lan
uci set dhcp.lan.ignore='1'
uci set network.lan.proto='dhcp'
uci del network.lan.ipaddr
uci del network.lan.netmask
uci commit dhcp
uci commit network
sync
/etc/init.d/dnsmasq restart
reboot
