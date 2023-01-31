#date: 2023-01-31T16:49:16Z
#url: https://api.github.com/gists/0aadeba3aa8bcbbc8b92a233977571ed
#owner: https://api.github.com/users/dafta

#!/bin/sh

modprobe libcomposite

mkdir cfg
mount none cfg -t configfs

cd cfg/usb_gadget/
mkdir -p g.1
cd g.1

#echo 0x04b3 > idVendor  # IN CASE BELOW DOESN'T WORK
#echo 0x4010 > idProduct # IN CASE BELOW DOESN'T WORK
echo 0x1d6b > idVendor   # Linux Foundation
echo 0x0104 > idProduct  # Multifunction Composite Gadget

echo 0x0100 > bcdDevice # v1.0.0
echo 0x0200 > bcdUSB # USB2
mkdir -p strings/0x409
echo "1234567890" > strings/0x409/serialnumber
echo "Valve" > strings/0x409/manufacturer
echo "Steam Deck" > strings/0x409/product
mkdir -p configs/c.1/strings/0x409
echo "Config 1: ECM network" > configs/c.1/strings/0x409/configuration
echo 250 > configs/c.1/MaxPower

mkdir -p functions/acm.usb0
ln -s functions/acm.usb0 configs/c.1/

mkdir -p functions/geth.usb0
HOST="48:6f:73:74:50:43"
SELF="42:61:64:55:53:42"
IFNAME="blablabla%d"
echo $HOST > functions/geth.usb0/host_addr
echo $SELF > functions/geth.usb0/dev_addr
echo $IFNAME > functions/geth.usb0/ifname
ln -s functions/geth.usb0 configs/c.1/
ls /sys/class/udc > UDC

ip link set dev usb0 up
ip addr add dev usb0 192.168.100.1/24
sysctl -w net.ipv4.ip_forward=1
iptables -t nat -A POSTROUTING -s 192.168.100.0/24 -o wlan0 -j MASQUERADE