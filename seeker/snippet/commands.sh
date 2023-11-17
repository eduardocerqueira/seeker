#date: 2023-11-17T17:02:43Z
#url: https://api.github.com/gists/e3819cbb78acfe0fe0e6cfd5c2aeac61
#owner: https://api.github.com/users/Morsine

#switch to root
sudo -i
#install
apt-get install libglib2.0-dev libnfc-dev autoconf libtool libusb-dev libpcsclite-dev build-essential unzip -y
#download and compile
wget https://github.com/nfc-tools/libnfc/releases/download/libnfc-1.7.1/libnfc-1.7.1.tar.bz2
tar -jxvf libnfc-1.7.1.tar.bz2
cd libnfc-1.7.1
autoreconf -vis
./configure --with-drivers=all --sysconfdir=/etc --prefix=/usr
make
make install
mkdir /etc/nfc
mkdir /etc/nfc/devices.d
#assuming that you use the usb to ttl converter
cp contrib/libnfc/pn532_via_uart2usb.conf.sample /etc/nfc/devices.d/pn532_via_uart2usb.conf
wget -O mfoc-master.zip https://github.com/nfc-tools/mfoc/archive/master.zip
unzip mfoc-master.zip
cd mfoc-master/
autoreconf -vis
./configure
make
make install
echo done