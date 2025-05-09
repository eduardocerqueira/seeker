#date: 2025-05-09T17:05:09Z
#url: https://api.github.com/gists/f6f027c7dcb7aa9b52402e8d0f021f1b
#owner: https://api.github.com/users/oodaaq

#!/bin/bash -ex

# Script will install multiple ADS-B feeders to Raspberry Pi OS
# Before running make sure you have your coordinates (lat/lon in a form of DD.DDDD) and antenna height (in both feet and m) handy.

if [ "$EUID" -ne 0 ]
  then echo "Please run as root"
  exit
fi

cd /tmp

echo "dump1090"

wget https://flightaware.com/adsb/piaware/files/packages/pool/piaware/p/piaware-support/piaware-repository_7.2_all.deb
dpkg -i piaware-repository_7.2_all.deb
apt-get update
apt-get install dump1090-fa rtl-sdr -y

echo "FlightAware"

apt-get install piaware -y
piaware-config allow-auto-updates yes
piaware-config allow-manual-updates yes

echo "ADS-B Exchange"

curl -L -o /tmp/axfeed.sh https://adsbexchange.com/feed.sh
bash /tmp/axfeed.sh

echo "Planefinder.net"

PFCLIENT_VER=5.0.161
wget http://client.planefinder.net/pfclient_${PFCLIENT_VER}_armhf.deb
dpkg -i pfclient_${PFCLIENT_VER}_armhf.deb
rm -f pfclient_${PFCLIENT_VER}_armhf.deb

echo "Radarbox"

bash -c "$(wget -O - http://apt.rb24.com/inst_rbfeeder.sh)"
rbfeeder --set-network-mode on --set-network-host 127.0.0.1 --set-network-port 30005 --set-network-protocol beast --no-start

echo "OpenSky"

wget https://opensky-network.org/files/firmware/opensky-feeder_latest_armhf.deb
dpkg -i opensky-feeder_latest_armhf.deb

echo "FlightRadar24"

bash -c "$(wget -O - https://repo-feed.flightradar24.com/install_fr24_rpi.sh)"

echo "adsb.fi"
curl -L -o /tmp/feed.sh https://raw.githubusercontent.com/d4rken/adsb-fi-scripts/master/install.sh
bash /tmp/feed.sh
rm -f /tmp/feed.sh

echo "Dump 1090 maps and graphs"

bash -c "$(curl -L -o - https://github.com/wiedehopf/graphs1090/raw/master/install.sh)"
bash -c "$(wget -nv -O - https://github.com/wiedehopf/tar1090/raw/master/install.sh)"
sed -i -e 's?.*flightawareLinks.*?flightawareLinks = true;?' /usr/local/share/tar1090/html/config.js

echo "fr24feed-status
piaware-status" > /root/feeder-status.sh
chmod +x /root/feeder-status.sh

# Optional step to replace dump1090 with readsb

bash -c "$(wget -O - https://github.com/wiedehopf/adsb-scripts/raw/master/readsb-install.sh)"

MYIP=$(ip route get 1.2.3.4 | grep -m1 -o -P 'src \K[0-9,.]*')
echo "
Links:
Map: http://${MYIP}/tar1090/
Graphs: http://${MYIP}/graphs1090/
Planefinder: http://${MYIP}:30053/
FR24 Status: http://${MYIP}:8754/
Radarbox: https://www.radarbox.com/raspberry-pi/claim
FlightAware: https://flightaware.com/adsb/piaware/claim
FlightRadar24: https://www.flightradar24.com/activate-raspberry-pi
https://www.adsbexchange.com/myip/

Please reboot
"