#date: 2021-12-20T17:14:37Z
#url: https://api.github.com/gists/50fcc7ea87be4ed81de0f4e5750838ec
#owner: https://api.github.com/users/dxm1603

# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #
#
# Install Wifite2 WiFi pentest tool on RaspberryPi with external WiFi adapter
# Tested on [Raspberry Pi 3 Model B] & [Alfa UQ2AWUS036H WiFi adapter]
#
# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #


# Install dependencies
sudo apt-get update && sudo apt-get install -y zsh git autoconf automake libtool pkg-config libnl-3-dev libnl-genl-3-dev libssl-dev ethtool shtool rfkill zlib1g-dev libpcap-dev libsqlite3-dev libpcre3-dev libhwloc-dev libcmocka-dev python-pip libpq-dev tshark macchanger


# Setup env (optional!)
cd ~/ && git clone https://github.com/avin/dotfiles.git --depth=1 && ./dotfiles/install.sh
echo 'alias crack-wifi="sudo python /opt/wifite2/Wifite.py sudo /opt/Wifite.py -i wlan1 --showb -mac --kill -p 30 --wps-time 30 --wps-timeouts 5 --wpat 60"' >> ~/.zshrc.local
echo 'alias crack-wifi-safe="sudo python /opt/wifite2/Wifite.py -i wlan1 --showb -mac --kill --wps --wps-only --nodeauths -p 30 --wps-time 30 --wps-timeouts 5"' >> ~/.zshrc.local
chsh -s /bin/zsh && zsh


# Prepare opt folder
mkdir -p /opt && cd /opt && sudo chmod 777 /opt


# Install aircrack-ng
git clone https://github.com/aircrack-ng/aircrack-ng.git --depth=1 --branch=1.3
cd ./aircrack-ng
./autogen.sh
./configure --with-experimental --with-ext-scripts
make
sudo make install


# Install reaver
cd /opt
git clone https://github.com/t6x/reaver-wps-fork-t6x --depth=1
cd ./reaver-wps-fork-t6x/src
./configure
make
sudo make install
     

# Install bully
cd /opt
git clone https://github.com/aanarchyy/bully --depth=1
cd ./bully/src 
make
sudo make install

# Install pixiewps
cd /opt
git clone https://github.com/wiire-a/pixiewps --depth=1
cd ./pixiewps
make
sudo make install


# Install pyrit
cd /opt
git clone https://github.com/JPaulMora/Pyrit --depth=1
cd ./Pyrit
sudo pip install psycopg2
sudo pip install scapy 
python setup.py clean
python setup.py build
sudo python setup.py install


# Install wifite2
cd /opt
git clone https://github.com/derv82/wifite2.git --depth=1
cd ./wifite2


# ======== Ready to play! ========
#
# ==== :: Fast safe check networks only with WPS (pixie-dust oriented)
# sudo ./Wifite.py -i wlan1 --showb -mac --kill --wps --wps-only --nodeauths -p 30 --wps-time 30 --wps-timeouts 5
#
# ==== :: Fast unsafe testing (with deauths!)
# sudo ./Wifite.py -i wlan1 --showb -mac --kill -p 30 --wps-time 30 --wps-timeouts 5 --wpat 60