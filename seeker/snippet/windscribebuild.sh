#date: 2021-11-18T17:05:54Z
#url: https://api.github.com/gists/84b916ea9dece94e3458595fc4573077
#owner: https://api.github.com/users/lilithium-hydride

#!/bin/sh
# apt-get install curl --yes && curl -s https://gist.githubusercontent.com/lilithium-hydride/84b916ea9dece94e3458595fc4573077/raw/76a0769775403c6be4f42a7b38e0b1784c354fdb/windscribebuild.sh | bash

apt-get update --yes
apt-get install build-essential git curl patchelf ruby-dev rpm libpam0g-dev golang-go autoconf libtool cmake fakeroot sudo python2 --yes 
yes | gem i fpm -f
apt-get install libfontconfig1-dev libfreetype6-dev libx11-dev libx11-xcb-dev libxext-dev libxfixes-dev libxi-dev libxrender-dev libxcb1-dev libxcb-glx0-dev libxcb-keysyms1-dev libxcb-image0-dev libxcb-shm0-dev libxcb-icccm4-dev libxcb-sync0-dev libxcb-xfixes0-dev libxcb-shape0-dev libxcb-randr0-dev libxcb-render-util0-dev libxkbcommon-dev libxkbcommon-x11-dev --yes
mkdir src && cd src
git clone --depth 1 https://github.com/Windscribe/desktop-v2 && cd desktop-v2
ln -s /usr/bin/python2 /usr/bin/python
python tools/bin/get-pip.py
python -m pip install -r tools/requirements.txt
cd tools/deps
./install_openssl
./install_qt
./install_cares
./install_boost
./install_curl
./install_lzo
./install_openvpn
./install_wireguard
./install_stunnel
./install_protobuf
./install_gtest
cd .. && ./build_all --no-sign