#date: 2024-12-23T17:10:42Z
#url: https://api.github.com/gists/ac48a069e9a5a2c3d8c04bc89dc6b912
#owner: https://api.github.com/users/john-clark

#!/bin/bash
# install 3.8 - latest vice
sudo apt-get remove vice
rm -rf /usr/share/vice/
wget https://psychz.dl.sourceforge.net/project/vice-emu/releases/binaries/debian/gtk3vice_3.8.deb
wget https://security.debian.org/debian-security/pool/updates/main/f/flac/libflac8_1.3.2-3+deb10u3_amd64.deb
sudo apt install libportaudio2 libgif7 libpcap0.8
sudo dpkg -i libflac8_1.3.2-3+deb10u3_amd64.deb gtk3vice_3.8.deb
exit
# install 3.7 - debian official and roms
sudo sed -i 's/main/main contrib/' /etc/apt/sources.list
sudo apt install vice
wget https://psychz.dl.sourceforge.net/project/vice-emu/releases/binaries/windows/GTK3VICE-3.8-win64.zip
unzip GTK3VICE-3.8-win64.zip
cd vice-3.8/
sudo cp GTK3VICE-3.8-win64/C64/*.bin /usr/share/vice/C64/
sudo cp GTK3VICE-3.8-win64/C128/*.bin /usr/share/vice/C128/
sudo cp GTK3VICE-3.8-win64/C128/kernal* /usr/share/vice/C128/
sudo cp GTK3VICE-3.8-win64/PET/*.bin /usr/share/vice/PET/
sudo cp GTK3VICE-3.8-win64/PLUS4/*.bin /usr/share/vice/PLUS4/
sudo cp GTK3VICE-3.8-win64/DRIVES/*.bin /usr/share/vice/DRIVES/
sudo cp GTK3VICE-3.8-win64/VIC20/*.bin /usr/share/vice/VIC20/
sudo cp GTK3VICE-3.8-win64/SCPU64/*.bin /usr/share/vice/SCPU64/
sudo cp GTK3VICE-3.8-win64/C64DTV/*.bin /usr/share/vice/C64DTV/
sudo cp GTK3VICE-3.8-win64/CBM-II/*.bin /usr/share/vice/CBM-II/
