#date: 2022-04-12T16:56:13Z
#url: https://api.github.com/gists/3441aeb99e73c7e8d53875209bc96f6a
#owner: https://api.github.com/users/LinauxTerminology

If you need openssL Setup contact with me: 
Email: urmirohman218@gmail.com
Skype: https://join.skype.com/
Telegram:https://t.me/LinauxTerminology
WhatsApp: +8801408694088 
Imo: +8801408694088

sudo apt update
sudo apt upgrade
sudo apt install build-essential checkinstall zlib1g-dev

#Step 2. Installing OpenSSL on Ubuntu 20.04.

#Now we download the source code OpenSSL from the official page:

cd /usr/local/src/
wget https://www.openssl.org/source/openssl-1.1.1k.tar.gz
sudo tar -xf openssl-1.1.1k.tar.gz
cd openssl-1.1.1k

#Then, we configure and compile OpenSSL:

sudo ./config --prefix=/usr/local/ssl --openssldir=/usr/local/ssl shared zlib
sudo make
sudo make test
sudo make install