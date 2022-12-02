#date: 2022-12-02T17:10:57Z
#url: https://api.github.com/gists/1a7b2deb9dec76fc4dab12611c59c223
#owner: https://api.github.com/users/arkjo

apt-get install -y wget build-essential gettext autoconf automake libtool

wget http://download.mono-project.com/sources/mono/mono-3.2.3.tar.bz2
bunzip2 -df mono-3.2.3.tar.bz2
tar -xf mono-3.2.3.tar
cd mono-3.2.3
./configure --prefix=/usr/local; make; make install

rm -rf /tmp/*
apt-get remove --purge wget build-essential gettext autoconf automake libtool