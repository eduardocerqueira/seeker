#date: 2022-04-14T16:49:46Z
#url: https://api.github.com/gists/cc7f72469237c38851910b1906cff994
#owner: https://api.github.com/users/SuperSaiyanYosh

# Mac OS X has some outdated build dependencies.
# This script will bring you up-to-date, then install ttfautohint.
# Only tested under 10.7.1, but should work
# You'll obviously need Developer Tools installed first.
mkdir ~/tmp
cd ~/tmp
curl -O http://mirrors.kernel.org/gnu/m4/m4-1.4.16.tar.gz
tar -xzvf m4-1.4.16.tar.gz
cd m4-1.4.16
./configure --prefix=/usr/local
make
sudo make install
cd ..

curl -O http://mirrors.kernel.org/gnu/autoconf/autoconf-2.68.tar.gz
tar -xzvf autoconf-2.68.tar.gz
cd autoconf-2.68
./configure --prefix=/usr/local # ironic, isn't it?
make
sudo make install
cd ..

# here you might want to restart your terminal session, to ensure the new autoconf is picked up and used in the rest of the script
curl -O http://mirrors.kernel.org/gnu/automake/automake-1.11.tar.gz
tar xzvf automake-1.11.tar.gz
cd automake-1.11
./configure --prefix=/usr/local
make
sudo make install
cd ..

curl -O http://mirrors.kernel.org/gnu/libtool/libtool-2.4.tar.gz
tar xzvf libtool-2.4.tar.gz
cd libtool-2.4
./configure --prefix=/usr/local
make
sudo make install
cd ..

curl -O http://sourceforge.net/projects/freetype/files/freetype2/2.4.5/freetype-2.4.5.tar.gz
tar zxvf freetype-2.4.5.tar.gz
cd freetype-2.4.5
./configure
make
sudo make install
cd ..

curl -O http://sourceforge.net/projects/freetype/files/ttfautohint/0.2/ttfautohint-0.2.tar.gz
tar zxvf ttfautohint-0.2.tar.gz
cd ttfautohint-0.2
./configure
make
sudo make install
