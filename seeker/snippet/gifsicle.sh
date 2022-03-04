#date: 2022-03-04T16:50:35Z
#url: https://api.github.com/gists/f957874fdbd65a02a032dfabd1c664c8
#owner: https://api.github.com/users/cquintini

#!/bin/bash

if [ "`/usr/bin/whoami`" != "root" ]; then
    echo "You need to execute this script as root."
    exit 1
fi

yum install -y autoconf automake bzip2 bzip2-devel cmake freetype-devel gcc gcc-c++ git libtool make mercurial pkgconfig zlib-devel

cd /usr/local/src/

#install
git clone --branch master --depth 1 https://github.com/kohler/gifsicle.git
cd gifsicle
autoreconf -i
./configure
make install
