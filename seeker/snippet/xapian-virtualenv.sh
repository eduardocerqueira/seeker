#date: 2021-09-29T16:46:42Z
#url: https://api.github.com/gists/3b9f22c9ab0db006f9e34c3478d3c751
#owner: https://api.github.com/users/forktheweb

#!/usr/bin/env bash


pkgver=1.2.21
mkdir -p $VIRTUAL_ENV/src && cd $VIRTUAL_ENV/src

curl -O http://oligarchy.co.uk/xapian/$pkgver/xapian-core-$pkgver.tar.xz && tar xf xapian-core-$pkgver.tar.xz
curl -O http://oligarchy.co.uk/xapian/$pkgver/xapian-bindings-$pkgver.tar.xz && tar xf xapian-bindings-$pkgver.tar.xz

cd $VIRTUAL_ENV/src/xapian-core-$pkgver
./configure --prefix=$VIRTUAL_ENV && make -j8 && make install

export LD_LIBRARY_PATH=$VIRTUAL_ENV/lib

cd $VIRTUAL_ENV/src/xapian-bindings-$pkgver
./configure --prefix=$VIRTUAL_ENV --with-python PYTHON_LIB=$VIRTUAL_ENV/lib && make -j8 && make install
