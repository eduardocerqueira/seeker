#date: 2023-11-23T16:39:42Z
#url: https://api.github.com/gists/de450129dafe52001f834b2686e495d1
#owner: https://api.github.com/users/osdeibi

#!/bin/bash

## Compile OpenSSL
OPENSSL=openssl-1.1.1h.tar.gz

DONE=openssl-compile-done

if [ ! -f "${DONE}" ] ;then
    wget https://www.openssl.org/source/${OPENSSL}

    tar zxvf ${OPENSSL}

    cd $(basename $OPENSSL .tar.gz)

    ./config shared no-idea no-md2 no-mdc2 no-rc5 no-rc4 --prefix=/usr/local/

    make

    sudo make install

    cd ..

    touch ${DONE}
fi

export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/usr/local/lib64/

read -n1 -r -p "$(/usr/local/bin/openssl version) - Press any key to continue..." key

source ./nginx-with-tls13-compile.sh

## copy systemctl config
cp nginx.service /lib/systemd/system/nginx.service
