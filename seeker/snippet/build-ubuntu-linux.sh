#date: 2021-08-31T00:51:40Z
#url: https://api.github.com/gists/d9c4d93a596827a53113ef830f0f0916
#owner: https://api.github.com/users/mattmcgiv

# update dependency lists
sudo apt-get update

# get repo
git clone https://github.com/bitcoin/bitcoin/ && cd bitcoin

# install build dependencies
sudo apt-get install build-essential libtool autotools-dev automake pkg-config bsdmainutils python3

# more dependencies
sudo apt-get install libevent-dev libboost-dev libboost-system-dev libboost-filesystem-dev libboost-test-dev

# build berkeleydb
./contrib/install_db4.sh `pwd`

# to build
./autogen.sh

#When compiling bitcoind, run `./configure` in the following way for BDB support
export BDB_PREFIX='/home/ubuntu/bitcoin/db4'
./configure BDB_LIBS="-L${BDB_PREFIX}/lib -ldb_cxx-4.8" BDB_CFLAGS="-I${BDB_PREFIX}/include"

make # use "-j N" for N parallel jobs
make install # optional
