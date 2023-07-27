#date: 2023-07-27T16:47:11Z
#url: https://api.github.com/gists/51d73c261542fb605b57d525ecd85438
#owner: https://api.github.com/users/markizano

#!/bin/bash -e

XMR_MINE_SCRIPT=$'#!/bin/bash

test "$UID" -eq 0 && {
  ulimit -l 10240
  modprobe msr
  chgrp msr /dev/cpu/*/msr
  chmod g+rw /dev/cpu/*/msr
}

RUNAS_USER=monero
RUNAS_GROUP=apps
test "$UID" -eq `id -u $RUNAS_USER` || {
  echo "Dropping privileges..."
  exec sudo -H -u$RUNAS_USER -g$RUNAS_GROUP MINER_DEBUG=${MINER_DEBUG} MINER_POOL=${MINER_POOL} MINER_WALLET=${MINER_WALLET} MINER_NAME=${MINER_NAME} MINER_HOME=${MINER_HOME} "$0" $@
}

test -z "$MINER_DEBUG" || {
  set -x
}

MINER_POOL=${MINER_POOL:-"monerod.markizano.net:3000"}
MINER_WALLET=${MINER_WALLET:-"insert-wallet-address"}
MINER_NAME=${MINER_NAME:-`hostname -s`}
MINER_HOME=${MINER_HOME:-"monerod.markizano.net:18081"}

cd

exec xmrig --threads=`nproc` \
  -o "${MINER_POOL}" \
  -u "${MINER_WALLET}" \
  -p "${MINER_NAME}" \
  -k \
  --tls \
  -o "${MINER_HOME}" \
  -u "${MINER_WALLET}" \
  --tls \
  --coin monero $@
'

# Setup some variables and build the xmrig software.
version=6.12.0
xmrig=~/git/cryptocurrency/xmrig
cuda=~/git/cryptocurrency/xmrig-cuda

sudo apt-get install -y git build-essential cmake libuv1-dev libssl-dev libhwloc-dev msr-tools

# https://stackoverflow.com/questions/18661976/reading-dev-cpu-msr-from-userspace-operation-not-permitted
# Found out why I get the MSR error. This is the fix.
modprobe msr || true
grep msr /etc/group || sudo groupadd --system msr
sudo chmod g+rw /dev/cpu/*/msr
sudo chgrp msr /dev/cpu/*/msr

test -d $xmrig || git clone --depth 1 --branch v$version git@github.com:xmrig/xmrig $xmrig
test -d $xmrig/build || mkdir $xmrig/build

test -z "$WITH_CUDA" || {
  test -d $cuda  || git clone --depth 1 --branch v$version git@github.com:xmrig/xmrig-cuda $cuda
  test -d $cuda/build  || mkdir $cuda/build
}

test -d /usr/local/share/xmrig-$version || sudo install -d -oroot -gstaff -m2755 /usr/local/share/xmrig-$version

(
  cd $xmrig/build
  cmake ..
  make -j`nproc`
  sudo install -v -oroot -gstaff ./xmrig /usr/local/share/xmrig-$version/
  sudo setcap cap_sys_rawio=ep /usr/local/share/xmrig-$version/xmrig
)

test -z "$WITH_CUDA" || {(
  cd $cuda/build
  # If you have issues with cuda: -DCUDA_LIB=/usr/local/cuda/targets/x86_64-linux/lib/stubs/libcuda.so -DCUDA_TOOLKIT_ROOT_DIR=/usr/local/cuda
  cmake .. -DCUDA_ARCH=61 #-DCMAKE_C_COMPILER=gcc-9 -DCMAKE_CXX_COMPILER=g++-9
  make -j`nproc`
  sudo install -v -oroot -gstaff ./libxmrig-cuda.so /usr/local/share/xmrig-$version/
  slaves="--slave /usr/lib/libxmrig-cuda.so libxmrig-cuda.so /usr/local/share/xmrig-$version/libxmrig-cuda.so"
)}

# Make the software live and available to the rest of the system thru update-alternatives.
sudo update-alternatives --install /usr/local/bin/xmrig xmrig /usr/local/share/xmrig-$version/xmrig 100 $slaves

# Enable hugepage support.
sudo sysctl -w vm.nr_hugepages=1

# setup the non-privileged account that will enable us to mine!
id monero || sudo useradd --gid=200 --shell /usr/sbin/nologin --system --home /var/lib/monero -m -G msr monero
test -d ~monero/bin || sudo install -d -omonero -gapps /var/lib/monero/bin
test -f /var/lib/monero/bin/mine.sh || echo "$XMR_MINE_SCRIPT" | sudo install -omonero -gapps /dev/stdin /var/lib/monero/bin/mine.sh
