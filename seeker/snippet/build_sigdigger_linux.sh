#date: 2024-05-23T17:04:40Z
#url: https://api.github.com/gists/ab3a610b365ca57c61e5b9e6e4552409
#owner: https://api.github.com/users/sultanqasim

#!/bin/sh
#
set -eux

CPU_COUNT="$(nproc)"

# get the code
if [ ! -e sigutils ]; then
    git clone -b develop --recurse-submodules https://github.com/sultanqasim/sigutils.git
fi

if [ ! -e suscan ]; then
    git clone -b develop --recurse-submodules https://github.com/sultanqasim/suscan.git
fi

if [ ! -e SuWidgets ]; then
    git clone -b develop https://github.com/sultanqasim/SuWidgets
fi

if [ ! -e SigDigger ]; then
    git clone -b develop https://github.com/sultanqasim/SigDigger.git
fi

# prepare the deploy path
DEPLOYROOT="$(pwd)/deploy-root"
rm -rf "$DEPLOYROOT"
mkdir -p "$DEPLOYROOT"
export PKG_CONFIG_PATH="$DEPLOYROOT/usr/lib/pkgconfig:$DEPLOYROOT/usr/lib64/pkgconfig"

DEBUG=0
if [ "$DEBUG" == "1" ]; then
    CMAKE_BUILD_TYPE="Debug"
    QMAKE_BUILD_TYPE="debug"
else
    CMAKE_BUILD_TYPE="Release"
    QMAKE_BUILD_TYPE="release"
fi

# build the source dependencies
(
    cd sigutils
    rm -rf build
    mkdir -p build
    cd build
    cmake .. -DCMAKE_INSTALL_PREFIX="$DEPLOYROOT/usr" -DCMAKE_SKIP_RPATH=ON -DCMAKE_SKIP_INSTALL_RPATH=ON -DCMAKE_BUILD_TYPE="$CMAKE_BUILD_TYPE"
    make -j${CPU_COUNT}
    make install
)

(
    cd suscan
    rm -rf build
    mkdir -p build
    cd build
    cmake .. -DCMAKE_INSTALL_PREFIX="$DEPLOYROOT/usr" -DCMAKE_SKIP_RPATH=ON -DCMAKE_SKIP_INSTALL_RPATH=ON -DCMAKE_PREFIX_PATH="$DEPLOYROOT/usr" -DCMAKE_BUILD_TYPE="$CMAKE_BUILD_TYPE"
    make -j${CPU_COUNT}
    make install
)

(
    cd SuWidgets
    echo $PKG_CONFIG_PATH
    qmake SuWidgetsLib.pro PREFIX="$DEPLOYROOT/usr" "CONFIG += ${QMAKE_BUILD_TYPE}"
    make clean
    make -j${CPU_COUNT}
    make install
)

# build SigDigger
(
    cd SigDigger
    qmake SigDigger.pro PREFIX="$DEPLOYROOT/usr" SUWIDGETS_PREFIX="$DEPLOYROOT/usr" "CONFIG += ${QMAKE_BUILD_TYPE}"
    make clean
    make -j${CPU_COUNT}
    make install
)
