#date: 2022-07-29T17:08:03Z
#url: https://api.github.com/gists/550d382166f33398cea4b381a77a6ef6
#owner: https://api.github.com/users/danomatika

#! /bin/sh
#
# download and build nn~ external for puredata on macOS:
# https://github.com/acids-ircam/nn_tilde
#
# notes:
# * binary external is placed in root dir
# * there doesn't appear to be a help file
# * on M1 machines. Pd has to be run in Rosetta using x86_64 arch, ie.
#   arch -X86_64 /Applications/Pd-0.52-2.app/Contents/Resources/bin/pd
#
# Dan Wilcox danomatika.com 2022

# stop on error
set -e

##### variables

TORCH_VER=1.12.0
PD_APP=/Applications/Pd-0.52-2.app

# official libtorch builds appear to be x86_64 only for now :(
# check https://pytorch.org/get-started/locally/ for info
ARCH="x86_64"

##### main

# clone repo
git clone https://github.com/acids-ircam/nn_tilde --recursive
cd nn_tilde

# download libtorch
mkdir -p lib
cd lib
rm -rf libtorch
curl -LO https://download.pytorch.org/libtorch/cpu/libtorch-macos-${TORCH_VER}.zip
unzip libtorch-macos-${TORCH_VER}.zip
rm -rf libtorch-macos-${TORCH_VER}.zip
cd ../

# setup build
mkdir -p build
cd build
export Torch_DIR=../lib/libtorch/share/cmake/Torch
cmake ../src/ -DPUREDATA_INCLUDE_DIR="${PD_APP}/Contents/Resources/src" \
              -DCMAKE_BUILD_TYPE=Release \
              -DCMAKE_OSX_ARCHITECTURES=${ARCH}

# build
make
cp frontend/puredata/nn_tilde/nn~.pd_darwin ../
cd ../

# add stub helpfile for testing
if [ ! -e "nn~-help.pd" ] ; then
    echo "#N canvas 538 97 450 300 12;" > "nn~-help.pd"
    echo "#X obj 155 105 nn~;"         >> "nn~-help.pd"
    echo ""                            >> "nn~-help.pd"
fi
