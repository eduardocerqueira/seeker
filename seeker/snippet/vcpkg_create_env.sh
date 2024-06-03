#date: 2024-06-03T16:50:08Z
#url: https://api.github.com/gists/429f78a95bc4680169a0c732aa15d779
#owner: https://api.github.com/users/SaulBerrenson

#!/bin/bash
echo "set defaut triplet as x64-linux-dynamic"
export VCPKG_DEFAULT_TRIPLET=x64-linux-dynamic
export VCPKG_BINARY_SOURCES="clear;files,${PWD}/cache,readwrite"
echo "install all dependencies"
vcpkg install --clean-after-build
echo "export all installed dependencies to $PWD/env/unix_x64"
vcpkg export --x-all-installed --raw --output-dir=$PWD/env --output=unix_x64
echo "clean installed dependencies"
sudo rm -R $PWD/vcpkg_installed

echo "using -DCMAKE_TOOLCHAIN_FILE=/home/kali/tmp/grids/env_creator/env/astra17_x64/scripts/buildsystems/vcpkg.cmake -DVCPKG_TARGET_TRIPLET=x64-linux-dynamic"