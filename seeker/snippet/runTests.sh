#date: 2023-02-15T16:48:37Z
#url: https://api.github.com/gists/51f9f06a7cef474a3f8bbe2aed20af9c
#owner: https://api.github.com/users/alexreinking

#!/bin/bash

set -eux -o pipefail

rm -rf build _local usepkg

cmake -G Ninja -S . -B build/main -DCMAKE_BUILD_TYPE=Release
cmake --build build/main
cmake --install build/main --prefix _local

mkdir usepkg
cp usepkg.cmake usepkg/CMakeLists.txt

cmake -G Ninja -S usepkg -B build/usepkg -DCMAKE_PREFIX_PATH=$PWD/build/main --fresh
cmake -G Ninja -S usepkg -B build/usepkg -DCMAKE_PREFIX_PATH=$PWD/_local --fresh