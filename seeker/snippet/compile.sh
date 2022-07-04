#date: 2022-07-04T02:48:04Z
#url: https://api.github.com/gists/099ad333618b7ded6d2b4719da122d55
#owner: https://api.github.com/users/sysuyl

#!/bin/bash

emcc -std=c++17 \
  -O3 --bind \
  -s SINGLE_FILE=1 \
  -s PTHREAD_POOL_SIZE=4 \
  -s USE_PTHREADS=1 \
  -s ASSERTIONS=1 \
  -s EXPORT_NAME=LibModule \
  -s WASM=1 \
  -s MODULARIZE=1 \
  -s ALLOW_MEMORY_GROWTH=1 \
  -o lib.js \
  main.cxx
