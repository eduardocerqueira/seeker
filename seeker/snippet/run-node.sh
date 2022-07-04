#date: 2022-07-04T02:48:04Z
#url: https://api.github.com/gists/099ad333618b7ded6d2b4719da122d55
#owner: https://api.github.com/users/sysuyl

#!/bin/bash

# Make sure node version is greater than 15.1.0
ver="$(node --version)"
ver=${ver:1} # remove the prefix 'v'
major="$(cut -d '.' -f 1 <<< "$ver")"
minor="$(cut -d '.' -f 2 <<< "$ver")"

if [ $major -ge 15 ] && [ $minor -ge 1 ]; then
    node --experimental-wasm-threads script.js
else
    echo "Cannot run node. Require version 15.1.0 or greater"
fi
