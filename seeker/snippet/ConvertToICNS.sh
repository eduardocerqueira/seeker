#date: 2021-10-12T17:04:32Z
#url: https://api.github.com/gists/fc18150a3a9e97f4972880779a16d0fe
#owner: https://api.github.com/users/Le0xFF

#!/bin/bash

# Source: https://apple.stackexchange.com/questions/388246/efi-partition-change-name-and-logo/388287#388287
# Usage in macOS: ./ConvertToICNS.sh ImageToConvert.png

if [[ $0 != "$BASH_SOURCE" ]]; then
    bash "$BASH_SOURCE" "$@"
    return
fi

Convert() {
    local -i "n=${1%%x*}"
    [[ $1 != *@2x ]] || n="2*n"
    sips -s format png -z "$n" "$n" "$full" --out "$name.iconset/icon_$1.png"
}

Main() {
    local "full=$1"
    local "base=$(basename "$full")"
    local "name=${base%.*}"
    rm -rf "$name.iconset"
    mkdir "$name.iconset"
    Convert "16x16"
    Convert "16x16@2x"
    Convert "32x32"
    Convert "32x32@2x"
    Convert "128x128"
    Convert "128x128@2x"
    Convert "256x256"
    Convert "256x256@2x"
    Convert "512x512"
    Convert "512x512@2x"
    iconutil -c icns "$name.iconset"
    rm -r "$name.iconset"
}

set -u
Main "$@"