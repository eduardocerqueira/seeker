#date: 2023-09-26T16:56:25Z
#url: https://api.github.com/gists/6faa1ea74c91b32e805908a3deceb3d8
#owner: https://api.github.com/users/duchs

#!/bin/bash

set -e

file=$(defaults read /Applications/Navicat\ Premium.app/Contents/Info.plist)

regex="CFBundleShortVersionString = \"([^\.]+)"
[[ $file =~ $regex ]]

version=${BASH_REMATCH[1]}

echo "Detected Navicat Premium version $version"

case $version in
    "16")
        file=~/Library/Preferences/com.navicat.NavicatPremium.plist
        ;;
    "15")
        file=~/Library/Preferences/com.prect.NavicatPremium15.plist
        ;;
    *)
        echo "Version '$version' not handled"
        exit 1
       ;;
esac

echo -n "Reseting trial time..."

regex="([0-9A-Z]{32}) = "
[[ $(defaults read $file) =~ $regex ]]

hash=${BASH_REMATCH[1]}

if [ ! -z $hash ]; then
    defaults delete $file $hash
fi

regex="\.([0-9A-Z]{32})"
[[ $(ls -a ~/Library/Application\ Support/PremiumSoft\ CyberTech/Navicat\ CC/Navicat\ Premium/ | grep '^\.') =~ $regex ]]

hash2=${BASH_REMATCH[1]}

if [ ! -z $hash2 ]; then
    rm ~/Library/Application\ Support/PremiumSoft\ CyberTech/Navicat\ CC/Navicat\ Premium/.$hash2
fi

echo " Done"
