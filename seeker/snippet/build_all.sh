#date: 2025-07-15T17:05:47Z
#url: https://api.github.com/gists/d5e64823f69127e1345ee97e8152b236
#owner: https://api.github.com/users/kdmukai

#!/bin/bash

# Requires: apt install time
export APP_REPO = https://github.com/kdmukai/seedsigner.git
export APP_BRANCH = 0.8.6

touch results.txt
for device in pi0 pi02w pi2 pi4
do
    ccache -d /root/.buildroot-ccache --zero-stats
    ccache --zero-stats
    echo "Building for: $device" >> results.txt
    /usr/bin/time -o results.txt -a ./build.sh --$device --app-repo=$APP_REPO --app-branch=$APP_BRANCH
    echo "---------------------------------------------------" >> results.txt
    ccache -d /root/.buildroot-ccache --show-stats >> results.txt
    echo "---------------------------------------------------" >> results.txt
    ccache --show-stats >> results.txt
    echo "---------------------------------------------------" >> results.txt
    echo " " >> results.txt
done

cat results.txt
