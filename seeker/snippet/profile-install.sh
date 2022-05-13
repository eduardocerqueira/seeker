#date: 2022-05-13T16:56:15Z
#url: https://api.github.com/gists/c4fd62a1bb621595bd58fefa99ee9739
#owner: https://api.github.com/users/chinloyal

#!/bin/bash

base64 -d <<< $PROVISIONING_PROFILE > profile.mobileprovision
UUID=`grep UUID -A1 -a profile.mobileprovision | grep -io "[-A-F0-9]\{36\}"`
cp profile.mobileprovision ~/Library/MobileDevice/Provisioning\ Profiles/$UUID.mobileprovision