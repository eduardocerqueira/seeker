#date: 2022-07-14T17:01:00Z
#url: https://api.github.com/gists/c098da27877753909301e0b340676dbe
#owner: https://api.github.com/users/odenwal

#!/bin/bash

# copied from 
# https://gist.github.com/sj-dan

URL=$(curl "https://dl.google.com/dl/edgedl/chromeos/recovery/cloudready_recovery2.json" \
-s --output - | \
grep "^.*\"url\".*$" | \
sed "s/.*\"url\": \"\(.*\)\".*$/\1/g")

printf "\nURL for latest CloudReady image is $URL\n"

printf "\nDownloading now...\n\n"

curl -O $URL

printf "\nFinished downloading!\n"