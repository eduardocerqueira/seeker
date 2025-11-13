#date: 2025-11-13T16:59:57Z
#url: https://api.github.com/gists/fcd988d0253c51104e8bbe050e057891
#owner: https://api.github.com/users/mykulyak

#!/usr/bin/env bash

TMP_DIR=$(mktemp -d)
KEYCHAIN_PASSWORD= "**********"
CERTIFICATE_PASSWORD= "**********"
KEYCHAIN_PATH=$TMP_DIR/keychain.keychain
CERTIFICATE_PATH=$TMP_DIR/certificate.p12

echo "$BASE_64_CERTIFICATE" | base64 --decode > $CERTIFICATE_PATH
security create-keychain -p $KEYCHAIN_PASSWORD $KEYCHAIN_PATH
security set-keychain-settings -lut 21600 $KEYCHAIN_PATH
security default-keychain -s $KEYCHAIN_PATH
security unlock-keychain -p $KEYCHAIN_PASSWORD $KEYCHAIN_PATH
security import $CERTIFICATE_PATH -P $CERTIFICATE_PASSWORD -A