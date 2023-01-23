#date: 2023-01-23T17:03:56Z
#url: https://api.github.com/gists/01b062e9980e5c2062cfcc763a8268f6
#owner: https://api.github.com/users/georgiyordanov

#!/usr/bin/env bash

sudo sed -i '' -e '/^#/a\'$'\n''auth       sufficient     pam_tid.so' /etc/pam.d/sudo