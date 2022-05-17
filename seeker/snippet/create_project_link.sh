#date: 2022-05-17T16:56:02Z
#url: https://api.github.com/gists/3fb35f7180471d9fe70aa59e18b2b023
#owner: https://api.github.com/users/bfagundes-ks

#!/bin/bash

# KS: Creating link for project folder
# Version: 220517A@bfagundes

SOURCE="$(pwd)/2022"
DEST="$(cd ..;pwd)/2022"

ln -s $SOURCE $DEST