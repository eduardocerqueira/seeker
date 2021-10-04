#date: 2021-10-04T17:15:43Z
#url: https://api.github.com/gists/2d9709b3ea782cf0620d8fdbfe55fb5b
#owner: https://api.github.com/users/Ghostbird

#!/bin/bash
dd if=/dev/random bs=1 count=${1:-16} 2> /dev/null | base64 | tee >(xclip -selection clip-board);
