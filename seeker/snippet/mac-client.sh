#date: 2022-03-07T17:11:39Z
#url: https://api.github.com/gists/1dd9cdbda9768511f3871c45b1feb348
#owner: https://api.github.com/users/MikhailKalikin

#!/usr/bin/env bash

# mount /home/joaomlneto @ linux-vm
mkdir -p /Users/joaomlneto/nfs/linux-vm
mount -t nfs -o nolocks,locallocks,rw,soft,intr,rsize=8192,wsize=8192,timeo=900,retrans=3,proto=tcp \
linux-vm:/home/joaomlneto \
/Users/joaomlneto/nfs/linux-vm