#date: 2022-03-09T16:54:32Z
#url: https://api.github.com/gists/e834f4389cc42470eaf8e52df5365dcc
#owner: https://api.github.com/users/HollowMan6

#!/bin/env python
import sys
import os
import hashlib

# BUF_SIZE is totally arbitrary, change for your app!
BUF_SIZE = 65536  # lets read stuff in 64kb chunks!

blake2b = hashlib.blake2b()
sha512 = hashlib.sha512()

with open(sys.argv[1], 'rb') as f:
    while True:
        data = f.read(BUF_SIZE)
        if not data:
            break
        blake2b.update(data)
        sha512.update(data)

print("{0} {1} BLAKE2B {2} SHA512 {3}".format(
    os.path.basename(sys.argv[1]),
    os.path.getsize(sys.argv[1]),
    blake2b.hexdigest(),
    sha512.hexdigest()))