#date: 2022-04-25T16:54:13Z
#url: https://api.github.com/gists/de577a1518b78c8fb12540caab2973ee
#owner: https://api.github.com/users/kuikka

#!/usr/bin/env phthon3

# rfc6287

import os
import hashlib
import hmac

K = b'mysecret'

m = hashlib.new('sha1')

#1 Device sends challenge
DC = os.urandom(16)

#2 Phone computes PC (Phone Challenge)
PC = os.urandom(16)

#3 Phone computes H(K | DC | PC)
m.update(K + DC + PC)

#4 Phone sends response and PC to device
PR = m.digest()
print(PR)

#5 Device computes H(K | DC | PC) and compares to PR
m = hashlib.new('sha1')
m.update(K + DC + PC)
DR = m.digest()
print(DR)



# HMAC
print("HMAC")
h = hmac.digest(K, DC + DC, 'sha256')
print(h)
