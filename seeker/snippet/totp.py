#date: 2024-08-21T16:53:02Z
#url: https://api.github.com/gists/8dfab5e8e1076eb7e32fe0b5ff3d87fd
#owner: https://api.github.com/users/Allain18

from hashlib import sha1
from time import time
import base64


key = "XXKEYBASE32XX"

if len(key) % 8:
    key += "=" * (8 - (len(key) % 8))
key = base64.b32decode(key)
key += (0).to_bytes(64 - len(key))

text = int(time() // 30).to_bytes(8)

ipad = (0x36).to_bytes(1)*64
opad = (0x5c).to_bytes(1)*64
k_ipad = bytearray(a ^ b for (a, b) in zip(key, ipad))
k_opad = bytearray(a ^ b for (a, b) in zip(key, opad))
hash = sha1(k_opad + sha1(k_ipad + text).digest()).digest()

offset = hash[19] & 0x0F
binary = int.from_bytes(hash[offset:offset+4]) & 0x7FFFFFFF
print(str(binary)[-6:])
