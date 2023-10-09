#date: 2023-10-09T16:58:01Z
#url: https://api.github.com/gists/4a69bda9529dcc51713bad38be7e5668
#owner: https://api.github.com/users/mattkozlowski

from Crypto.Cipher import AES
from Crypto.Util.Padding import pad, unpad
from Crypto.Random import new, get_random_bytes
import hashlib

msg = 'Hello'
key = get_random_bytes(32)
print('KEY = ', key.hex())

cipher = AES.new(key, AES.MODE_ECB)
data = pad(msg.encode(), cipher.block_size)

enc = cipher.encrypt(data)
print(enc.hex())
print(enc)

decrypted = cipher.decrypt(enc)
decrypted_unpadded = unpad(decrypted, cipher.block_size)
print(decrypted_unpadded.decode())
