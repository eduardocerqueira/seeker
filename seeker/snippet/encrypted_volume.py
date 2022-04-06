#date: 2022-04-06T16:58:35Z
#url: https://api.github.com/gists/05997f8f2f0029062e59502bf6cd1140
#owner: https://api.github.com/users/rgrizzell

import aesio
import adafruit_hashlib as hashlib
from binascii import hexlify, unhexlify
import json
import os


class EncryptedVolume:
    def __init__(self, path, key, mode=aesio.MODE_CTR):
        m = hashlib.sha256()
        m.update(key)
        
        self._path = path
        self._iv = os.urandom(16)
        self._mode = mode
        self._key = m.digest()

    def decrypt_volume(self):
        cipher = aesio.AES(self._key, 6, self._iv)
        with open(self._path, 'rb') as volume:
            data_in = volume.read()
            data = bytearray(len(data_in))
            cipher.decrypt_into(data_in, data)
            return data.decode('utf-8')

    def encrypt_volume(self, data):
        if isinstance(data, str):
            data = bytes(data, 'utf-8')

        cipher = aesio.AES(self._key, 6, self._iv)
        data_out = bytearray(len(data))
        with open(self._path, 'wb') as volume:
            cipher.encrypt_into(data, data_out)
            volume.write(data_out)

        return data_out


vol = EncryptedVolume("/test.vol", "password1")

test_data = {"foo": "bar"}

print(test_data)
print(hexlify(vol.encrypt_volume(json.dumps(test_data))))
print(vol.decrypt_volume())
