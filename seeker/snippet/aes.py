#date: 2023-04-06T17:05:46Z
#url: https://api.github.com/gists/0443668071be15c6dc9fa95f702e3120
#owner: https://api.github.com/users/jdgoettsch

import base64

from Crypto.Cipher import AES
from Crypto import Random


class Cipher(object):

    def __init__(self, key):
        self.key = Cipher._pad(key[:32])

    def decrypt(self, enc_msg):
        enc_msg = base64.b64decode(enc_msg)
        iv = enc_msg[:AES.block_size]
        cipher = AES.new(self.key, AES.MODE_CBC, iv)
        decrypted = self._unpad(cipher.decrypt(enc_msg[AES.block_size:]))
        return decrypted.decode('utf-8')

    def encrypt(self, msg):
        msg = Cipher._pad(msg)
        iv = Random.new().read(AES.block_size)
        cipher = AES.new(self.key, AES.MODE_CBC, iv)
        return base64.b64encode(b''.join([iv, cipher.encrypt(msg)]))

    @staticmethod
    def _pad(s_):
        bs = 32
        if len(s_) == bs:
            return s_
        return ''.join(
            [s_, (bs - len(s_) % bs) * chr(bs - len(s_) % bs)])

    @staticmethod
    def _unpad(s_):
        return s_[:-ord(s_[len(s_)-1:])]


if __name__ == "__main__":
    pass