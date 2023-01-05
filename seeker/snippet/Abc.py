#date: 2023-01-05T16:55:39Z
#url: https://api.github.com/gists/c6237d63fb2b2bd86fce54983ebb7cbb
#owner: https://api.github.com/users/oMegaPB

import base64
import hashlib
import random
import typing as t


class Abc:
    """Basic Encryption support."""
    def __init__(
        self, key: str,
        keyfunc: t.Callable[[str], str] = lambda x: hashlib.sha1(x.encode()).hexdigest() # needed for unicode support
    ) -> None:
        self.__key = keyfunc(key)
    
    def decrypt(self, message: str) -> str:
        if isinstance(message, str) and not message:
            return ""
        data = [list(map(ord, base64.b64decode(message).decode()[1:])), list(map(ord, self.key))]
        for x in data[0]:
            data[1] = [data[1][0:len(data[0])], data[1] + data[1][::-1]][len(data[1]) < len(data[0])]
        enc_data = [abs(data[0][x] - 27 * data[1][x]) for x, _ in enumerate(data[0])]
        for x, y in enumerate(enc_data):
            enc_data[x] = y - (ord(base64.b64decode(message).decode()[0]) - x)
        return "".join(map(lambda x: chr(abs(x)), enc_data))
    
    def __repr__(self) -> str:
        return f"<BasicDecryptor object at {hex(id(self))}>"
    
    @property
    def key(self) -> str:
        return self.__key
    
    def encrypt(self, message: str) -> str:
        if isinstance(message, str) and not message:
            return ""
        m = random.randint(0, 255)
        data = [list(map(ord, message)), list(map(ord, self.key))]
        for x, y in enumerate(data[0]):
            data[0][x] = y + (m - x)
        for x in data[0]:
            data[1] = [data[1][0:len(data[0])], data[1] + data[1][::-1]][len(data[1]) < len(data[0])]
        encdata = "".join(map(chr, [data[0][x] + data[1][x] * 27 for x, _ in enumerate(data[0])]))
        return base64.b64encode(chr(m).encode() + encdata.encode()).decode()

s = Abc("12345")
enc = s.encrypt("hello world!")
print(enc) # in most cases encrypted data will be 3.x or 4.x times bigger than original data

dec = s.decrypt(enc)
print(dec) 