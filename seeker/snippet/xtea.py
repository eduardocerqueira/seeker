#date: 2021-12-30T17:02:57Z
#url: https://api.github.com/gists/39a8a0d73995e512c1b44495b1964f37
#owner: https://api.github.com/users/LeKSuS-04

from ctypes import c_uint32

def pad(data: bytes, size: int):
    data += ((size - len(data)) % size) * b'\x00'
    return data

def unpad(data: bytes):
    data = data.strip(b'\x00')
    return data


def bytes_to_c_uint32(bts: bytes) -> list[c_uint32]:
    assert len(bts) % 4 == 0
    out = []
    for i in range(0, len(bts), 4):
        n = int.from_bytes(bts[i:i+4], byteorder='little')
        out.append(c_uint32(n))
    return out

def c_uint32_to_bytes(uints: list[c_uint32]) -> bytes:
    out = b''
    for u in uints:
        out += u.value.to_bytes(4, byteorder='little')
    return out


def xtea_encrypt(rounds: int, v: list[c_uint32], key: list[c_uint32]):
    v0, v1 = v
    sum_ = c_uint32(0)
    delta = 0x9E3779B9
    for _ in range(rounds):
        v0.value += (((v1.value << 4) ^ (v1.value >> 5)) + v1.value) ^ (sum_.value + key[sum_.value & 3])
        sum_.value += delta
        v1.value += (((v0.value << 4) ^ (v0.value >> 5)) + v0.value) ^ (sum_.value + key[(sum_.value>>11) & 3])
    return [v0, v1]

def xtea_decrypt(rounds: int, v: list[c_uint32], key: list[c_uint32]):
    v0, v1 = v
    delta = 0x9E3779B9
    sum_ = c_uint32(delta * rounds)
    for _ in range(rounds):
        v1.value -= (((v0.value << 4) ^ (v0.value >> 5)) + v0.value) ^ (sum_.value + key[(sum_.value>>11) & 3])
        sum_.value -= delta
        v0.value -= (((v1.value << 4) ^ (v1.value >> 5)) + v1.value) ^ (sum_.value + key[sum_.value & 3])
    return v0, v1


def decrypt(rounds: int, data: bytes, key: bytes) -> bytes:
    assert len(key) == 16, 'Size of key must be 16 bytes'
    plain = b''

    for i in range(0, len(data), 8):
        v = bytes_to_c_uint32(data[i:i+8])
        v = xtea_decrypt(rounds, v, key)
        plain += c_uint32_to_bytes(v)

    return unpad(plain)
    

def encrypt(rounds: int, data: bytes, key: bytes) -> bytes:
    assert len(key) == 16, 'Size of key must be 16 bytes'

    data = pad(data, 8)
    cipher = b''

    for i in range(0, len(data), 8):
        v = bytes_to_c_uint32(data[i:i+8])
        v = xtea_encrypt(rounds, v, key)
        cipher += c_uint32_to_bytes(v)

    return cipher
