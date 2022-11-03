#date: 2022-11-03T16:54:14Z
#url: https://api.github.com/gists/3346540e119076b1b296eb1a1adcb4c0
#owner: https://api.github.com/users/khanh-nguyen-code

import io
import os
import struct
from typing import BinaryIO

from Cryptodome.Cipher import AES
from Cryptodome.Hash import SHA256

CHUNK_SIZE = 65536  # chunk size to read from io in bytes
HASH_SIZE = 32  # size of hash value in bytes
BLOCK_SIZE = AES.block_size  # size of AES block in bytes
KEY_SIZE = 32  # size of AES key in bytes
UINT64_SIZE = 8  # size of uint64 in bytes
AES_MODE = AES.MODE_CBC  # cipher block chaining


def random_bytes(count: int = 1) -> bytes:
    return os.urandom(count)


def read_hex_file(path: str) -> bytes:
    return bytes.fromhex(open(path, "r").read())


def write_hex_file(path: str, b: bytes):
    open(path, "w").write(b.hex())


def uint64_to_bytes(i: int) -> bytes:
    return struct.pack("<Q", i)  # little endian, uint64


def bytes_to_uint64(b: bytes) -> int:
    return struct.unpack("<Q", b)[0]  # little endian, uint64


def sha256_hash(f_in: BinaryIO) -> bytes:
    h = SHA256.new()
    while True:
        b = f_in.read(CHUNK_SIZE)
        if len(b) == 0:
            b = h.digest()
            assert len(b) == HASH_SIZE
            return b
        h.update(b)


def aes256_encrypt(key: bytes, iv: bytes, f_in: BinaryIO, f_out: BinaryIO):
    assert BLOCK_SIZE == 16
    assert CHUNK_SIZE % BLOCK_SIZE == 0
    assert len(iv) == BLOCK_SIZE
    assert len(key) == KEY_SIZE
    aes = AES.new(key, AES_MODE, iv)
    while True:
        chunk = f_in.read(CHUNK_SIZE)
        if len(chunk) == 0:
            return
        if len(chunk) % BLOCK_SIZE != 0:
            chunk += b"\0" * (BLOCK_SIZE - len(chunk) % BLOCK_SIZE)  # padded with 0 to equal BLOCK_SIZE
        b = aes.encrypt(chunk)
        f_out.write(b)


def aes256_decrypt(key: bytes, iv: bytes, size: int, f_in: BinaryIO, f_out: BinaryIO):
    assert BLOCK_SIZE == 16
    assert CHUNK_SIZE % BLOCK_SIZE == 0
    assert len(iv) == BLOCK_SIZE
    assert len(key) == KEY_SIZE
    aes = AES.new(key, AES_MODE, iv)
    remaining_size = size
    while True:
        chunk = f_in.read(CHUNK_SIZE)
        if len(chunk) == 0:
            return
        b = aes.decrypt(chunk)
        if remaining_size < len(b):
            b = b[:remaining_size]
        f_out.write(b)
        remaining_size -= len(b)


def wrap(key: bytes, plain: bytes) -> bytes:
    iv = random_bytes(16)
    size = len(plain)

    h = sha256_hash(io.BytesIO(plain))
    encrypted_io = io.BytesIO()
    encrypted_io.write(h)
    encrypted_io.write(uint64_to_bytes(size))
    encrypted_io.write(iv)
    aes256_encrypt(key, iv, io.BytesIO(plain), encrypted_io)
    encrypted_io.seek(0)
    encrypted = encrypted_io.read()
    return encrypted


def unwrap(key: bytes, encrypted: bytes) -> bytes:
    encrypted_io = io.BytesIO(encrypted)
    h = encrypted_io.read(HASH_SIZE)
    size = bytes_to_uint64(encrypted_io.read(UINT64_SIZE))
    iv = encrypted_io.read(BLOCK_SIZE)
    decrypted_io = io.BytesIO()
    aes256_decrypt(key, iv, size, encrypted_io, decrypted_io)
    decrypted_io.seek(0)
    decrypted = decrypted_io.read()
    return decrypted


def verify(plain: bytes, encrypted: bytes) -> bool:
    return sha256_hash(io.BytesIO(plain)) == io.BytesIO(encrypted).read(HASH_SIZE)


if __name__ == "__main__":
    key = random_bytes(32)
    plain = b"hello world, this is an example message."
    # encrypt
    encrypted = wrap(key, plain)
    # hash
    assert verify(plain, encrypted)
    assert not verify(plain + b"\1", encrypted)
    # decrypt
    decrypted = unwrap(key, encrypted)
    assert decrypted == plain
