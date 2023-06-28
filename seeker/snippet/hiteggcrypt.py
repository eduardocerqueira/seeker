#date: 2023-06-28T17:04:40Z
#url: https://api.github.com/gists/6d19fb8f0eef8297e85371bf3f89e924
#owner: https://api.github.com/users/thesupersonic16

# Encrypt and decrypt Game Gear ROMs for Sonic Origins
# Requires pycryptodome
#
# SuperSonic16, 2023

import sys
import os
import base64
from Crypto.Cipher import AES
from Crypto.Util.Padding import pad

# Encryption Key
_key = bytes.fromhex('24584485619843E24BA06324577EE51A')
# Initialisation Vector
_iv = bytes.fromhex('99E98C02F5BEC78CB95598EE5E5B9E9A') # Might aswell reuse the IV from the Sonic 2 GG ROM

def encrypt(data, key, iv):
    aes = AES.new(key, AES.MODE_CBC, iv)
    return iv + aes.encrypt(pad(data, AES.block_size))

def decrypt(data, key, iv):
    iv = data[:16]
    aes = AES.new(key, AES.MODE_CBC, iv)
    return aes.decrypt(data[16:])

def processFile(filename):
    with open(filename, 'rb') as f:
        data = f.read()
        if filename[-4:] == '.bin':
            data = decrypt(data, _key, _iv)
            filename = filename[:-4] + '.gg'
        elif filename[-3:] == '.gg':
            data = encrypt(data, _key, _iv)
            filename = filename[:-3] + '.bin'
        else:
            print('Unknown file extension: %s' % filename[-4:])
            sys.exit(1)
        # Write file
        with open(filename, 'wb') as f:
            f.write(data)

if __name__ == '__main__':
    if len(sys.argv) != 2:
        print('Usage: python3 hiteggcrypt.py <filename>')
        sys.exit(1)

    if os.path.isdir(sys.argv[1]):
        for filename in os.listdir(sys.argv[1]):
            filename = os.path.join(sys.argv[1], filename)
            processFile(filename)
            print('Processed file: %s' % filename)
    else:
        filename = sys.argv[1]
        processFile(filename)
        print('Processed file: %s' % filename)