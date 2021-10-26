#date: 2021-10-26T17:07:37Z
#url: https://api.github.com/gists/716e9682e91f4c2eceab39d8a122d56d
#owner: https://api.github.com/users/CyberCommands

#!/usr/bin/env python3
import json

def encrypt():
    with open("encrypt.json") as file:
        data = json.load(file)
    return data["encrypted"]

png_header = bytes([0x89, 0x50, 0x4e, 0x47, 0x0d, 0x0a, 0x1a, 0x0a, 0x00, 0x00, 0x00, 0x0d, 0x49, 0x48, 0x44, 0x52])
encrypted = bytes.fromhex(encrypt())

keys = []
for i in range(len(png_header)):
    keys.append(png_header[i] ^ encrypted[i])

print(keys)

flag = [0] * len(encrypted)
for i in range(len(encrypted)):
    flag[i] = encrypted[i] ^ keys[i % len(keys)]

with open("recovery.png", "wb") as fd:
    fd.write(bytes(flag))

print("\n\033[32m[+] The Flag was Recovered! \33[0m")