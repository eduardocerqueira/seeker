#date: 2022-07-29T16:51:00Z
#url: https://api.github.com/gists/791fc53a62d9a42836fef5e0412dd686
#owner: https://api.github.com/users/usualsuspect

#!/usr/bin/env python3
#
#   Author: @jaydinbas
#
#   Extract config from Knotweed Jumplump samples
#   Note: Not all samples tagged as 'Jumplump' in the MS report
#         contain a config, some just load other samples that do
#
#   Works for
#       cbae79f66f724e0fe1705d6b5db3cc8a4e89f6bdf4c37004aa1d45eeab26e84b
#       4611340fdade4e36f074f75294194b64dcf2ec0db00f3d958956b4b0d6586431
#       fd6515a71530b8329e2c0104d0866c5c6f87546d4b44cc17bbb03e64663b11fc
#       7f84bf6a016ca15e654fb5ebc36fd7407cb32c69a0335a32bfc36cb91e36184d
#       5d169e083faa73f2920c8593fb95f599dad93d34a6aa2b0f794be978e44c8206
#       7f29b69eb1af1cc6c1998bad980640bfe779525fd5bb775bc36a0ce3789a8bfc
#
#   Rest of the 'Jumplump' hashes are loaders
#

import pefile
import capstone
import sys
import struct
import binascii

DEBUG = 0

def rc4crypt(data, key):
    key = bytearray(key)
    data = bytearray(data)

    x = 0
    box = bytearray([i for i in range(256)])
    for i in range(256):
        x = (x + box[i] + key[i % len(key)]) % 256
        box[i], box[x] = box[x], box[i]
    x,y = 0, 0
    for (i,char) in enumerate(data):
        x = (x + 1) % 256
        y = (y + box[x]) % 256
        box[x], box[y] = box[y], box[x]

        data[i] ^= box[(box[x] + box[y]) % 256]
    return bytes(data)

def parse_config(config):
    print("Config:")
    (unk1,unk2) = struct.unpack_from("<II",config)
    print("    ",unk1) # ports prob
    print("    ",unk2) #



    strings = config[8:].split(b"\x00")
    for s in strings:
        print("    ",s.decode("ascii"))

def find_payload(data,p):
    md = capstone.Cs(capstone.CS_ARCH_X86,capstone.CS_MODE_64)
    md.detail = True

    # Follow code until first jz
    jz = None
    for insn in md.disasm(data[p:],p):
        if DEBUG:
            print("%s %s" % (insn.mnemonic,insn.op_str))
        if insn.mnemonic == "je":
            jz = insn.operands[0].imm
            break
    if not jz:
        if DEBUG:
            print("jz not found")
        return None

    # then follow code flow until 1st call
    payload = None
    p = jz

    for i in range(10): #max 10 instructions
        insn = next(md.disasm(data[p:],p))
        if insn.mnemonic == "jmp":
            p = insn.operands[0].imm
        elif insn.mnemonic == "call":
            payload = p
            break
        else:
            p += insn.size

    if not payload:
        return None
    else:
        payload += 5 # skip call
        return payload

# call+5/pop rcx/sub rcx, 5
pic_pattern = b"\xE8\x00\x00\x00\x00\x59\x48\x83\xE9\x05"

data = open(sys.argv[1],"rb").read()

p = data.find(pic_pattern)
if p == -1:
    print("[-] PIC not found")
    sys.exit(0)

payload = find_payload(data,p)
if payload:
    print("[+] Payload found @ %x" % payload)
else:
    print("[-] Payload not found")
    sys.exit(0)

rc4_key = data[payload:payload+16]
config_len_enc = data[payload+16:payload+16+4]

config_len = struct.unpack("<I",rc4crypt(config_len_enc,rc4_key))[0]

print("[+] RC4 key: %s" % binascii.hexlify(rc4_key))
print("[+] Config len: %d bytes" % config_len)

config_enc = data[payload+20:payload+20+config_len]
config = rc4crypt(config_enc,rc4_key)
parse_config(config)
