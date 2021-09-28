#date: 2021-09-28T17:03:31Z
#url: https://api.github.com/gists/c3e332ad2ef78432700e133428f7fa9c
#owner: https://api.github.com/users/Rajchowdhury420

f = bytearray(b'\x16\x0a\x1f\x10\x18\x1b\x09\x0b\x1e\x03\x08\x14\x1d\x00\x07\x11\x17\x13\x15\x1c\x12\x02\x06\x04\x19\x05\x1a\x01\x00')
enc_flag = bytearray(b'\x70\x53\x6a\x71\x34\x7d\x81\x50\x63\x48\x68\x58\x59\x63\x49\x6d\x47\x65\x4a\x73\x58\x72\x4c\x63\x79\x4b\x7f\x78')

def foo():
    res = [0] * 0x1c
    for i in range(0x1c):
        res[i] = f[i] ^ 0x12
    return res

buf = []
flag = [''] * 0x1c
a = foo()
for i in range(0x1c):
    flag[a[i]] += chr(enc_flag[i] - 4)

print(''.join(flag))

# TamilCTF{y0u_foUnD_tHE_GOLd}