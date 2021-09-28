#date: 2021-09-28T16:58:38Z
#url: https://api.github.com/gists/9bd6ec56efdd1a827dce84c38469ec1c
#owner: https://api.github.com/users/Rajchowdhury420

from Crypto.Util.number import long_to_bytes

enc = open('enc.txt', 'r').read().split(' ')
flag = ''

for i, e in enumerate(enc):
    if i % 2 == 0:
        t = int(e.replace('1', '2').replace('0', '1').replace('2', '0'), 2)
    else:
        t = int(e, 2)
    flag += long_to_bytes(t ^ 0x4d415253).decode()

print(flag)

# TamilCTF{D1g1T_CiRCu1T5_aRe_AwE50Me}