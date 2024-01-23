#date: 2024-01-23T17:00:43Z
#url: https://api.github.com/gists/4c65048a466f5a8b19a98371d1a9e6b0
#owner: https://api.github.com/users/woodrush

import sys

def leven(n):
    if n == 0:
        return "0"
    ret = ""
    c = 0
    while True:
        c += 1
        s = f"{n:b}"[1:]
        ret = s + ret
        m = len(s)
        if m > 0:
            n = m
        else:
            break
    ret = ("1" * c) + "0" + ret
    return ret

src = input().rstrip()

l_vars = []
i_bit = 0

out = ""
while i_bit < len(src):
    if src[i_bit] == "0":
        out = out + src[i_bit]
        i_bit += 1
        out = out + src[i_bit]
        i_bit += 1
    else:
        i_bit += 1
        N = 0
        while src[i_bit] == "1":
            i_bit += 1
            N += 1
        l_vars.append(N)
        out = out + "1" + leven(N)
        i_bit += 1

def count_vars(l):
    d_count = {}
    for v in l:
        if v not in d_count.keys():
            d_count[v] = 0
        d_count[v] += 1
    return d_count

print(out, end="")

print("Variable stats:", file=sys.stderr)
for v, count in sorted(count_vars(l_vars).items()):
    print(v, count, file=sys.stderr)
