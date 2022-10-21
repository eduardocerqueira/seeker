#date: 2022-10-21T17:20:12Z
#url: https://api.github.com/gists/3fd75bcd0bf8e5cb5e9ab52cd8902263
#owner: https://api.github.com/users/dyoform

from math import factorial as f

def Z(n,c):
    return int(f(n+c-1)/(f(n)*f(c-1)))

def C(m,n,c):
    if c == 1:
        return [n]
    q = 0
    for d in range(0,n+1):
        z = Z(n-d,c-1)
        if q + z > m:
            return [d] + C(m-q,n-d,c-1)
        q += z