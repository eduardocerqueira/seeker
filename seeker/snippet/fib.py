#date: 2021-09-13T17:03:42Z
#url: https://api.github.com/gists/901e9c2a95d4eccbc814d67f15386690
#owner: https://api.github.com/users/carlosmorette

fib_cache = {}

def fib(n):
    if n in fib_cache:
        return fib_cache[n]
    elif n <= 1:
        return n
    else:
        value = fib(n - 1) + fib(n - 2)
        fib_cache[n] = value
        return value


def other_fib(n, t):
    print((n, t))
    if t == 1:
        return n
    else:
        x = n + [n[-1] + n[-2]]
        return other_fib(x, t - 1)