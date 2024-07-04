#date: 2024-07-04T17:11:13Z
#url: https://api.github.com/gists/f9998fe7424230363f23afd323846744
#owner: https://api.github.com/users/motyzk

from pprint import pprint

def fact(n: int) -> int:
    if n == 1:
        return 1
    return n * fact(n-1)

# print(fact(3))


# memoization
cache = {}
def fib(n):
    if n == 0 or n == 1:
        return n
    if n not in cache:
        cache[n] = fib(n-2) + fib(n-1)
    return cache[n]


cache = {0: 0, 1: 1}
def fib(n):
    if (n) not in cache:
        cache[n] = fib(n-2) + fib(n-1)
    return cache[n]


print(fib(40))
pprint(cache)
