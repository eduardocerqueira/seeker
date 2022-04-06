#date: 2022-04-06T17:14:21Z
#url: https://api.github.com/gists/2aadd337f9a9df5769812bafe0b3cb39
#owner: https://api.github.com/users/raianmr

from random import randint
from math import inf, gcd, sqrt
from typing import Generator

max_int: int = int(2e64)
max_trials: int = int(2e64)


def get_pi() -> Generator:
    '''approximate pi using coprime integers'''
    rand = lambda: randint(1, max_int)
    n_coprimes: int = 0
    for n_trials in range(1, max_trials + 1):
        if gcd(rand(), rand()) == 1:
            n_coprimes += 1
        try:
            yield sqrt((6 * n_trials) / n_coprimes)
        except ZeroDivisionError:
            yield inf


pi_approxs: Generator = get_pi()
for i in pi_approxs:
    print(F"\u03c0 is approx. {i}", end='\r')
