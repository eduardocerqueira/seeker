#date: 2024-07-16T16:53:32Z
#url: https://api.github.com/gists/a89be36f99ce0ecc5bbdc2fc027b6bbd
#owner: https://api.github.com/users/ptmcg

import functools
import itertools
from collections.abc import Iterator


def is_multiple_of(p: int, x: int) -> bool:
    # if x % p == 0:
    #     print("discarding", x, "multiple of", p)
    return x % p == 0


def sieve_of_eratosthenes(n: int) -> Iterator[int]:
    # start with list of all integers, starting with 2
    ints: Iterator[int] = itertools.count(start=2)

    for _ in range(n):
        next_prime = next(ints)
        yield next_prime

        # wrap ints iterator in another filter, removing
        # multiples of next_prime
        ints = itertools.filterfalse(
            functools.partial(is_multiple_of, next_prime),
            ints,
        )


# print the first 30 prime numbers
print(list(sieve_of_eratosthenes(30)))
