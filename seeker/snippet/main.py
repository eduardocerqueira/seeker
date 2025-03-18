#date: 2025-03-18T16:58:34Z
#url: https://api.github.com/gists/8294ea8d49b3905b764fd2680673a3b7
#owner: https://api.github.com/users/mypy-play

from typing import Iterator


def fib(n: int) -> Iterator[int]:
    a, b = 0, 1
    while a < n:
        yield a
        a, b = b, a + b


fib(10)
fib("10")
