#date: 2025-11-18T17:06:56Z
#url: https://api.github.com/gists/4e3cad30b12c7309968909dfb57f1e40
#owner: https://api.github.com/users/mypy-play

from typing import Iterator


def fib(n: int) -> Iterator[int]:
    a, b = 0, 1
    while a < n:
        yield a
        a, b = b, a + b


fib(10)
fib("10")
