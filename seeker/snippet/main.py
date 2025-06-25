#date: 2025-06-25T17:09:47Z
#url: https://api.github.com/gists/f4e6279a69580bf9150e70e7545fb1b3
#owner: https://api.github.com/users/mypy-play

from typing import Iterator


def fib(n: int) -> Iterator[int]:
    a, b = 0, 1
    while a < n:
        yield a
        a, b = b, a + b


fib(10)
fib("10")
