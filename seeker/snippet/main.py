#date: 2021-10-14T17:08:07Z
#url: https://api.github.com/gists/00a521de30367688163d94a9f01fc595
#owner: https://api.github.com/users/mypy-play

from typing import Iterator


def fib(n: int) -> Iterator[int]:
    a, b = 0, 1
    while a < n:
        yield a
        a, b = b, a + b


fib(10)
fib("10")
