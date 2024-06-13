#date: 2024-06-13T17:07:04Z
#url: https://api.github.com/gists/811ef37c11afe70cec6f41d6ad8eb3db
#owner: https://api.github.com/users/mypy-play

from typing import Iterator


def fib(n: int) -> Iterator[int]:
    a, b = 0, 1
    while a < n:
        yield a
        a, b = b, a + b


fib(10)
fib("10")
