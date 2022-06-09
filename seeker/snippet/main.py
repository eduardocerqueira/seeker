#date: 2022-06-09T17:07:23Z
#url: https://api.github.com/gists/a6dafa30b07b4e658579eb37f69ea6f1
#owner: https://api.github.com/users/mypy-play

from typing import Iterator


def fib(n: int) -> Iterator[int]:
    a, b = 0, 1
    while a < n:
        yield a
        a, b = b, a + b


fib(10)
fib("10")
