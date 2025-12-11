#date: 2025-12-11T16:59:40Z
#url: https://api.github.com/gists/da69f7e9eef9544a0e19fe5f32ffc34d
#owner: https://api.github.com/users/mypy-play

from typing import Iterator


def fib(n: int) -> Iterator[int]:
    a, b = 0, 1
    while a < n:
        yield a
        a, b = b, a + b


fib(10)
fib("10")
