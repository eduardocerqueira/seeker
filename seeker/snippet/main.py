#date: 2025-12-10T16:53:43Z
#url: https://api.github.com/gists/c8524dd3d7d41f6f2a4b8903b3c2e121
#owner: https://api.github.com/users/mypy-play

from typing import Iterator


def fib(n: int) -> Iterator[int]:
    a, b = 0, 1
    while a < n:
        yield a
        a, b = b, a + b


fib(10)
fib("10")
