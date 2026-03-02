#date: 2026-03-02T17:29:43Z
#url: https://api.github.com/gists/0b652aaa4ec3f480216b435fb485e840
#owner: https://api.github.com/users/mypy-play

from typing import Iterator


def fib(n: int) -> Iterator[int]:
    a, b = 0, 1
    while a < n:
        yield a
        a, b = b, a + b


fib(10)
fib("10")
