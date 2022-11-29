#date: 2022-11-29T17:05:18Z
#url: https://api.github.com/gists/444e060e7f040691aec79ef1454e1c1c
#owner: https://api.github.com/users/mypy-play

from typing import Iterator, Generic, TypeVar

T = TypeVar('T')

class Myg(Generic[T]):
    pass

def fib(n: int) -> Iterator[int]:
    a, b = 0, 1
    while a < n:
        yield a
        a, b = b, a + b


print(fib(10))
# fib("10")
