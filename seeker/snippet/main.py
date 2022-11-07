#date: 2022-11-07T17:15:08Z
#url: https://api.github.com/gists/70a93587197c46fc884b1c14459ac937
#owner: https://api.github.com/users/mypy-play

from typing import Callable, TypeVar, Iterable

T = TypeVar("T")

def f(a: int, f: Callable[[Iterable[T]], Iterable[T]]) -> int:
    for b in f([a] * 3):
        print(b)
    
    return b