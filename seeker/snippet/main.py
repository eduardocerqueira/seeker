#date: 2025-02-13T17:01:14Z
#url: https://api.github.com/gists/1b52e554cf157afd01973a1e7de3dc4f
#owner: https://api.github.com/users/mypy-play

from typing import Protocol, assert_type
from collections.abc import Callable

def tuple_of_nums(n: int) -> tuple[int,...]:
    return tuple(range(n))

class IntCallable(Protocol):
    def __call__(self, n: int) -> tuple[int, ...]: ...
    
assert_type(tuple_of_nums, IntCallable)