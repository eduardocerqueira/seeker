#date: 2023-11-24T17:05:04Z
#url: https://api.github.com/gists/cd965206f2ae7f9cc06448c52846e1b8
#owner: https://api.github.com/users/mypy-play

from typing import *

T = TypeVar("T")
R = TypeVar("R")

@overload
def _map(value: T, func: Callable[[T], R]) -> R:
    pass

@overload
def _map(value: None, func: Callable[[T], R]) -> None:
    pass


def _map(value: T | None, func: Callable[[T], R]) -> R | None:
    if value is None:
        return None
    else:
        return func(value)
        
x: None = _map(None, lambda x: "lol")
y: str = _map("sjid", lambda x: "lol")