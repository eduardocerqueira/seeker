#date: 2023-02-02T16:46:12Z
#url: https://api.github.com/gists/d71f32addd640429793c4f60c164d374
#owner: https://api.github.com/users/mypy-play

from typing import assert_never
from enum import Enum, auto


def f(x: int | list[int] | str) -> None:
    if isinstance(x, int):
        ...
    elif isinstance(x, str):
        ...
    else:
        assert_never(x)


class E(Enum):
    A = auto()
    B = auto()
    C = auto()

    
def g(e: E) -> None:
    if e is E.A:
        ...
    elif e is E.B:
        ...
    else:
        assert_never(e)
