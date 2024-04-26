#date: 2024-04-26T16:50:39Z
#url: https://api.github.com/gists/df0641e6b228044acd5f1d9c21c70d65
#owner: https://api.github.com/users/mypy-play

from typing import TypeVar

T = TypeVar("T")
U = T

def f(x: T) -> U: ...

x: int = 3
reveal_type(f(x))