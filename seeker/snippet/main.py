#date: 2024-11-12T16:49:09Z
#url: https://api.github.com/gists/7d6f76d5d127ba3210fed1dad39758f7
#owner: https://api.github.com/users/mypy-play

from typing import Literal, reveal_type

def f[T](x: T) -> T:
    return x

reveal_type(f(1))

def produces_literal_1() -> Literal[1]:
    return 1

reveal_type(f(produces_literal_1()))
