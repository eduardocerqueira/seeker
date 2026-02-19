#date: 2026-02-19T17:33:35Z
#url: https://api.github.com/gists/3ca33d23318be39443147a97f64ada7b
#owner: https://api.github.com/users/mypy-play

from typing import Any, TypeVar

T = TypeVar('T')

def cart_prod(*sets: set[T]) -> set[tuple[T, ...]]:
    pass

cart_prod(({1,}))