#date: 2023-10-25T17:02:45Z
#url: https://api.github.com/gists/8dc441cef674100f78ed663afd465029
#owner: https://api.github.com/users/mypy-play

from __future__ import annotations
from typing import Optional


def func1(value: Optional[object]) -> None:
    if callable(value):
        _ = True  # got: Statement is unreachable [unreachable]
                  # expected: no error


# ADDITIONAL MATERIAL - NOT NECESSARY TO REPRODUCE:

def func2(value: object) -> None:
    if callable(value):
        _ = True  # no error (as expected)
        
obj1: object
reveal_type(callable(obj1))  # Revealed type is 'bool' (as expected)

obj2: Optional[object]
reveal_type(callable(obj2))  # Revealed type is 'bool' (as expected)