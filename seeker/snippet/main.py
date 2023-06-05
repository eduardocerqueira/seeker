#date: 2023-06-05T16:57:17Z
#url: https://api.github.com/gists/4eb527cbef6cebc403b3d6bb3fae3873
#owner: https://api.github.com/users/mypy-play

from typing import TypeVar

T = TypeVar("T")

def f1(x: T) -> T:
    if isinstance(x, str):
        return x  # Type error here
    return x

def f2(x: T) -> T:
    if not isinstance(x, str):
        return x
    return x      # Type error here
    
def f3(x: T) -> T:
    if not isinstance(x, str):
        pass
    return x      # But not here (even though the same code path is taken)
    
    
