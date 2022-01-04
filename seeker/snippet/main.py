#date: 2022-01-04T16:56:39Z
#url: https://api.github.com/gists/c3203762bd20579f26872607e1bcd42e
#owner: https://api.github.com/users/mypy-play

from typing import Optional


def something(x: int) -> Optional[int]:
    if x == 5:
        return None
    return x + 1
    
    
x = something(1)
y = something(5)
z = x + y