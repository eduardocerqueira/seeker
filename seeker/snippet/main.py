#date: 2025-08-14T16:51:18Z
#url: https://api.github.com/gists/d87a730220ffcefe63d4cb81203fb70a
#owner: https://api.github.com/users/mypy-play

from typing import NamedTuple, assert_type

class A(NamedTuple):
    name: str

def f[T: NamedTuple](x: T) -> T:
    return x

aa =A(name="John")
fr = f(aa)