#date: 2023-08-22T17:04:25Z
#url: https://api.github.com/gists/8e3a4d537d6054de128e1ff4765c7a0c
#owner: https://api.github.com/users/mypy-play

from typing import Generic, Self, TypeVar

_T = TypeVar("_T")

class Foo(Generic[_T]):
    t: _T

    def __new__(cls, t: _T) -> Self:
        instance = super().__new__(cls)
        instance.t = t
        return instance


class SubFoo(Foo[_T]):
    def __new__(cls, t: _T) -> Self:
        instance = super().__new__(cls, t)
        return instance
