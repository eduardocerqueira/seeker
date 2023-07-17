#date: 2023-07-17T16:46:00Z
#url: https://api.github.com/gists/98f51d9fa1f7280719dc6df5da23503b
#owner: https://api.github.com/users/mypy-play

from __future__ import annotations

from typing import Generic, TypeVar, cast


T = TypeVar("T")
U = TypeVar("U")


class Foo(Generic[T, U]):
    def __init__(self, x: T, y: U):
        self.x = x
        self.y = y

    def op1(self) -> float:
        if not isinstance(self.y, float):
            raise ValueError("can't use op1 with non-numeric types")
        return self.y * 2.5



Foo(0, 1).op1()     # ok
Foo(0, 1.0).op1()   # ok (float type for bound of U also accepts ints)
Foo(0, "1").op1()   # error: Value of type variable "U" of "Foo" cannot be "str"  [type-var]
