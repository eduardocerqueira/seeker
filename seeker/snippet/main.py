#date: 2026-02-09T17:46:14Z
#url: https://api.github.com/gists/12a903f8a2c80999db5adf4449261344
#owner: https://api.github.com/users/mypy-play

from collections.abc import Callable
from types import FunctionType
from typing import NewType, Literal as L, Protocol

PositiveInt = NewType("PositiveInt", int)

bogus: Callable[[str], str] = PositiveInt
confusion: Callable[[None, None, str], None] = PositiveInt

class What(Protocol):
    def __call__(self, foo: str, bar: bytes) -> str:
        ...

z: What = PositiveInt
z(3, foo = "", bar = b"")


unconstructable: type[PositiveInt] = PositiveInt 
nested: type[type[PositiveInt]] = type(PositiveInt)


def infered_ok(x: int):
    return PositiveInt, PositiveInt(x)

def annotated_bad(x: int) -> tuple[type[PositiveInt], PositiveInt]:
    return PositiveInt, PositiveInt(x)







