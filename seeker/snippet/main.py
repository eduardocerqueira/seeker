#date: 2023-08-02T16:49:01Z
#url: https://api.github.com/gists/6ead32a47aa5a6fb4d6f8548669b8359
#owner: https://api.github.com/users/mypy-play

from typing import Protocol, TypeVar, Self, Generic
from dataclasses import dataclass

class SupportsSub(Protocol):
    def __sub__(self, other: Self) -> Self: pass

T = TypeVar('T', bound=SupportsSub)

@dataclass
class Consumption(Generic[T]):
    first: T
    second: T
    def difference(self) -> T:
        return self.second - self.first


def shouldnt_type(a: T, b: int) -> T:
    return a - b # doesn't type

def should_type1(a: int, b: int) -> int:
    return a - b

def should_type2(a: T, b: T) -> T:
    return a - b

data = Consumption(6, 3)
data.difference()

bad_data = Consumption("a", "b")
bad_data.difference()
