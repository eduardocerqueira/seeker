#date: 2025-04-01T17:14:08Z
#url: https://api.github.com/gists/a55ac74757b3481f1ce9cebe6ec299fa
#owner: https://api.github.com/users/mypy-play

from typing import Generic, TypeVar, Protocol
from typing_extensions import Self

T = TypeVar("T", bound="WithIndex")

class Index(int, Generic[T]):  # int that says what it's the index of
    pass

TipTypeIndex = Index["TipType"]  # expose tagged ints for the API
DeckGeometryIndex = Index["DeckGeometry"]

class WithIndex(Protocol):
    index: Index[Self]  # model must have an index of it's own type

class TipType:
    index: TipTypeIndex  # MyPy says this is ok
    volume: float  # etc.
    
class DeckGeometry:
    index: DeckGeometryIndex
    ...

class BadModel:
    index: int
    name: str
    
class Table(list[T]):
    def fetch(self, index: Index[T]) -> T:
        return next(e for e in self if e.index == index)

t = Table[BadModel]()  # MyPy says untagged index is not good enough

t2 = Table[TipType]([])   # pretend we have some table data
tiptype = t2.fetch(t2[0].index)  # ok and tiptype: TipType
reveal_type(tiptype)

bad_tiptype = t2.fetch(DeckGeometryIndex(2))  # booh