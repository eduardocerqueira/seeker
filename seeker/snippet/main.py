#date: 2024-04-15T16:41:00Z
#url: https://api.github.com/gists/d98b2520e6a9bc2d5e4106a6a80484f7
#owner: https://api.github.com/users/mypy-play

from collections.abc import Iterator
from typing import Protocol, Self, TypeVar, overload, runtime_checkable


T_co = TypeVar("T_co", covariant=True)


@runtime_checkable
class Collection(Protocol[T_co]):
  def __len__(self) -> int: ...

  def __iter__(self) -> Iterator[T_co]: ...

  @overload
  def __getitem__(self, idx: int) -> T_co: ...
  @overload
  def __getitem__(self, idx: slice) -> Self: ...

  def __add__(self, other: Self) -> Self: ...

  def __mul__(self, other: int) -> Self: ...


coll: Collection[Collection[int]]
reveal_type(coll[3])
