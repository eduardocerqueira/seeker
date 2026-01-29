#date: 2026-01-29T17:31:10Z
#url: https://api.github.com/gists/8cc28f445b2bea9676e35d0b32b9b005
#owner: https://api.github.com/users/mypy-play

from collections.abc import Callable
from typing import Protocol


class RandomAccessContainer[T](Protocol):

    def append(self, value: T) -> None: ...

    def __getitem__(self, index: int) -> T: ...

    def __len__(self) -> int: ...


class OfflinePacker[T]:

    container: RandomAccessContainer[T]
    var_a: str
    var_b: str

    def __init__(
        self,
        var_a: str,
        var_b: str,
        *,
        container_factory: Callable[[], RandomAccessContainer[T]] = list
    ) -> None:
        self.var_a = var_a
        self.var_b = var_b
        self.container = container_factory()
