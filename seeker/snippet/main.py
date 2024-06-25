#date: 2024-06-25T17:11:00Z
#url: https://api.github.com/gists/339a5dda513e25300f481742d919be33
#owner: https://api.github.com/users/mypy-play

from __future__ import annotations

import dataclasses
from typing import Generic, Iterable, TypeVar

T = TypeVar("T")
InternalIdT = str


@dataclasses.dataclass
class ExternalIdMapping(Generic[T]):
    internal_id: InternalIdT
    external_id: T


class TidMapping(ExternalIdMapping[int]):
    pass


class BicMapping(ExternalIdMapping[str]):
    pass


def get_many(
    mapping_cls: type[ExternalIdMapping[T]], ids: Iterable[InternalIdT]
) -> dict[InternalIdT, T]:
    raise NotImplementedError


reveal_type(get_many(TidMapping, ["a"]))
reveal_type(get_many(BicMapping, ["a"]))
