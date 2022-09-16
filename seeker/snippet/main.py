#date: 2022-09-16T22:01:48Z
#url: https://api.github.com/gists/df994c767a44875604ea5c26cc351c23
#owner: https://api.github.com/users/mypy-play

from __future__ import annotations

from collections import OrderedDict
from typing import Iterable, TypeVar, overload

from _typeshed import SupportsKeysAndGetItem

_KT = TypeVar('_KT')
_VT = TypeVar('_VT')


class LRU(OrderedDict):

    @overload
    def __init__(self) -> None:
        ...

    @overload
    def __init__(self: dict[str, _VT], **kwargs: _VT) -> None:
        ...

    @overload
    def __init__(self, __map: SupportsKeysAndGetItem[_KT, _VT]) -> None:
        ...

    @overload
    def __init__(
        self: dict[str, _VT], __map: SupportsKeysAndGetItem[str, _VT], **kwargs: _VT
    ) -> None:
        ...

    @overload
    def __init__(self, __iterable: Iterable[tuple[_KT, _VT]]) -> None:
        ...

    @overload
    def __init__(
        self: dict[str, _VT], __iterable: Iterable[tuple[str, _VT]], **kwargs: _VT
    ) -> None:
        ...

    @overload
    def __init__(self: dict[str, str], __iterable: Iterable[list[str]]) -> None:
        ...

    def __init__(
        self: dict[str, _VT] | dict[str, str],
        __map_or_iterable: SupportsKeysAndGetItem[_KT, _VT]
        | SupportsKeysAndGetItem[str, _VT]
        | Iterable[tuple[_KT, _VT]]
        | Iterable[tuple[str, _VT]]
        | Iterable[list[str]]
        | None = None,
        **kwargs: _VT | None,
    ) -> None:
        pass

my_cache = LRU()