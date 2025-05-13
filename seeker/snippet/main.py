#date: 2025-05-13T16:41:44Z
#url: https://api.github.com/gists/f4cfec7e84ea501437bcf486eac997d5
#owner: https://api.github.com/users/mypy-play

from __future__ import annotations

import typing as t


_ST = t.TypeVar("_ST")
_GT = t.TypeVar("_GT")


class Field(t.Generic[_ST, _GT]):
    @t.overload
    def __new__(
        cls, *, null: t.Literal[True]
    ) -> Field[_ST | None, _GT | None]: ...

    @t.overload
    def __new__(
        cls, *, null: t.Literal[False] = False
    ) -> Field[_ST, _GT]: ...

    def __new__(   # type: ignore[misc]
        cls, *, null: bool = False
    ) -> (
        Field[_ST, _GT]
        | Field[_ST | None, _GT | None]
    ):
        return super().__new__(cls)


_ST_IntegerField = t.TypeVar("_ST_IntegerField", default=float | int | str)
_GT_IntegerField = t.TypeVar("_GT_IntegerField", default=int)

class IntegerField(Field[_ST_IntegerField, _GT_IntegerField]):
    ...


x = IntegerField(null=False)
reveal_type(x)
y = IntegerField(null=True)
reveal_type(y)
z = IntegerField()
reveal_type(z)
