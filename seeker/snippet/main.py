#date: 2022-09-29T17:06:15Z
#url: https://api.github.com/gists/6f2b3c5f87f6d85d0e6f0a4627fe87ee
#owner: https://api.github.com/users/mypy-play

from __future__ import annotations

import typing as _typing

number = _typing.Union[float, int]


class Vec2:
    __slots__ = "x", "y"

    def __init__(self, x: number = 0.0, y: number = 0.0) -> None:
        self.x = x
        self.y = y


v = Vec2(2, 3)
range(v.x - 2)
