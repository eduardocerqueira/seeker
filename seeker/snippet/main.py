#date: 2022-04-05T16:54:45Z
#url: https://api.github.com/gists/bcbea1778f44f8cbbf60015e32fd356c
#owner: https://api.github.com/users/mypy-play

import dataclasses
from typing import Generic, TypeVar

T = TypeVar("T", bound="NotDefined")

@dataclasses.dataclass
class C(Generic[T]):
    x: float