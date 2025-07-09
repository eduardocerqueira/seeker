#date: 2025-07-09T17:02:35Z
#url: https://api.github.com/gists/793a7496c7789e6211430a25bab446a3
#owner: https://api.github.com/users/mypy-play

from dataclasses import dataclass
from typing import Protocol


class Child(Protocol):
    val: float

class Parent(Protocol):
    sub: Child


@dataclass
class Child1(Child):
    val: float

@dataclass
class Parent1(Parent):
    sub: Child1


