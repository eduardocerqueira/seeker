#date: 2021-10-06T17:08:25Z
#url: https://api.github.com/gists/3343d611717db1c15f241a29a4d20653
#owner: https://api.github.com/users/mypy-play

from typing import Optional
from dataclasses import dataclass


@dataclass
class A:
    b: Optional[int] = ...