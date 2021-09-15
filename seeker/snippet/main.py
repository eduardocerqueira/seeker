#date: 2021-09-15T17:14:03Z
#url: https://api.github.com/gists/e643ccc4cf7246e8899fe35e0e370263
#owner: https://api.github.com/users/mypy-play

from typing import *

TElement = Union[Tuple[Literal["int"], int], Tuple[Literal["str"], str]]

xs: List[TElement] = []
xs.append(("int", 1))
xs.append(("str", "what"))
