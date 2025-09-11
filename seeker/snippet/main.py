#date: 2025-09-11T17:00:50Z
#url: https://api.github.com/gists/9921c25d6e9422d15bdc2164763803d8
#owner: https://api.github.com/users/mypy-play

from typing import Iterator, Collection, Sized

col: Collection | Iterator = range(3)

if isinstance(col, Sized):
    stpe = col.step if isinstance(col, range) else 1
