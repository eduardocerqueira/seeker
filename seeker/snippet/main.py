#date: 2024-05-21T17:01:37Z
#url: https://api.github.com/gists/a6011d355ec44069336271824c152f79
#owner: https://api.github.com/users/mypy-play

from typing import *

def asd(*a: object, **aa: int) -> None:
    reveal_type(a)
    reveal_type(aa)