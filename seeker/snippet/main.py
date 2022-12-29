#date: 2022-12-29T16:29:46Z
#url: https://api.github.com/gists/6ba6b1037f983f1de25d685f9045577d
#owner: https://api.github.com/users/mypy-play

from typing import Optional, List
def foo(a: dict = None, b: Optional[List[str]] = None):
    if b is not None:
        ...
