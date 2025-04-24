#date: 2025-04-24T16:57:32Z
#url: https://api.github.com/gists/5ad7c80a66c5149aa2b66bd408da2d48
#owner: https://api.github.com/users/mypy-play

from typing import Callable

def g(y: Callable[[int], str]):
    reveal_type(type(y))
    reveal_type(y.__class__)
    reveal_type(type(y).__call__)