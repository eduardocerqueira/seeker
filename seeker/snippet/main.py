#date: 2025-04-29T16:51:20Z
#url: https://api.github.com/gists/d16882075378d49917975e85ecb2970f
#owner: https://api.github.com/users/mypy-play

from typing import Never

def never_returns() -> Never:
    raise NotImplementedError

def g(x: str) -> str:
    return x

def h() -> str:
    return g(never_returns())  # no warning


s: str = h()  # no warning
print(s)  # no warning
