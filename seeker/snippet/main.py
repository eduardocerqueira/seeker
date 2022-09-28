#date: 2022-09-28T17:27:49Z
#url: https://api.github.com/gists/03e5dddadaa9eb99e4e18d3e99ad96a7
#owner: https://api.github.com/users/mypy-play

from typing import Any, Callable, Optional

class C: ...
class D(C): ...


def d2(f: Callable[[C], None]) -> None: ...

def f(x: D) -> None: ...


d2(f=f)
