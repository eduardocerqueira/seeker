#date: 2024-05-02T16:54:30Z
#url: https://api.github.com/gists/0ddfcbd100d5bf168eb625e0eac94251
#owner: https://api.github.com/users/mypy-play

from functools import lru_cache
from typing import override

class Foo:
    def bar(self, x: int) -> None: pass

class Bar(Foo):
    @override
    @lru_cache
    def bar(self, x: int) -> None: pass

reveal_type(Bar.bar)