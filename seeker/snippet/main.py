#date: 2023-11-02T17:04:52Z
#url: https://api.github.com/gists/c9a6c7936a8b288ff5e9b13aa9f0b843
#owner: https://api.github.com/users/mypy-play

from typing import ClassVar, Self

class ProxyProvider:
    _instance: ClassVar[Self | None] = None

    @classmethod
    def singleton(cls) -> Self:
        if cls._instance is None:
            cls._instance = cls()
        return cls._instance
