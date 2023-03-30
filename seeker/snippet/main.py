#date: 2023-03-30T16:42:51Z
#url: https://api.github.com/gists/e65912984086b1cb9faa29a1b5ff7796
#owner: https://api.github.com/users/mypy-play

from typing import TypeVar

Type = TypeVar('Type', str, int)

def outer(value: Type) -> None:
    async def inner() -> None:
        print(1)
        print(2)