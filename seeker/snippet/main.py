#date: 2021-11-19T17:08:46Z
#url: https://api.github.com/gists/ae3fc37534bf0861c953156a74a91485
#owner: https://api.github.com/users/mypy-play

import asyncio
from typing import Any, Awaitable


async def foo(y):
    await asyncio.sleep(y)
    return 1


def bar(x):
    return foo(x)
    
    
def baz() -> Awaitable[int]:
    return bar(x=4.2)