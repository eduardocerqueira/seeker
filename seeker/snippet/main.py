#date: 2025-06-23T17:12:30Z
#url: https://api.github.com/gists/85ac121be4cb2c733c620ef80e8cf132
#owner: https://api.github.com/users/mypy-play

import asyncio
import typing

async def count() -> int:
    print("One")
    await asyncio.sleep(1)
    print("Two")
    return 1
    
async def main3() -> int:
    y = [x.result() async for x in asyncio.as_completed(count() for _ in range(3))]
    typing.reveal_type(y)
    return len(y)