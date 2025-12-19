#date: 2025-12-19T16:51:11Z
#url: https://api.github.com/gists/8bbf4ed9c85b78fa67284b9002739b1b
#owner: https://api.github.com/users/mypy-play

from typing import *

    
async def async_iter_with_yield() -> AsyncIterator[None]: #defines an async iterator
    yield

async def async_iter_with_return() -> AsyncIterator[None]: #returns an async iterator (after calling "async_iter_with_yield", an async iterator) - returns a coroutine that must be awaited first to get the async iterator
    return async_iter_with_yield()

async def async_gen_with_yield() -> AsyncGenerator[None, None]: #defines an async generator
    yield

async def async_gen_with_return() -> AsyncGenerator[None, None]: #returns an async generator (after calling "async_gen_with_yield", an async generator) - returns a coroutine that must be awaited to get the async generator
    return async_gen_with_yield()

async def test() -> None:
    # pyre is right
    async for _ in async_iter_with_yield():
        ...

    # pyre says OK, but it should have reported errors
    async for _ in async_iter_with_return():
        ...

    # pyre wrongly reports: Expected an awaitable but got `AsyncIterator[None]`
    async for _ in await async_iter_with_return():
        ...

    # pyre is right
    async for _ in async_gen_with_yield():
        ...

    # pyre correctly reports: `typing.Coroutine` has no attribute `__aiter__`
    async for _ in async_gen_with_return():
        ...

    # pyre is right
    async for _ in await async_gen_with_return():
        ...