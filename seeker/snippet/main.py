#date: 2023-11-13T16:52:18Z
#url: https://api.github.com/gists/be6764bbead8f238976dd1be12d8d7e8
#owner: https://api.github.com/users/mypy-play

# mypy: old-type-inference
from __future__ import annotations

import contextlib
from typing import AsyncIterator, Callable, TypeVar

T = TypeVar("T")


class C:
    pass


@contextlib.asynccontextmanager
async def _do_open_client(make_client: Callable[[], T]) -> AsyncIterator[T]:
    yield make_client()


@contextlib.asynccontextmanager
async def open_client() -> AsyncIterator[C]:
    def make_client() -> C:
        return C()

    client: C
    reveal_type(make_client)
    reveal_type(_do_open_client)
    reveal_type(_do_open_client(make_client))
    async with _do_open_client(make_client) as client:
        yield client