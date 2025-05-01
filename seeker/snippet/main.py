#date: 2025-05-01T16:45:59Z
#url: https://api.github.com/gists/9ec02e4e70b4fa04c07eb0ccece364f8
#owner: https://api.github.com/users/mypy-play

from collections.abc import AsyncIterator, Callable
from typing import Any, Callable


type TaskGroup = Any
create_task_group: Any


async def yield_to_start[*A, R](
    tg: TaskGroup,
    func: Callable[[*A], AsyncIterator[R]],
    *args: *A,
    name: str | None = None,
) -> R:
    ...


class MyResult: ...


def my_task() -> AsyncIterator[MyResult]: ...


async def main() -> None:
    async with create_task_group() as tg:
        result = await yield_to_start(tg, my_task)
        reveal_type(result)
