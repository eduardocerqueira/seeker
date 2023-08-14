#date: 2023-08-14T17:04:35Z
#url: https://api.github.com/gists/3ace70ead7cf7bfa93469167e009d88b
#owner: https://api.github.com/users/mypy-play

from typing import Any, Awaitable, Callable, TypeVar, cast


Func = TypeVar('Func', bound=Callable[[int], Awaitable[int]])


def my_dec(func: Func) -> Func:
    async def wrapper(value: Any) -> int:
        reply = await func(value)
        return reply
    return cast(Func, wrapper)


Func2 = TypeVar('Func2', Callable[[int], Awaitable[int]], Callable[[str], Awaitable[int]])


def my_dec2(func: Func2) -> Func2:
    async def wrapper(value: Any) -> int:
        reply = await func(value)
        return reply  # error: Incompatible return value type (got "int", expected "Coroutine[Any, Any, int]")  [return-value]
    return cast(Func2, wrapper)