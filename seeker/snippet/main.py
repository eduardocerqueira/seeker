#date: 2021-12-10T17:07:57Z
#url: https://api.github.com/gists/1cbb7ab4a946e9b043218e7b1300694b
#owner: https://api.github.com/users/mypy-play

from typing import Any, Awaitable, Callable, Generic, TypeVar, Union, overload

R = TypeVar("R")

ContravariantT = TypeVar("ContravariantT", contravariant=True)

class Deferred(Awaitable[ContravariantT]):
    pass

@overload
def run_in_background(  # type: ignore[misc]
    f: Callable[..., Awaitable[R]], *args: Any, **kwargs: Any
) -> "Deferred[R]":
    # The `type: ignore[misc]` above suppresses
    # "Overloaded function signatures 1 and 2 overlap with incompatible return types"
    ...

@overload
def run_in_background(
    f: Callable[..., R], *args: Any, **kwargs: Any
) -> "Deferred[R]":
    ...

def run_in_background(
    f: Union[
        Callable[..., R],
        Callable[..., Awaitable[R]],
    ],
    *args: Any,
    **kwargs: Any,
) -> "Deferred[R]":
    pass

def sync() -> int: pass
async def async_async() -> str: pass
def async_awaitable() -> Awaitable[float]: pass
def async_deferred() -> "Deferred[bool]": pass

reveal_type(run_in_background(sync))
reveal_type(run_in_background(async_async))
reveal_type(run_in_background(async_awaitable))
reveal_type(run_in_background(async_deferred))
