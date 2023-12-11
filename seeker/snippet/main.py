#date: 2023-12-11T16:51:55Z
#url: https://api.github.com/gists/c451dedbb5ec867b98fcfec6aa24d698
#owner: https://api.github.com/users/mypy-play

from __future__ import annotations
from typing import overload, Callable, TypeVar, Coroutine, cast

from typing_extensions import ParamSpec, reveal_type

P = ParamSpec("P")
T = TypeVar("T")


@overload
def decorator(
    to_be_decorated: Callable[P, Coroutine[None, None, T]],
    *,
    some_boolean: bool = False,
) -> Callable[P, T]:
    ...


@overload
def decorator(
    to_be_decorated: Callable[P, T],
    *,
    some_boolean: bool = False,
) -> Callable[P, T]:
    ...


@overload
def decorator(
    to_be_decorated: None = None,
    *,
    some_boolean: bool = False,
) -> Callable[[Callable[P, Coroutine[None, None, T]] | Callable[P, T]], Callable[P, T]]:
    ...


def decorator(
    to_be_decorated: Callable[P, Coroutine[None, None, T]]
    | Callable[P, T]
    | None = None,
    *,
    some_boolean: bool = False,
) -> (
    Callable[P, T]
    | Callable[[Callable[P, Coroutine[None, None, T]] | Callable[P, T]], Callable[P, T]]
):
    def inner_decorator(
        to_be_decorated: Callable[P, Coroutine[None, None, T]] | Callable[P, T]
    ) -> Callable[P, T]:
        def wrapped(*args: P.args, **kwargs: P.kwargs) -> T:
            if some_boolean:
                print(f"Calling {to_be_decorated.__name__}")
            return cast("Callable[P, T]", to_be_decorated)(*args, **kwargs)

        return wrapped

    if to_be_decorated:
        return inner_decorator(to_be_decorated)
    else:
        return inner_decorator


@decorator(some_boolean=False)
async def f() -> bool:
    return False


reveal_type(f)
# Pylance reveals () -> bool
# MyPy reveals (*Never, **Never) -> Never
# MyPy also complains on decorator application in line 63.
