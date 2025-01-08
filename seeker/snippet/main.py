#date: 2025-01-08T17:00:44Z
#url: https://api.github.com/gists/abb729c2b41bf64c0a58f72214e27350
#owner: https://api.github.com/users/mypy-play

from typing import TypeVar, Callable, ParamSpec, Concatenate, cast
from functools import wraps
import inspect

class Hub: ...

default_hub = Hub()

P = ParamSpec('P')
R = TypeVar('R')


def coroutine_t(f: Callable[Concatenate[Hub, P], R]) -> Callable[P, R]:
    def inner(*args: P.args, **kwargs: P.kwargs) -> R:
        return f(default_hub, *args, **kwargs)
    return inner

@coroutine_t
def sum_as_coro(hub: Hub, numbers: list[float]) -> float:
    # Use hub...
    return sum(numbers)

sum_as_coro([1.1, 2.2, 3.3])
reveal_type(sum_as_coro)  # Revealed type is "def (numbers: builtins.list[builtins.float]) -> builtins.float"