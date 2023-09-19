#date: 2023-09-19T17:06:31Z
#url: https://api.github.com/gists/f12b2f4fde87a2c9511d59b5cd7d919f
#owner: https://api.github.com/users/mypy-play

from typing import Callable, TypeVar, Any
from typing_extensions import Concatenate, ParamSpec

P = ParamSpec('P')
T = TypeVar('T')
F = TypeVar('F', bound=Callable[..., Any])

def decorator_factory(num: int) -> Callable[[F], F]:
    def decorator(func: Callable[P, T]) -> Callable[P, T]:
        def wrapper(*args: P.args, **kwds: P.kwargs) -> T:
            return func(*args, **kwds)
        return wrapper
    return decorator

@decorator_factory(3)
def add_forty_two(value: int) -> int:
    return value + 42

a = add_forty_two(3)
reveal_type(add_forty_two)
reveal_type(a)