#date: 2024-08-21T17:11:12Z
#url: https://api.github.com/gists/f83671bbbc02711ac90962ae786b418b
#owner: https://api.github.com/users/mypy-play

from typing import Union, Type, Tuple, TypeVar

T = TypeVar("T")


def some_types(type_: Union[Type[T], Tuple[Type[T], ...]]) -> T:
    ...

def int_str_float_fn(a: int | str | float) -> None:
    ...

a = some_types((int, str, float))

int_str_float_fn(a)  # should't be an error
