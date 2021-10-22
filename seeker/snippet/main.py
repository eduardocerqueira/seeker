#date: 2021-10-22T17:06:22Z
#url: https://api.github.com/gists/7a88942461df4adec2c3e90dd3def260
#owner: https://api.github.com/users/mypy-play

from typing import Union, TypeGuard

MultiTuple = Union[tuple[object], tuple[object, object]]

def is_double(a: tuple[object, ...]) -> TypeGuard[tuple[object, object]]:
    return len(a) == 2

def process_tuples(tup: MultiTuple) -> object:
    if not is_double(tup):
        return tup[0]
    else:
        first, second = tup
        return second
