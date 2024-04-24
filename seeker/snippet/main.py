#date: 2024-04-24T17:11:31Z
#url: https://api.github.com/gists/150cd7a46baf6648dab1fe824853b9e4
#owner: https://api.github.com/users/mypy-play

from typing import TypeVar
from typing_extensions import TypeIs

_T = TypeVar('_T', str, int)

def is_str(val: str | int) -> TypeIs[str]:
    return isinstance(val, str)

def is_int(val: str | int) -> TypeIs[int]:
    return isinstance(val, int)

def process_str_int(data: _T) -> _T:
    if is_str(data):
        # At this point, `data` is narrowed down to `list[str]`
        print("Returning a string")
        return data
    elif is_int(data):
        print("Returning an int")
        return data
    raise # So that all branches return something

def process_str_int_with_isinstance(data: _T) -> _T:
    if isinstance(data, str):
        # At this point, `data` is narrowed down to `list[str]`
        print("Returning a string")
        return data
    elif isinstance(data, int):
        print("Returning an int")
        return data

process_str_int("hello")
