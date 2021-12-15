#date: 2021-12-15T17:10:05Z
#url: https://api.github.com/gists/73c6e9c687d099064ea375f5a8985678
#owner: https://api.github.com/users/mypy-play

from typing import Optional, TypeVar, Union


T = TypeVar("T")


def none_check_v1(param: Optional[T], result: bool = True) -> T:
    test_condition = param is not None if result else param is None
    if test_condition:
        raise ValueError(f"Param ({param}) must{' ' if result else ' not '}be None")

    return param
    
    

def none_check_v2(param: Optional[T], result: bool = True) -> Union[T, None]:
    test_condition = param is not None if result else param is None
    if test_condition:
        raise ValueError(f"Param ({param}) must{' ' if result else ' not '}be None")

    return param
    
    

arg: Optional[int] = 1

p0 = none_check_v2(arg)
p1 = none_check_v2(arg, True)
p2 = none_check_v2(arg, False)

p0 - 1
p1 - 1
p2 - 1