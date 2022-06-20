#date: 2022-06-20T17:12:07Z
#url: https://api.github.com/gists/56959b409df6a1e03935ad17aaabf07b
#owner: https://api.github.com/users/mypy-play

from typing import TypeVar, Union, NoReturn, Any

class FooException:
    pass

class BarException:
    pass

def foo() -> Union[int,  FooException]:
    return 1
    

def bar() -> int:
    x = foo()
    if isinstance(x, int):
        return 4
    elif isinstance(x, FooException):
        return 4
    elif isinstance(x, BarException):
        return 4