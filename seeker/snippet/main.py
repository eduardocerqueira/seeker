#date: 2022-09-02T17:10:15Z
#url: https://api.github.com/gists/950a76f3f878c7b9e1cf4d0d02d6cffc
#owner: https://api.github.com/users/mypy-play

from collections.abc import Callable
from typing import TypeVar, ParamSpec, overload, Any, Union, Optional
import random
import logging

R = TypeVar("R", bound=Callable[..., Any])

FuncReturn = TypeVar("FuncReturn")
RedirectReturn = TypeVar("RedirectReturn")

P = ParamSpec('P')

@overload
def call_when_enabled(feature_name: str, redirect: Callable[..., RedirectReturn]) -> Callable[[Callable[P, FuncReturn]], Callable[P, Union[RedirectReturn, FuncReturn]]]:
    ...
@overload
def call_when_enabled(feature_name: str, redirect: None = None) -> Callable[[Callable[P, FuncReturn]], Callable[P, Optional[FuncReturn]]]:
    ...
def call_when_enabled(feature_name: str, redirect: Optional[Callable[..., RedirectReturn]] = None) -> Callable[[Callable[P, FuncReturn]], Callable]:
    def func_decorator(func: Callable[P, FuncReturn]) -> Callable:
        return func

    return func_decorator
    

def double(x: int) -> str:
    return str(2*x)

@call_when_enabled("feature", redirect=double)
def new_double(x: int) -> int:
    return x + x
    

z = new_double(4)