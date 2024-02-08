#date: 2024-02-08T16:51:56Z
#url: https://api.github.com/gists/1c294f1203ee61528b9b10e838f786aa
#owner: https://api.github.com/users/cainmagi

# -*- coding: UTF-8 -*-
"""
solve_overloads
===============

Author
------
Yuchen Jin
cainmagi@gmail.com

Description
-----------
Turn the `@overload` decorator from typehint-purposed to run-time effective.

The module provides a `solve_overloads` decorator which is specifically designed for
decorating function with overloads. The decorated function will be able to dispatch
the input arguments according to the overload versions.

The decorated function will try to bind the input arguments with each overload until
a binding is successful. The trial order of the overloads follows the same order of
the definition.
"""

import inspect
import functools

from typing import Any, TypeVar, cast
from typing_extensions import Protocol, get_overloads


class ArgsProtocol(Protocol):
    def __call__(self, *args: Any, **kwargs: Any) -> Any: ...


_ArgsProtocol = TypeVar("_ArgsProtocol", bound=ArgsProtocol)
__all__ = ("solve_overloads",)


def solve_overloads(func: _ArgsProtocol) -> _ArgsProtocol:
    """Decorator for branching overloads.

    This method can turn the typehint-purposed `@overload` decorator to an effective
    argument validator during the run time.

    This decorator is used for decorating any function with overloads. The decorator
    will find a wrapped function that tries to bind the provided arguments to each
    signature of the overloads until a succesful binding is done. If the arguments
    cannot be bounded to any overload, raise a `TypeError`.

    Arguments
    ---------
    func: `(*args: Any, **kwargs: Any) -> Any`
        The function to be bounded. The prototype of this function needs to contain
        `*args` and `**kwargs`. The function also needs to have at least one overload.

    Returns
    -------
    #1: `(*args: Any, **kwargs: Any) -> Any`
        The wrapped function. It has totally the same signature and the same
        functionality of the original function. However. After the decoration,
        the input arguments in the function implementation will be like this:
        ```python
        {
            "ver": int,
            "arguments": {*args, **kwargs}
        }
        ```
        Note that if the overload contains var arguments like `*args` or `**kwargs`,
        the values will be compressed as sequences or dictionaries, respectively.
        The value `ver` is the version of the overload.
        For example,
        ```python
        @overload
        def func(val: int): ...


        @overload
        def func(*vals: int, **kwvals: str): ...


        @solve_overloads
        def func(*args, **kwargs):
            print(args)
            print(kwargs)


        func(1, 2, 3, val="a", val2="b")
        ```
        The printed results will be
        ```python
        ()
        {
            "ver": 1,
            "arguments": {
                "vals": (1, 2, 3),
                "kwvals": {"val": "a", "val2": "b"}
            }
        }
        ```
    """

    @functools.wraps(func)
    def wrapped(*args, **kwargs):
        for idx_ver, func_ver in enumerate(get_overloads(func)):
            try:
                all_args = inspect.signature(func_ver).bind(*args, **kwargs)
            except TypeError:
                continue
            all_args.apply_defaults()
            return func(ver=idx_ver, arguments=all_args.arguments)
        raise TypeError(
            "The provided arguments do not match any overload of the function."
        )

    return cast(_ArgsProtocol, wrapped)
