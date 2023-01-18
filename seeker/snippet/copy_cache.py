#date: 2023-01-18T16:53:58Z
#url: https://api.github.com/gists/785a36f389c5a409cbbc798ac46edebb
#owner: https://api.github.com/users/clbarnes

from functools import lru_cache, wraps
from typing import Optional, Callable
from copy import copy, deepcopy


def copy_cache(deep: bool = True, maxsize: Optional[int] = 128, typed: bool = False):
    """Decorator factory wrapping functools.lru_cache, which copies return values.

    Use this for functions with mutable return values.

    N.B. must be called with parentheses.

    Parameters
    ----------
    deep : bool, optional
        Whether to deepcopy return values, by default True
    maxsize : Optional[int], optional
        See lru_cache, by default 128.
        Use None for an unbounded cache (which is also faster).
    typed : bool, optional
        See lru_cache, by default False

    Returns
    -------
    Callable[[Callable], Callable]
        Returns a decorator.
        
    Examples
    --------
    >>> @copy_cache()  # must include parentheses, unlike some zero-arg decorators
    ... def my_function(key: str, value: int) -> dict:
    ...     return {key: value}
    ...
    >>> d1 = my_function("a", 1)
    >>> d1["b"] = 2
    >>> d2 = my_function("a", 1)
    >>> "b" in d2  # would be True with raw functools.lru_cache
    False
    """
    copier = deepcopy if deep else copy

    def wrapper(fn):

        wrapped = lru_cache(maxsize, typed)(fn)

        @wraps(fn)
        def copy_wrapped(*args, **kwargs):
            out = wrapped(*args, **kwargs)
            return copier(out)

        copy_wrapped.cache_info = wrapped.cache_info
        copy_wrapped.cache_clear = wrapped.cache_clear

        return copy_wrapped

    return wrapper
