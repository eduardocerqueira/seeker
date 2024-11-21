#date: 2024-11-21T16:58:41Z
#url: https://api.github.com/gists/2ac3079d8938fa160857862bd1178f4b
#owner: https://api.github.com/users/mypy-play

from typing import Any
from collections.abc import Callable
import inspect


class CallableClass:
    def __init__(self) -> None:
        pass
    
    def __call__(self, *args: Any, **kwargs: Any) -> None:
        return None



assert callable(CallableClass) # Success


callable_instance = CallableClass()
assert isinstance(callable_instance, Callable[..., None]) # Errors
assert isinstance(callable_instance, Callable) # Also erros