#date: 2023-09-06T16:56:33Z
#url: https://api.github.com/gists/d0956f53649322c51e5b2a3b00013dbf
#owner: https://api.github.com/users/mypy-play

from typing import Callable, ParamSpec, TypeVar, cast, Any, Type, Literal, Concatenate

# Define some specification, see documentation
P = ParamSpec("P")
T = TypeVar("T")

# For a help about decorator with parameters see 
# https://stackoverflow.com/questions/5929107/decorators-with-parameters
def copy_kwargs_int(kwargs_call: Callable[P, Any]) -> Callable[[Callable[..., T]], Callable[P, T]]:
    """Decorator does nothing but returning the casted original function"""
    def return_func(func: Callable[..., T]) -> Callable[P, T]:
        return cast(Callable[P, T], func)

    return return_func

# Our test function
def source_func(foo: str, bar: int, default: bool = True) -> str:
    if not default:
        return "Not Default!"
    return f"{foo}_{bar}"

def copy_kwargs_with_int(
    kwargs_call: Callable[P, Any]
) -> Callable[[Callable[..., T]], Callable[Concatenate[int, P], T]]:
    """Decorator does nothing but returning the casted original function"""

    def return_func(func: Callable[..., T]) -> Callable[Concatenate[int, P], T]:
        return cast(Callable[Concatenate[int, P], T], func)

    return return_func

@copy_kwargs_with_int(source_func)
def something(first: int, *args, **kwargs) -> str:
    print(f"Yeah {first}")
    return str(source_func(*args, **kwargs))

something("a", "string", 3) # error: Argument 1 to "something" has incompatible type "str"; expected "int"  [arg-type]
okay_call: str
okay_call = something(3, "string", 3) # okay

