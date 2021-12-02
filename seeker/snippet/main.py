#date: 2021-12-02T16:47:26Z
#url: https://api.github.com/gists/d5c65cf0752e54b72252256aff9de8f7
#owner: https://api.github.com/users/mypy-play

from typing_extensions import Literal
from typing import Tuple, Union
from typing import overload



@overload
def bar(name: str, *, return_length: Literal[True] = ...) -> Tuple[str, int]:
    ...

@overload
def bar(name: str, *, return_length: Literal[False]) -> str:
    ...
    
@overload
def bar(name: str, *, return_length: bool = ...) -> Union[str, Tuple[str, int]]:
    ...

def bar(name: str, *, return_length: bool = True) -> Union[str, Tuple[str, int]]:
    if return_length:
        return name, len(name)
    else:
        return name


x = bar("anthonk", return_length=True)
reveal_type(x)  # Revealed type is "Tuple[builtins.str, builtins.int]" (as expected)
y = bar("anthonk", return_length=False)
reveal_type(y)  # Revealed type is "builtins.str" (as expected)


@overload
def baz(name: str, *, return_length: Literal[True] = ...) -> Tuple[str, int]:
    ...

@overload
def baz(name: str, *, return_length: Literal[False]) -> str:
    ...
    
@overload
def baz(name: str, *, return_length: bool) -> Union[str, Tuple[str, int]]:
    ...

def baz(name: str, *, return_length: bool = True) -> Union[str, Tuple[str, int]]:
    new_name = name.upper()
    result = bar(new_name, return_length=return_length)  # mypy throws this : No overload variant of "bar" matches argument types "str", "bool"
    return result