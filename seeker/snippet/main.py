#date: 2025-09-16T16:44:34Z
#url: https://api.github.com/gists/548f06df05b51868e125f8cfd4a15919
#owner: https://api.github.com/users/mypy-play

from typing import Tuple
from typing_extensions import TypeVarTuple, Unpack

Ts = TypeVarTuple("Ts")
def foo(arg: Tuple[int, Unpack[Ts], str]) -> None:
    x = *arg,
    reveal_type(x)  # N: Revealed type is "tuple[builtins.int, Unpack[Ts`-1], builtins.str]"
    y = 1, *arg, 2
    reveal_type(y)  # N: Revealed type is "tuple[builtins.int, builtins.int, Unpack[Ts`-1], builtins.str, builtins.int]"
    z = (*arg, *arg)
    reveal_type(z)  # N: Revealed type is "tuple[builtins.int, Unpack[builtins.tuple[Union[Any, builtins.str, builtins.int], ...]], builtins.str]"
