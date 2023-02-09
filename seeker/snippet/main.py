#date: 2023-02-09T16:48:45Z
#url: https://api.github.com/gists/6ad577cb649e7de7d7e03661ed91e3d6
#owner: https://api.github.com/users/mypy-play

from typing import Literal, overload

@overload
def f(name: Literal["integer"], value: int) -> int: ...

@overload
def f(name: Literal["string"], value: str) -> str: ...

def f(name: Literal["integer", "string"], value: int | str) -> int | str:
    if name == "integer":
        reveal_type(value)
        assert isinstance(value, int)
        reveal_type(value)

        def do_integer() -> int:
            reveal_type(value)  # XXX: Expected `builtins.int`
            return value        # XXX: Expected no error.

        reveal_type(value)

        return do_integer()

    if name == "string":
        reveal_type(value)
        assert isinstance(value, str)
        reveal_type(value)

        def do_string() -> str:
            reveal_type(value)  # XXX: Expected `builtins.str`
            return value        # XXX: Expected no error.

        reveal_type(value)

        return do_string()
