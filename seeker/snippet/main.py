#date: 2022-01-25T16:54:30Z
#url: https://api.github.com/gists/96008494b835a7ebd4d50ff724fa06da
#owner: https://api.github.com/users/mypy-play

from typing import Any, BinaryIO, Optional, overload, Protocol, SupportsInt, SupportsIndex, Type, TypeVar, Union


Self = TypeVar("Self")


# https://github.com/python/typeshed/blob/c2182fdd3e572a1220c70ad9c28fd908b70fb19b/stdlib/_typeshed/__init__.pyi#L68-L69
class SupportsTrunc(Protocol):
    def __trunc__(self) -> int: ...


ta = Union[str, bytes, SupportsInt, SupportsIndex, SupportsTrunc]
tb = Union[str, bytes, bytearray]

class C(int):
    # https://github.com/python/typeshed/blob/5d07ebc864577c04366fcc46b84479dbec033921/stdlib/builtins.pyi#L181-L185
    @overload
    def __new__(cls: Type[Self], __x: ta = ...) -> Self:
        ...

    @overload
    def __new__(cls: Type[Self], __x: tb, base: SupportsIndex) -> Self:
        ...

    def __new__(cls: Type[Self], __x: Union[None, ta, tb] = None, base: Optional[SupportsIndex] = None) -> Self:
        if __x is None:
            value = int()
        elif base is not None:
            value = int(__x, base)
        else:
            value = int(__x)
        return int.__new__(cls, value)
