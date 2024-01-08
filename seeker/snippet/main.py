#date: 2024-01-08T16:53:06Z
#url: https://api.github.com/gists/1bb8a85936486c0c182376ebf36f62b5
#owner: https://api.github.com/users/mypy-play

from typing import TypeVar

Self = TypeVar("Self", bound="A")

class A:
    def __init__(self: Self, other: type[Self]):
        return


reveal_type(A(A))
