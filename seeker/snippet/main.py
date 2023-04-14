#date: 2023-04-14T16:58:39Z
#url: https://api.github.com/gists/fdd3d72f616b8ce786c6c7f8bc2a79fd
#owner: https://api.github.com/users/mypy-play

from typing import *
import enum


class EnumType(enum.EnumMeta):
    _value_map_: Mapping[Any, Enum]
    _member_map_: Mapping[str, Enum]


class Enum(enum.Enum, metaclass=EnumType):
    _value_map_: Mapping[Any, Self]
    _member_map_: Mapping[str, Self]
    

class E(Enum):
    pass


x = E._member_map_
