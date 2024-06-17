#date: 2024-06-17T16:51:12Z
#url: https://api.github.com/gists/c9ed0c92f2ac053a18d36e83c16e6b20
#owner: https://api.github.com/users/mypy-play

from abc import ABC
from collections.abc import Mapping, Sequence
from typing import Protocol, TypeAlias


Json_ro: TypeAlias = str | int | bool | Mapping[str, Json_ro] | Sequence[Json_ro]


class HasJson(Protocol):
    def to_json(self) -> Json_ro:
        ...


class MyAbc(ABC):
    foo: str
    
    # This still conforms to HasJson because Mapping[str, Json_ro] is a subtype
    # of Json_ro
    def to_json(self) -> Mapping[str, Json_ro]:
        return {
            "foo": self.foo,
        }


class MyConcrete(MyAbc):
    bar: int
    
    def __init__(self, foo: str, bar: int) -> None:
        self.foo = foo
        self.bar = bar
    
    def to_json(self) -> Mapping[str, Json_ro]:
        return {**super().to_json(), "bar": self.bar}


def to_json(obj: HasJson) -> Json_ro:
    return obj.to_json()
    
    
my_concrete = MyConcrete("hello", 1)

print(to_json(my_concrete))