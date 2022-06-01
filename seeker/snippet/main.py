#date: 2022-06-01T17:18:41Z
#url: https://api.github.com/gists/87a99442d9d00dae3743fdfb7526a8c1
#owner: https://api.github.com/users/mypy-play

from typing import Literal, Union

from enum import Enum

class BenchlingStrEnum(str, Enum):
    def __str__(self) -> str:
        return str(self.value)

    @classmethod
    def values(cls) -> list[str]:
        return [t.value for t in cls]
        
class MyEnum(BenchlingStrEnum):
    foo = "foo"
    bar = "bar"
    baz = "baz"


MySubEnum = Union[
    Literal[MyEnum.foo],
    Literal[MyEnum.bar],
]

def do_thing(value: MySubEnum) -> None:
    pass

do_thing(MyEnum.foo)
do_thing(MyEnum.bar)
do_thing(MyEnum.baz)