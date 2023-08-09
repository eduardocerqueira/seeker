#date: 2023-08-09T16:59:04Z
#url: https://api.github.com/gists/b5cdd2c383d488c2fb610dd850d4667c
#owner: https://api.github.com/users/mypy-play

from typing import Type, TypeVar

SomeType = TypeVar('SomeType')


def add_c_method_decorator(cls: Type[SomeType]) -> Type[SomeType]:
    return cls


@add_c_method_decorator
class TestClass:
    def a(self) -> int:
        return 1

    def b(self) -> str:
        return 'b'


reveal_type(TestClass)
