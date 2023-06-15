#date: 2023-06-15T16:46:16Z
#url: https://api.github.com/gists/83bafb47437a3ae3eb750d305532f828
#owner: https://api.github.com/users/mypy-play

from typing import Type, TypeVar

class B:
    pass
class C:
    pass
class D1(C, B):
    pass
class D2(C, B):
    pass

T = TypeVar("T", bound=B)

def fn(a: Type[T]) -> None:
    pass

val: Type[D1] | Type[D2]
fn(val)