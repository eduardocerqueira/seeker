#date: 2025-04-14T16:49:51Z
#url: https://api.github.com/gists/2a6a02924367c8b1e1241309a6744f95
#owner: https://api.github.com/users/mypy-play

from typing import Generic, TypeVar



T = TypeVar("T")


class Foo(Generic[T]):
    @staticmethod
    def hehe(v: T) -> T:
        return v


Rst = TypeVar("Rst", str, int)

class Fee(Foo[Rst]):
    pass

fee = Foo[Rst]


reveal_type(Fee.hehe((1,1)))

