#date: 2022-04-01T16:50:16Z
#url: https://api.github.com/gists/77efe095d01edeb1a4614166f0c9cf68
#owner: https://api.github.com/users/mypy-play

from types import TracebackType
from typing import Any
from typing_extensions import Protocol

class AbstractFoo(Protocol):
    def __exit__(self, typ: type[BaseException] | None, exc: BaseException | None, tb: TracebackType | None) -> None: ...

class ConcreteFoo(AbstractFoo):
    def __exit__(self, *args: object) -> None:
        print("Hi")

f: AbstractFoo = ConcreteFoo()