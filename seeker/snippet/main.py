#date: 2023-12-14T16:49:28Z
#url: https://api.github.com/gists/c8126557ae0349640416893479c0292a
#owner: https://api.github.com/users/mypy-play

from typing import Iterator, TypeVar,Generic,Callable,Optional,Type, overload, Any, Generic, Sequence
from typing_extensions import TypeVarTuple, ParamSpec

Ts = TypeVarTuple("Ts")
P = ParamSpec("P")

class Foo(Generic[*Ts]):
    CallableType = Callable[[*Ts], None]

    def __call__(self) -> None:  # type:ignore[override]
        ...

    def Register(
        self,
        func: CallableType,
        extra_args: Sequence[object] = (),
    ) -> Any:
        ...

    def Unregister(
        self,
        func: CallableType,
        extra_args: Sequence[object] = (),
    ) -> Any:
        ...

    def Contains(
        self,
        func: CallableType,
        extra_args: Sequence[object] = (),
    ) -> Any:
        ...