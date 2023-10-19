#date: 2023-10-19T16:59:45Z
#url: https://api.github.com/gists/7c5398428f526b552b130cea036383b1
#owner: https://api.github.com/users/mypy-play

from typing import (
    Awaitable,
    Callable,
    TypeVar,
    Union,
    Generic,
    Optional,
    overload,
)

ProtocolParamType = TypeVar("ProtocolParamType", contravariant=True)
ProtocolReturnType = TypeVar("ProtocolReturnType", covariant=True)
ProtocolSelfType = TypeVar("ProtocolSelfType", contravariant=True)


CDecFunc = Callable[
    [ProtocolSelfType, ProtocolParamType],
    Union[ProtocolReturnType, Awaitable[ProtocolReturnType]],
]


class CallableDec(Generic[ProtocolSelfType, ProtocolParamType, ProtocolReturnType]):
    def __init__(
        self,
        fn: CDecFunc,
        *,
        name: Optional[str] = None,
    ):
        self._fn = fn
        self._name = name

    def __call__(
        __self, self: ProtocolSelfType, __param: ProtocolParamType
    ) -> Union[ProtocolReturnType, Awaitable[ProtocolReturnType]]:
        return __self._fn(self, __param)


@overload
def FnDecorator(
    fn: CDecFunc,
) -> CallableDec[ProtocolSelfType, ProtocolParamType, ProtocolReturnType]:
    ...


@overload
def FnDecorator(
    *, name: str
) -> Callable[
    [CDecFunc],
    CallableDec[ProtocolSelfType, ProtocolParamType, ProtocolReturnType],
]:
    ...


def FnDecorator(
    fn: Optional[CDecFunc] = None,
    *,
    name: Optional[str] = None,
) -> Union[
    CallableDec[ProtocolSelfType, ProtocolParamType, ProtocolReturnType],
    Callable[
        [CDecFunc],
        CallableDec[ProtocolSelfType, ProtocolParamType, ProtocolReturnType],
    ],
]:
    print("Decorating with", fn, name)
    if fn is not None:
        return CallableDec(fn, name=name)

    def wrapper(
        calledfn: CDecFunc,
    ) -> CallableDec[ProtocolSelfType, ProtocolParamType, ProtocolReturnType]:
        print("Inside wrapper")
        return CallableDec(calledfn, name=name)

    return wrapper


class MyClass:
    @FnDecorator
    async def some_fun(self, arg: int) -> str:
        return "hi"

    @FnDecorator(name="hi")
    async def some_fun_named(self, arg: int) -> str:
        return "hi"

    @FnDecorator
    def some_sync_fun(self, arg: int) -> str:
        return "hisync"
