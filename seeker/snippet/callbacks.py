#date: 2026-02-10T17:54:40Z
#url: https://api.github.com/gists/d5b740e0f0371903f66ce60ace102a41
#owner: https://api.github.com/users/mcshlain

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable, Sequence, overload


@dataclass(frozen=True, slots=True)
class Callback[S]:
    start: Callable[[], S]
    end: Callable[[S], None]

    @overload
    def __add__[S2](self, other: Callback[S2]) -> MultipleCallbacks[S, S2]: ...

    @overload
    def __add__[*S2](self, other: MultipleCallbacks[*S2]) -> MultipleCallbacks[S, *S2]: ...

    def __add__(self, other: MultipleCallbacks | Callback) -> MultipleCallbacks:
        return _combine(self, other)


@dataclass(frozen=True, slots=True)
class MultipleCallbacks[*S]:
    subcallbaks: Sequence[Callback[Any]]

    def start(self) -> tuple[*S]:
        return tuple(s.start() for s in self.subcallbaks)

    def end(self, *args: *S) -> None:
        for callback, arg in zip(self.subcallbaks, args, strict=True):
            callback.end(arg)

    @overload
    def __add__[S2](self, other: Callback[S2]) -> MultipleCallbacks[*S, S2]: ...

    @overload
    def __add__[*S2](self, other: MultipleCallbacks[*S2]) -> MultipleCallbacks[*S, *S2]: ...

    def __add__(self, other: MultipleCallbacks | Callback) -> MultipleCallbacks:
        return _combine(self, other)


type Callbacks = Callback[Any] | MultipleCallbacks[Any]


def _combine(c1: MultipleCallbacks | Callback, c2: MultipleCallbacks | Callback) -> MultipleCallbacks:
    match (c1, c2):
        case (Callback(), Callback()):
            return MultipleCallbacks([c1, c2])
        case (MultipleCallbacks(), Callback()):
            return MultipleCallbacks([*c1.subcallbaks, c2])
        case (Callback(), MultipleCallbacks()):
            return MultipleCallbacks([c1, *c2.subcallbaks])
        case (MultipleCallbacks(), MultipleCallbacks()):
            return MultipleCallbacks([*c1.subcallbaks, *c2.subcallbaks])


def _start1() -> int:
    print("start1")
    return 8


def _end1(a: int) -> None:
    print(f"end1 {a}")


def _start2() -> str:
    print("start2")
    return "hello"


def _end2(b: str) -> None:
    print(f"end2 {b}")


def _start3() -> bool:
    print("start3")
    return False


def _end3(c: bool) -> None:
    print(f"end3 {c}")


if __name__ == "__main__":
    c1 = Callback(_start1, _end1)
    c2 = Callback(_start2, _end2)
    c3 = Callback(_start3, _end3)

    z1 = c1 + c2
    z2 = z1 + c3

    s = z2.start()
    z2.end(*s)
