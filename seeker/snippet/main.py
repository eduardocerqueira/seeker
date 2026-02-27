#date: 2026-02-27T17:23:47Z
#url: https://api.github.com/gists/ccbb398c6b68a7a954220e29a1e35d71
#owner: https://api.github.com/users/AlexWaygood

from typing import TypeVar, Generic

T = TypeVar("T")
U = TypeVar("U", default=T)












class Foo(Generic[U, T]):
    x: U
    y: T
