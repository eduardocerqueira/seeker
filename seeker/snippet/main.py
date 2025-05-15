#date: 2025-05-15T16:51:50Z
#url: https://api.github.com/gists/effd2ed9e8970ca1698c58235b6e579b
#owner: https://api.github.com/users/mypy-play

from typing import Any, Callable


class A:
    def __call__(self) -> Any: ...

class B:
    def __call__(self) -> Any: ...


test_a: set[Callable[..., Any]] = {A(), B()}  # passes
test_b: set[Callable[..., Any]] = {A(), B()} | {A(), B()}  # fails
test_c: list[Callable[..., Any]] = [A(), B()]  # passes
test_d: list[Callable[..., Any]] = [A(), B()] + [A(), B()]  # fails