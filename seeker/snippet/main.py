#date: 2024-12-04T16:56:44Z
#url: https://api.github.com/gists/2e782bb6132502e0c5dffd6bb06ac6cb
#owner: https://api.github.com/users/mypy-play

from typing import Callable, Sequence, Any, MutableSequence

def f(x: list[int]) -> None:
    x[0] / 10

def g(x: list[str]) -> None:
    x[0].startswith("a")

def do_it(fn: Callable[[list[Any]], None]) -> None:
    fn([1])

do_it(f)
do_it(g)
