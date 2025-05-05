#date: 2025-05-05T16:37:08Z
#url: https://api.github.com/gists/7e7edfa1d3debec48847590a6c4532c9
#owner: https://api.github.com/users/mypy-play

class Foo:
    value: bool

    def __init__(self, value: bool) -> None:
        self.thing = value


def bad(maybe: bool) -> None:
    foo = None
    if maybe and (foo := Foo(True)).value:
        reveal_type(foo)


def correct1(maybe: bool) -> None:
    foo = None
    if maybe and (foo := Foo(True)):
        reveal_type(foo)


def correct2(maybe: bool) -> None:
    foo = None
    if maybe:
        if (foo := Foo(True)).value:
            reveal_type(foo)
