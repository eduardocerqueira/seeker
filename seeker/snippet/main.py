#date: 2024-12-06T16:53:15Z
#url: https://api.github.com/gists/ab69360a59973d779645203c39d076f1
#owner: https://api.github.com/users/mypy-play

from __future__ import annotations

class Bar: ...

class Foo:
    x: "Bar" = Bar()

    class Bar: ...
    y: "Bar" = Bar()

reveal_type(Foo.x)  # __main__.Bar
reveal_type(Foo.y)  # __main__.Foo.Bar
