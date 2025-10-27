#date: 2025-10-27T17:12:12Z
#url: https://api.github.com/gists/4f753938f810b5d2bf69ec78931fba0d
#owner: https://api.github.com/users/mypy-play

from typing import Self

class Foo:
    def __init__(self) -> None:
        self.instance: Self = self
