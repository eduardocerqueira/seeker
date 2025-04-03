#date: 2025-04-03T17:12:38Z
#url: https://api.github.com/gists/3ad72d5595fa743cc2d41178a1c24e61
#owner: https://api.github.com/users/mypy-play

from typing import Self


class Klass:
    def list(self) -> list[int]:
        raise NotImplementedError

    @classmethod
    def from_list(cls, items: list[int]) -> Self:
        raise NotImplementedError
