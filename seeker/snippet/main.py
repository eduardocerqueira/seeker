#date: 2021-12-17T16:57:31Z
#url: https://api.github.com/gists/bef85df6def2bb296db4fa310cc592cc
#owner: https://api.github.com/users/mypy-play

from typing import TypeVar

T = TypeVar("T", int, str)

def prints_twice(t: T) -> None:
    reveal_type(t)