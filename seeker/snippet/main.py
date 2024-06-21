#date: 2024-06-21T16:46:32Z
#url: https://api.github.com/gists/d16da1574497dbe585eab679f09432ef
#owner: https://api.github.com/users/mypy-play

from dataclasses import dataclass

@dataclass
class A:
    thing: list[str]


A(thing=[])



