#date: 2025-05-09T16:44:48Z
#url: https://api.github.com/gists/db22422d57b267d586bf92a0fd5929a4
#owner: https://api.github.com/users/mypy-play

from dataclasses import dataclass


@dataclass
class P:
    name: str | None


@dataclass
class Q(P):
    name: str


class A:
    def __init__(self) -> None:
        self._name: str | None = None

    @property
    def name(self) -> str | None:
        return self._name

    @name.setter
    def name(self, v: str | None) -> None:
        self._name = v

class B(A):
    def __init__(self) -> None:
        self._name = ""

    @property
    def name(self) -> str:
        assert self._name is not None
        return self._name

    @name.setter
    def name(self, v: str) -> None:
        self._name = v


def f(obj: P | A) -> None:
    obj.name = None


q = Q(name="")
b = B()
f(q)
f(b)