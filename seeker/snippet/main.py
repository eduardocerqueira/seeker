#date: 2023-07-31T16:55:27Z
#url: https://api.github.com/gists/77c5b76a5c075b1459c78cea898b05ff
#owner: https://api.github.com/users/mypy-play

from abc import ABC, abstractmethod
from typing import Type


class SomeABC(ABC):
    name: str

    @abstractmethod
    def something(self) -> bool:
        pass


class Concrete(SomeABC):
    name = "concrete"
    
    def something(self) -> bool:
        return True


class Concrete2(SomeABC):
    name = "concrete2"
    
    def something(self) -> bool:
        return False

    
classes: dict[str, Type[SomeABC]] = {
    cls.name: cls
    for cls in (Concrete, Concrete2)
}
