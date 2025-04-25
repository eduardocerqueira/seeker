#date: 2025-04-25T16:45:32Z
#url: https://api.github.com/gists/ed85394f44de0677ea5079ef0c388ba6
#owner: https://api.github.com/users/mypy-play

from collections.abc import Callable


class classproperty[T, C]:
    def __init__(self, func: Callable[[type[C]], T]):
        self.func = func

    def __get__(self, _obj: C, owner: type[C]) -> T:
        return self.func(owner)

class Test:
    @classproperty
    def my_property(cls):
        return 1
