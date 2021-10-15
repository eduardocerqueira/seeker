#date: 2021-10-15T17:13:21Z
#url: https://api.github.com/gists/3aa913dd984e654b8f0f230736f24996
#owner: https://api.github.com/users/mypy-play

from collections import UserDict
from collections.abc import Mapping
from typing import TypeVar, Optional

K = TypeVar('K')
V = TypeVar('V')

class TypeCheckedDict(UserDict[K, V]):
    def __init__(
        self, 
        key_type: type[K], 
        value_type: type[V], 
        initdict: Optional[Mapping[K, V]] = None
    ) -> None:
        self._key_type = key_type
        self._value_type = value_type
        super().__init__(initdict)

    def __setitem__(self, key: K, value: V) -> None:
        if not isinstance(key, self._key_type):
            raise TypeError(
                f'Invalid type for dictionary key: '
                f'expected "{self._key_type.__name__}", '
                f'got "{type(key).__name__}"'
            )
        if not isinstance(value, self._value_type):
            raise TypeError(
                f'Invalid type for dictionary value: '
                f'expected "{self._value_type.__name__}", '
                f'got "{type(value).__name__}"'
            )
        super().__setitem__(key, value)