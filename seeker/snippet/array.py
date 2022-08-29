#date: 2022-08-29T16:57:41Z
#url: https://api.github.com/gists/49c6f325eefe3bfe9d274584fa62578d
#owner: https://api.github.com/users/dot1mav

from dataclasses import dataclass, field
from typing import Any, Union, TypeVar, Literal, Tuple, List

T = TypeVar("T")
null = TypeVar("null", None, Literal["null", "Null", "NULL"])


def size_check(_size: int, _id: int) -> None:
    if not isinstance(_id, int):
        raise TypeError("index type of array should be int")

    if _id < 0:
        raise ValueError(
            f"array index should be in range {0!r} ... {(_size-1)!r}")

    if _size - 1 < _id:
        raise ValueError("array index more than size of array")


def type_check(_type: Any, value: Any) -> None:
    if not isinstance(value, _type):
        raise TypeError(
            f"data type should be {_type!r} youre data is {type(value)!r}")


@dataclass(frozen=True)
class Array:
    """
        make static array like c for you
    """
    _size: int = field(compare=False)
    _type: Any = field(default=int)
    _data: list = field(default_factory=list, init=False,
                        repr=False, compare=True)

    def __post_init__(self) -> None:
        object.__setattr__(self, "_data", [null] * self._size)

    def __setitem__(self, key: int, value: Any) -> None:
        size_check(self._size, key)
        type_check(self._type, value)

        self._data[key] = value

    def __getitem__(self, key: int) -> Union[T, null]:
        size_check(self._size, key)

        return self._data[key]

    def __len__(self) -> int:
        return len(list(filter(lambda x: x != null, self._data)))

    def r_shift(self) -> None:
        for i, j in zip(range(self._size-1, -1, -1), range(self._size-2, -1, -1)):
            self._data[i], self._data[j] = self._data[j], self._data[i]

    def l_shift(self) -> None:
        for i, j in zip(range(0, self._size), range(1, self._size)):
            self._data[i], self._data[j] = self._data[j], self._data[i]

    @property
    def array(self) -> List[Union[T, null]]:
        return self._data.copy()

    def insert(self, value: T, _id: int = 0) -> None:
        size_check(self._size, _id)
        type_check(self._type, value)

        self._data[_id] = value

    def remove(self, value: T) -> Union[T, null]:
        arr = list(map(lambda x: x == value, self._data))
        if any(arr):
            i = arr.index(True)
            v = self._data[i]
            self._data[i] = null
            return v
        return null

    def pop(self) -> Tuple[Union[int, null], Union[T, null]]:
        index, value = None, None

        for i in range(self._size-1, -1, -1):
            if self._data[i] != null:
                index, value = (i, self._data[i])
                break

        if index and value:
            self._data[index] = null
            return index, value

        return (null, null)

    def search(self, value: T) -> bool:
        type_check(self._type, value)

        return any(map(lambda x: x == value, self._data))
