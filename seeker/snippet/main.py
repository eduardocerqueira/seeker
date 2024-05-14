#date: 2024-05-14T16:55:07Z
#url: https://api.github.com/gists/523713f57cc341de1593dca3b96658df
#owner: https://api.github.com/users/mypy-play

from enum import Enum

class MyStrEnum(str, Enum):
    A = "a"  # ok
    B = b"b", "utf-8"  # ok, see the typeshed definition of `str.__new__`
    C = "too", "many", "arguments", "provided"  # runtime error: TypeError: str() takes at most 3 arguments (4 given)
