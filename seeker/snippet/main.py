#date: 2023-05-03T16:51:45Z
#url: https://api.github.com/gists/a590ecd40364183b563b8d2ab83f5bd9
#owner: https://api.github.com/users/mypy-play

from typing import *

class Timestamp: ...

@overload
def store_expiry(upstream_name: str, expiry: Optional[Timestamp], /):  # the `/` is important, otherwise mypy complains that the implementation doesn't accept it
    ...

@overload
def store_expiry(upstream_name: str, min_expiry: Optional[Timestamp] = ..., max_expiry: Optional[Timestamp] = ...):
    ...

def store_expiry(
    upstream_name: str,
    min_expiry: Optional[Timestamp] = None,
    max_expiry: Optional[Timestamp] = None,
):
    pass


min_max = {"min": ..., "max": ...}

store_expiry(
    "dataset",
    cast(Optional[Timestamp], min_max["min"]),
    cast(Optional[Timestamp], min_max["max"]),
)