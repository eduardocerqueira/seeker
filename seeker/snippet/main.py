#date: 2024-05-31T17:02:42Z
#url: https://api.github.com/gists/8816afb67fc8676708c660b7e46dc312
#owner: https://api.github.com/users/mypy-play

from typing import assert_type, Any, Optional, TypeVar, overload

T = TypeVar("T")

@overload
def get(key: str) -> Optional[Any]: ...

@overload
def get(key: str, default: Any | T) -> Any | T: ...

def get(key: str, default: Any = None) -> Optional[Any]:
    ...

assert_type(get("a"), Any | None)
assert_type(get("a", default="b"), Any)
assert_type(get("a", default=42), Any)
assert_type(get("a", default=None), Any | None)

