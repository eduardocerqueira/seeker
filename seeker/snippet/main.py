#date: 2023-05-25T16:54:57Z
#url: https://api.github.com/gists/024e73f63ed574413d478c524660fb2f
#owner: https://api.github.com/users/mypy-play

from typing import TypeVar, Protocol


T = TypeVar("T")
T_co = TypeVar("T_co", covariant=True)
T_contra = TypeVar("T_contra", contravariant=True)

class X(Protocol[T]):...
class X_co(Protocol[T_co]):...
class X_contra(Protocol[T_contra]):...
