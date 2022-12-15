#date: 2022-12-15T16:58:05Z
#url: https://api.github.com/gists/ea23c99a545f7772126fd293a1d3c75a
#owner: https://api.github.com/users/mypy-play

from typing import Generic, TypeVar, Union
T = TypeVar('T')
class A(Generic[T]): pass
class B: pass
C = Union[A[T],  B]
D = C[int]
x: D
reveal_type(x)