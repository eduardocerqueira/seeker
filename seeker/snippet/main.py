#date: 2022-03-14T16:48:04Z
#url: https://api.github.com/gists/8978a983cb1f92334d5e5627c872701a
#owner: https://api.github.com/users/mypy-play

from typing import TypeVar, Type, Iterable

T = TypeVar('T')

class A:
    @classmethod
    def sum(cls: Type[T], factors: Iterable[T]) -> T:
        ...
    
M = TypeVar('M')

class Mixin:
    @classmethod
    def sum(cls: Type[M], factors: Iterable[M]) -> M:
        ...
        
class C(Mixin, A):
    ...
