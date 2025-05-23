#date: 2025-05-23T16:47:55Z
#url: https://api.github.com/gists/e6f92a3bade1e0ad000e886eba522bf3
#owner: https://api.github.com/users/mypy-play

from typing import Protocol, Type, TypeVar



class MyProtocol(Protocol):
     a: int


T = TypeVar("T", bound=MyProtocol)


def foo(bar: Type[T]) -> T:
     raise NotImplementedError


class MyClass:
    a: int 

    def __init__(self, a: int):
        self.a = a
        
class MyClass2:
    a: float
    
    def __init__(self, a: float):
        self.a = a
        
inst: MyClass = foo(MyClass)
inst2: MyClass2 = foo(MyClass2)
