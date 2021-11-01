#date: 2021-11-01T17:05:05Z
#url: https://api.github.com/gists/9a5a6589e78335c9e21bea81d176bbdc
#owner: https://api.github.com/users/IgorZyktin

from abc import ABC, abstractmethod


class A(ABC):
    @property
    @abstractmethod
    def attr(self) -> str:
        ...


class B:
    attr: str


class C(A):
    def __init__(self):
        self.attr = '25'


a = A()  # TypeError: Can't instantiate abstract class A with abstract methods attr

b = B()
b.attr  # AttributeError: 'B' object has no attribute 'attr'

c = C()  # AttributeError: 'B' object has no attribute 'attr'
