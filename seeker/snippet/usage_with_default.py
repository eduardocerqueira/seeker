#date: 2025-09-16T16:58:59Z
#url: https://api.github.com/gists/02a62544692e57d278c6b63bf87fb42f
#owner: https://api.github.com/users/AlexandrDragunkin

from dataclasses import dataclass
from logging import basicConfig
from typing import ClassVar

from metaclasses import field_property_support

basicConfig(level='DEBUG')


@dataclass
class Test(metaclass=field_property_support):
    name: str
    _name: ClassVar[str] = 'schbell'

    @property
    def name(self):
        return self._name

    @name.setter
    def name(self, val):
        print(f'Setting name to: {val!r}')
        self._name = val


if __name__ == "__main__":
    t1 = Test()
    print(t1)
    assert t1.name == 'schbell'

    t2 = Test(name='hello')
    print(t2)
    assert t2.name == 'hello'
    t2.name = 'not-schbell'
    print(t2)
    assert t2.name == 'not-schbell'

    t3 = Test("llebhcs")
    print(t3)
    assert t3.name == 'llebhcs'
