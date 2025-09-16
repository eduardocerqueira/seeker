#date: 2025-09-16T16:58:59Z
#url: https://api.github.com/gists/02a62544692e57d278c6b63bf87fb42f
#owner: https://api.github.com/users/AlexandrDragunkin

from dataclasses import dataclass
from logging import basicConfig

from metaclasses import field_property_support

basicConfig(level='DEBUG')


@dataclass
class Test(metaclass=field_property_support):
    my_int: int
    name: str
    my_bool: bool = True

    @property
    def name(self):
        return self._name

    @name.setter
    def name(self, val):
        print(f'Setting name to: {val!r}')
        self._name = val


if __name__ == "__main__":
    t1 = Test(123, 'hello', False)
    print(t1)
    assert t1.name == 'hello'

    t2 = Test(name='my name', my_int=123)
    print(t2)
    assert t2.name == 'my name'

    # a TypeError is raised (missing required argument 'name')
    t3 = Test(123)
