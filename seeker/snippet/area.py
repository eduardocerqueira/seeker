#date: 2022-09-23T17:27:36Z
#url: https://api.github.com/gists/9e6690ebfcf1fd93b3d59e891aa65b4c
#owner: https://api.github.com/users/viniciusao

from typing import NamedTuple


class FunctionNumber101(NamedTuple):
    width: float
    height: float

    def area(self):
        return self.width * self.height

square = FunctionNumber101(10, 10)
assert square.area() == 10
