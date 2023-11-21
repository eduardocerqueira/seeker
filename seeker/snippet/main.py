#date: 2023-11-21T16:47:28Z
#url: https://api.github.com/gists/b9455e239835a8faa55cb4719dd70e70
#owner: https://api.github.com/users/mypy-play

from typing import *

class A:
    _b: bool
    
    @property
    def b(self) -> Never:
        raise NotImplementedError("b cannot be read")
        
    @b.setter
    def b(self, x: bool) -> None:
        self._b = x
        
a = A()
a.b = False
print(a.b)