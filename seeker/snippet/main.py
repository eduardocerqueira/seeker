#date: 2024-04-23T16:57:24Z
#url: https://api.github.com/gists/cc5b97283732e1da7de06717e0583b6f
#owner: https://api.github.com/users/mypy-play

from typing import Self
from enum import Enum

class MyBaseEnum(Enum):
    
    @classmethod
    def get_something(cls) -> Self:
        ...
        
        
class MyConcreteEnum1(MyBaseEnum):
    pass

class MyConcreteEnum2(MyBaseEnum):
    pass



reveal_type(MyConcreteEnum1.get_something())
reveal_type(MyConcreteEnum2.get_something())