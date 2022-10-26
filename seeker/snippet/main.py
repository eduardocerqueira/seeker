#date: 2022-10-26T16:56:00Z
#url: https://api.github.com/gists/82a35d384f747fd09cc48cd241d12672
#owner: https://api.github.com/users/mypy-play

from typing import Protocol

class P(Protocol):
    def default_impl(self) -> None:
        print("hello")
        
    def virtual(self) -> int:
        ...
        

class B(P):
    pass


x: P = B()