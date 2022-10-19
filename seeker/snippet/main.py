#date: 2022-10-19T17:19:02Z
#url: https://api.github.com/gists/eb6ac0e04b6f432d3baf5aba1fe5cb4f
#owner: https://api.github.com/users/mypy-play

from typing import Protocol

class MyProtocol(Protocol):
    def func(self) -> str: ...

def meets_prot() -> list[MyProtocol]:
    class ImplsA:
        def func(self) -> str:
            return "a"
    
    class ImplsB:
        def func(self) -> str:
            return "b"
    
    c: list[MyProtocol] = []
    c.append(ImplsA())
    c.append(ImplsB())
    return c

for obj in meets_prot():
    reveal_type(obj)
    print(obj.func())