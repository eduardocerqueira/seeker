#date: 2025-07-24T16:52:49Z
#url: https://api.github.com/gists/4a2e337fef1b0f71bef9ef936ac48883
#owner: https://api.github.com/users/mypy-play

from typing import Literal, Protocol, runtime_checkable

@runtime_checkable
class Foo(Protocol):
    def __len__(self) -> Literal[2]: ...
    
class Bar:
    def __len__(self) -> Literal[3]: ...

def f(x: Bar):
    if isinstance(x, Foo):
        print('is this unreachable?')
