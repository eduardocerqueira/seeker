#date: 2023-02-27T16:55:14Z
#url: https://api.github.com/gists/22b72e434e979c16b0fec6ebce7437b8
#owner: https://api.github.com/users/mypy-play


from typing import *

T = TypeVar("T")

def foo(x: T, y: T) -> T:
    print(x, y)
    return x
    
    
foo(1, "2")