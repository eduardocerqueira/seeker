#date: 2025-03-26T16:51:59Z
#url: https://api.github.com/gists/7e2b4813378a163ed9b89bd75ca129c2
#owner: https://api.github.com/users/mypy-play

from typing import Self

class A:
    B: Self
    
A.B = A()
