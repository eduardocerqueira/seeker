#date: 2024-10-01T16:52:27Z
#url: https://api.github.com/gists/83787e0c1a896753fca80236164c6f4f
#owner: https://api.github.com/users/mypy-play

from typing import Optional
def f(input: bool):
    if input is not None:
        eligibility: Optional[int] = 42
    else:
        eligibility: Optional[int] = None # Cannot redefine local variable eligibility error on this line