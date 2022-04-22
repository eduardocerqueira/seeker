#date: 2022-04-22T17:09:46Z
#url: https://api.github.com/gists/b6633be60b6f68530a1bc273d18eef5d
#owner: https://api.github.com/users/mypy-play

from typing import Sequence, TypeVar

T = TypeVar("T")

def first(seq: Sequence[T]) -> T:
    return seq[0]
    
"a" + first("bcde")
1 + first(range(12))
1 + first("abcde")