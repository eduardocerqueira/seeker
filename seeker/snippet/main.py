#date: 2025-03-12T16:47:11Z
#url: https://api.github.com/gists/0ca4e075345f6e7af84f47d3bc0c8b28
#owner: https://api.github.com/users/mypy-play

from typing import override

class A:
    @override
    def __eq__(self, o: object, /) -> bool:
        return self is o

reveal_type(A().__hash__)