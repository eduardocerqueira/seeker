#date: 2023-07-19T16:53:47Z
#url: https://api.github.com/gists/3b86f01292179ebf1c80c51ee2410d9d
#owner: https://api.github.com/users/mypy-play

from collections.abc import Hashable

def needs_something_hashable(x: Hashable) -> None:
    hash(x)

needs_something_hashable(object())