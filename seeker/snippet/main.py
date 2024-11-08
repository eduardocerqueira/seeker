#date: 2024-11-08T16:50:22Z
#url: https://api.github.com/gists/8aa81aec21fba06843f878d110fe8f10
#owner: https://api.github.com/users/mypy-play

from typing import Any


x: tuple[int | str] = (1,)
y: tuple[int] | tuple[str] = (1,)
y = x

