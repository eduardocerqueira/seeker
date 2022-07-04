#date: 2022-07-04T03:18:25Z
#url: https://api.github.com/gists/81af31f1e016bcd48345dddf0db221a5
#owner: https://api.github.com/users/mypy-play

from typing import Final

TYPE_NONE = type(None)
reveal_type(type(None))  # => `Type[None]`
reveal_type(TYPE_NONE)  # => `builtins.object`
TYPE_NONE_F: Final = type(None)
reveal_type(TYPE_NONE_F)  # => `Type[None]`