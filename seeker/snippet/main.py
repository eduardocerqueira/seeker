#date: 2023-05-26T16:47:40Z
#url: https://api.github.com/gists/de1aad3526ca1eec4d442cb4fb97d802
#owner: https://api.github.com/users/mypy-play

from functools import lru_cache
from collections.abc import Callable


def translate_symbol() -> None:
    pass


translate_symbol_with_cache: Callable[[], None] = lru_cache(maxsize=None)(
    translate_symbol
)
