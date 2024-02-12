#date: 2024-02-12T17:02:52Z
#url: https://api.github.com/gists/71b46c303191dd3f01d95df4b9b3a755
#owner: https://api.github.com/users/mypy-play

from typing import TYPE_CHECKING, Any

try:
    from dataclasses import dataclass
except ImportError:
    assert not TYPE_CHECKING
    dataclass = Any