#date: 2023-07-11T16:51:23Z
#url: https://api.github.com/gists/1825df9e20d4f699f6549b2a31a54a8d
#owner: https://api.github.com/users/mypy-play

from __future__ import annotations

import sys
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from typing import Literal

NO: Literal["no"] = "no"
YES: Literal["yes"] = "yes"


def yes_no(question: str, default: Literal[YES, NO] | bool | None = YES) -> bool:
    pass