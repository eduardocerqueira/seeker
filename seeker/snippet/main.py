#date: 2023-07-27T16:42:32Z
#url: https://api.github.com/gists/98b9ba3edd02db5023edccb9f0099208
#owner: https://api.github.com/users/mypy-play

from typing import SupportsFloat
from fractions import Fraction
from decimal import Decimal


NumberType = type[SupportsFloat]


def foo(bar: NumberType = 1) -> None:
    pass


foo(float)
foo(int)
foo(Fraction)
foo(Decimal)

foo(complex)

foo(str)
foo(bytes)
foo(list)
foo(dict)