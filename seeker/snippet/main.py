#date: 2021-09-29T16:48:45Z
#url: https://api.github.com/gists/70ee1fef5167e349899291e06353e44c
#owner: https://api.github.com/users/mypy-play

from typing import TypeVar, Type
from numbers import Number
from fractions import Fraction
from decimal import Decimal


T = TypeVar('T', bound=Number)


def foo(bar: T):
    pass


foo(10)
foo(20)
foo(0.5)
foo("A")
