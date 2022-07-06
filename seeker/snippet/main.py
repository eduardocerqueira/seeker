#date: 2022-07-06T17:03:48Z
#url: https://api.github.com/gists/382259e2880fe26e47db36cf9500f3dc
#owner: https://api.github.com/users/mypy-play

from typing import TypeAlias, Union, TypeVar, Tuple, Optional, Sequence, overload, Any
from datetime import date, time, datetime

SliderScalar: TypeAlias = Union[int, float, date, time, datetime]
# SliderScalarT = TypeVar("SliderScalarT", bound=SliderScalar)
SliderScalarT = TypeVar("SliderScalarT", int, float, date, time, datetime)

SliderTupleGeneric: TypeAlias = Union[Tuple[SliderScalarT], Tuple[SliderScalarT, SliderScalarT]]
SliderReturnGeneric: TypeAlias = Union[SliderScalarT, SliderTupleGeneric[SliderScalarT]]
SliderReturn: TypeAlias = Union[
    SliderReturnGeneric[int],
    SliderReturnGeneric[float],
    SliderReturnGeneric[date],
    SliderReturnGeneric[time],
    SliderReturnGeneric[datetime]
    ]

SliderValue = Union[SliderReturnGeneric[SliderScalarT], list[SliderScalarT], Tuple[()], None]

SliderReturnT = TypeVar("SliderReturnT", bound=SliderReturn)


@overload
def foo(x: None) -> int:
    ...

@overload
def foo(x: Tuple[()]) -> Tupe[int]:
    ...
    
@overload
def foo(x: list[SliderScalarT]) -> SliderTupleGeneric[SliderScalarT]:
    ...

@overload
def foo(x: SliderReturnT) -> SliderReturnT:
    ...

def foo(x: SliderValue) -> SliderReturn:
    y: Any = 42
    return y
    

reveal_type(
    foo(1)
)
reveal_type(
    foo((1,))
)
reveal_type(
    foo([])
)
reveal_type(
    foo([1.0])
)
reveal_type(
    foo([1])
)
reveal_type(
    foo(None)
)

reveal_type(
    foo(())
)