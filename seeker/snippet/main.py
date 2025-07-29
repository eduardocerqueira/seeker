#date: 2025-07-29T16:59:30Z
#url: https://api.github.com/gists/354e2893ccaab62b330f74ba0919d06e
#owner: https://api.github.com/users/mypy-play

from typing import TypeVar, overload, NoReturn

from typing_extensions import assert_never

T = TypeVar('T', str, float)

def ComputeA(min_value: T, max_value: T) -> tuple[T, T]:
    match (min_value, max_value):
        case float(), float():
            return min_value*0.1, max_value*0.1
        case str(), str():
            return min_value, max_value
        case unreachable:
            reveal_type(unreachable)
            assert_never(unreachable[0])
            assert_never(unreachable[1])


def ComputeB(min_value: T, max_value: T) -> tuple[T, T]:
    x = (min_value, max_value)
    match x:
        case float(), float():
            return min_value*0.1, max_value*0.1
        case str(), str():
            return min_value, max_value
        case unreachable:
            reveal_type(unreachable)
            assert_never(unreachable[0])
            assert_never(unreachable[1])
