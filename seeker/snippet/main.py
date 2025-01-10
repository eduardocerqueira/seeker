#date: 2025-01-10T16:55:05Z
#url: https://api.github.com/gists/77cccd913f6c38a3536c4306c1439cf3
#owner: https://api.github.com/users/mypy-play

from typing import assert_never


def a(k: str, d: dict[str, int|str]):
    if isinstance(d[k], str):
        print(d[k].upper())


def b(k: str, d: dict[str, int|str]):
    match d[k]:
        case str(s):
            print(s.upper())
        case int(x):
            print(x)
        case unreachable:
            assert_never(unreachable)