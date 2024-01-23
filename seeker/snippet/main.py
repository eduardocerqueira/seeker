#date: 2024-01-23T16:59:14Z
#url: https://api.github.com/gists/d2cd2e164a4b8c8a8c566996d48b3f40
#owner: https://api.github.com/users/mypy-play

from typing import overload, Literal

@overload
def get_teams(include_none: Literal[True] = True) -> dict[str | None, list[str]]: ...


@overload
def get_teams(include_none: Literal[False] = False) -> dict[str, list[str]]: ...


@overload
def get_teams(include_none: bool = True) -> dict[str, list[str]]: ...


def get_teams(include_none: bool = True) -> dict[str | None, list[str]]:
    d: dict[str | None, list[str]] = {"a": ["b"]}
    if include_none:
        d[None] = ["c"]
    return d