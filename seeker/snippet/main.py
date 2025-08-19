#date: 2025-08-19T17:08:07Z
#url: https://api.github.com/gists/9a9c8d905dc4711d73e5dfd6f649344d
#owner: https://api.github.com/users/mypy-play

def takes_at_least3(x1: str, x2: str, x3: str, *args: str) -> None: ...

def test(
    x0: tuple[str, ...],
    x1: tuple[str, *tuple[str, ...]],
    x2: tuple[str, str, *tuple[str, ...]],
) -> None:
    takes_at_least3(*x0)  # no error
    takes_at_least3(*x1)  # Missing positional arguments "x2", "x3"
    takes_at_least3(*x2)  # no error
