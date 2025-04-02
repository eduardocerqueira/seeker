#date: 2025-04-02T17:12:21Z
#url: https://api.github.com/gists/cb9a519dc0edb1f268314fb5c9260f29
#owner: https://api.github.com/users/mypy-play


def check_int(a: int | None) -> None:
    if a is None:
        raise TypeError()

v: int | None = 1
check_int(v)
logging.info(f"{v+1}")