#date: 2022-02-03T16:59:16Z
#url: https://api.github.com/gists/12961ec25f126406256d31c84b39ae08
#owner: https://api.github.com/users/mypy-play

from typing import NoReturn

class Foo:
    def __enter__(self) -> NoReturn:
        pass

    def __exit__(self, *args) -> None:
        pass
    

def with_statement() -> NoReturn:
    with Foo(): pass


def call_method() -> NoReturn:
    Foo().__enter__()