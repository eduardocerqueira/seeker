#date: 2024-09-12T17:06:45Z
#url: https://api.github.com/gists/90f6017582cecd5f00eb132d45a73172
#owner: https://api.github.com/users/mypy-play

from dataclasses import asdict

import dataclasses

@dataclasses.dataclass
class Foo:
    a_string: int = 1
    a_float: str = "Hey"
    invalid_param: None = None


def bar(a_string: str, a_float: float) -> None:
    ...
    
foo = Foo()
bar(**asdict(foo))