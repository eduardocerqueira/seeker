#date: 2023-08-15T17:04:51Z
#url: https://api.github.com/gists/567685e44818c7fb1ef123a77d8ef27d
#owner: https://api.github.com/users/mypy-play

from dataclasses import dataclass
from datetime import datetime
import enum
from typing import assert_never


class MyEnum(enum.Enum):
    a = "a"
    b = "b"
    c = "c"
    

@dataclass
class MyClass:
    expires_at: datetime | None
    
    @property
    def status(self):
        match self.expires_at:
            case None:
                return MyEnum.a
            case time if time <= datetime.now():
                return MyEnum.b
            case _:
                return MyEnum.c