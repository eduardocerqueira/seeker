#date: 2021-09-17T16:52:34Z
#url: https://api.github.com/gists/dd50e2925f581920a4987376679a648e
#owner: https://api.github.com/users/mypy-play

from __future__ import annotations

from typing import *

_T = TypeVar("_T")


def identity(v: _T) -> _T:
    return v


class Model:
    pass


class AdminSite:
    pass


def noop_register(*args: type[object], **kwargs: object) -> Callable:
    return identity


def admin_register(*models: Type[Model], site: Optional[AdminSite] = None) -> Callable:
    return identity
    
    
def get_setting(v: str) -> bool:
    return False


register = admin_register if get_setting("ham_enabled") else noop_register
reveal_type(register)


class Ham(Model):
    pass

@register(Ham)
class HamAdmin:
    pass

