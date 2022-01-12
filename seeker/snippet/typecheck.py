#date: 2022-01-12T17:03:51Z
#url: https://api.github.com/gists/ec16c5606cf1024106d9cf3de6281dbe
#owner: https://api.github.com/users/alex-shafer-1001

"""
For Python 3.10+ ONLY

Type-checks a value that has been decoded from JSON against a Python type hint,
for example, as obtained from dataclasses.Field.type
"""

from typing import Any
import types, typing


class NotEncodableError(Exception):
    def __init__(self, type_hint):
        Exception.__init__(
            self,
            'type_hint {type_hint!r} is invalid, it cannot be encoded in JSON'
        )


def check_type(value: Any, type_hint: Any) -> bool:
    """Check a value decoded from JSON against a Python type hint

    Raises NotEncodableError in the case that the type hint has no
    way of being encoded in JSON (for example, if there is a class
    used in the hint that does not have a corresponding JSON type)
    """

    try:
        real_type = type_hint.__origin__
    except AttributeError:
        real_type = type(type_hint)
        if real_type == type:
            real_type = type_hint

    if real_type in (types.UnionType, typing.Union):
        for union_subtype in type_hint.__args__:
            if check_type(value, union_subtype):
                return True
        return False
    elif real_type == list:
        if not isinstance(value, real_type):
            return False
        list_element_type = type_hint.__args__[0]
        for list_element in value:
            if not check_type(list_element, list_element_type):
                return False
        return True
    elif real_type == dict:
        if not isinstance(value, real_type):
            return False
        key_type = type_hint.__args__[0]
        value_type = type_hint.__args__[1]
        for dict_key, dict_val in value.items():
            if not check_type(dict_key, key_type):
                return False
            if not check_type(dict_val, value_type):
                return False
        return True
    elif real_type in (str, int, float, bool, type(None)):
        return isinstance(value, real_type)
    raise NotEncodableError(type_hint)