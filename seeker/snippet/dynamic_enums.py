#date: 2026-02-03T17:39:29Z
#url: https://api.github.com/gists/c80bcdb1d75bd934d928d5d9a87f359f
#owner: https://api.github.com/users/datavudeja

"""I saw on StackOverflow _It isn't possible_ - then I asked GPT, and it parroted the stack overflow answer.

In this example we wish to create a ready-to-use Enum extended class.

    class Cooked(Enum):
        POP = 'pop'
        FOO =  'foo'

This,  with the ability to append new enums members. 
Understandably this cannot be done _after_ the Enum is defined, 
as they special c things (not a generic object type) - so they act a bit strange under the `type()`.

But we can build the `Cooked` class dynamically using the same typing:

    members = {
        'POP': 'pop',
        'FOO': 'foo',
        'EGG': 'egg'
    }
    Cooked = generate_enum('Cooked', members)

Under-the-hood this essentially _writes_ the class definition as if we wrote it manually, this is hoisted as a normal class.

---

Notably there is a library called `aenum` that performs the same. So rock-on aenum (\m/ >.< \m/); it clearly is possible.
"""
from enum import Enum, EnumMeta
import enum


class DynamicStrEnum(EnumMeta):
    def __new__(metacls, cls, bases=None, classdict=None):
        init_members = classdict or {}

        enum_dict = enum._EnumDict()
        bases = bases or (Enum,)

        for key, value in init_members.items():
            enum_dict[key] = value

        props = type(enum_dict)()
        props['has'] = classmethod(has)

        for key in enum_dict._member_names:
            value = enum_dict[key]
            props[key] = key.lower() if len(value) == 0 else value

        names = set(enum_dict._member_names)
        for key, value in enum_dict.items():
            if key in names:
                continue
            props[key] = value

        return super(DynamicStrEnum, metacls).__new__(metacls,
                                                      cls, bases, props)

class CookedBase(Enum):
    """The base for the dynamic bake content
    """
    alt_has = classmethod(has)

    
def generate_enum(name, members_dict, enum_bases=None):
    """Build and return a new Enum class

        Baked = DynamicStrEnum('Cooked', (CookedBase,), get_members())
    """
    return DynamicStrEnum(name, enum_bases or (CookedBase,), members_dict)


def has(cls, value):
    return value in cls._value2member_map_


class Cooked(Enum):
    """An existing (already cooked) enum
    """
    POP = 'pop'
    FOO =  'foo'

    @classmethod
    def has(cls, value):
        return value in cls._value2member_map_


members = {
    'POP': 'pop',
    'FOO': 'foo',
    'EGG': 'egg'
}


Baked = generate_enum('Cooked', members)

# Usage
assert Baked.POP.value == Cooked.POP.value
assert Baked.has('pop')
assert Baked.alt_has == Baked.has
assert Cooked.has('pop')
