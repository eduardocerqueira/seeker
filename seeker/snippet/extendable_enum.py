#date: 2026-02-03T17:40:29Z
#url: https://api.github.com/gists/9b278250154078245f5ec98c9d4c0423
#owner: https://api.github.com/users/datavudeja

"""An extendable Enum

Though this one is probably better:
https://pypi.python.org/pypi/aenum
"""

from enum import Enum

class ExtendableEnum(Enum):
    @classmethod
    def add(cls, name, val=None):
        if name not in cls.__members__:
            if not val:
                val = len(cls.__members__) + 1
            new_key = name
            cls.__dict__['_member_map_'][new_key] = val
            cls.__dict__['_member_names_'].append(new_key)
            cls.__dict__['_value2member_map_'][val] = new_key

class Colors(ExtendableEnum):
    RED = 1
    GREEN = 2

# Example Usage:

# add a color
print(Colors(2))
Colors.add('CYAN', 3)
print(Colors(3), Colors.CYAN)

# add a duplicate color
Colors.add('TURQUOISE')
print(Colors.TURQUOISE, Colors(4))

# after adding duplicate old still works
Colors.add('TURQUOISE')
print(Colors.TURQUOISE)
print(Colors(4))

# and the duplicate was prevented
try:
    Colors(5)
except ValueError as e:  # 5 is not a valid Colors
    print(e)
