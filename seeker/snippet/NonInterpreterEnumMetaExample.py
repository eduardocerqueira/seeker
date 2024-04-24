#date: 2024-04-24T17:11:23Z
#url: https://api.github.com/gists/6edaad976f1ca0eb814dd2306a7bd175
#owner: https://api.github.com/users/CodeByAidan

from enum import EnumMeta, Enum
Color3 = EnumMeta('Color3', (Enum,), (_ := EnumMeta.__prepare__(
    'Color3', (Enum,),)) and any(map(_.__setitem__, *(zip(*{
        'RED': 1,
        'GREEN': 2,
        'BLUE': 3,
    }.items())))) or _
)
print(repr(Color3(1))) # <Color3.RED: 1>
