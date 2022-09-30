#date: 2022-09-30T17:06:04Z
#url: https://api.github.com/gists/af15a9765b01cee7d594435494ec5bcd
#owner: https://api.github.com/users/mypy-play

from dataclasses import dataclass
from typing import List, Literal, NoReturn

def assert_never(value: NoReturn) -> NoReturn:
    assert False, f'This code should never be reached, got: {value}'

@dataclass
class Cat:
    type: str

@dataclass
class Dog:
    good: Literal[True]


def assess_pets(pets: List[Cat | Dog]) -> str:
    for c in pets:
        match c:
            case Cat(type=t) if t == "tabby":
                return "lovely"
            case Cat(type=t) if t == "ginger":
                return "delightful"
            case Cat():
                return "fabulous"
            case Dog():
                return "good boy!"
        assert_never(c)

# pets = [Cat(type="tabby"), Cat(type="ginger"), Cat(type="lazy"), Dog(good=True)]
# assess_pets(pets)
