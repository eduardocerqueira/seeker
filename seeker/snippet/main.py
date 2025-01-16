#date: 2025-01-16T16:49:11Z
#url: https://api.github.com/gists/d7137b6ec0287ceeda65c6df2c0b085c
#owner: https://api.github.com/users/mypy-play

from typing import Iterator, List

class Crap:
    def __init__(self):
        print("YO")

def blah() -> list[int]:
    return [3]

def blah3() -> list['Crap']:
    return [Crap()]

