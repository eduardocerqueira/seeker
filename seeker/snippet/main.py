#date: 2023-09-25T16:48:04Z
#url: https://api.github.com/gists/61af435c15e57ce5221abada2221008a
#owner: https://api.github.com/users/mypy-play

import enum


class ListType(enum.IntFlag):
    ABC = enum.auto()
    DEF = enum.auto()
    GHI = enum.auto()
    
    @staticmethod
    def get_names() -> list[str]:
        to_return = []

        for e in ListType:
            reveal_type(e)
            reveal_type(e.name)
            to_return.append(e.name)
        

        return to_return
