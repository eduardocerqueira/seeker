#date: 2024-11-15T16:57:58Z
#url: https://api.github.com/gists/7b7bd1deef77883bac6b65cb5abc73ae
#owner: https://api.github.com/users/mypy-play

from enum import Enum

class E(Enum):
    A = 1
    B = 2
    C = 3

    def ret_num(self) -> str:
        match self:
            case E.A:
                return 'ALPHA'
            case E.B:
                return 'BETA'
            case E.C:
                return 'GAMMA'
