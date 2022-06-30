#date: 2022-06-30T16:59:54Z
#url: https://api.github.com/gists/078a53189ef3bfae1bdc248904e013e4
#owner: https://api.github.com/users/mypy-play

from typing import Optional


def area_of_square(width: Optional[float] = None, 
                   height: Optional[float] = None) -> float:
    if width is not None and height is not None:
        raise ValueError('Please specify a width or height, not both')
    elif width is not None:
        area = width**2 
    elif height is not None:
        area = height**2
    else:
        raise ValueError('You have not specified a width or height')
    return area