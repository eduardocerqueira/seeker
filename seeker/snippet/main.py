#date: 2023-02-24T17:08:06Z
#url: https://api.github.com/gists/db17b5ffdc0f53476a7873dffce3e870
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