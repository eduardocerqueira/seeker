#date: 2021-12-10T17:03:12Z
#url: https://api.github.com/gists/fcff4e15b84e3393ceefe578d6f1ecfe
#owner: https://api.github.com/users/Este30

from math import *
from kandinsky import *

option_logo = [
[0,0,0,0,0,0,1,1,1,0,0,0,0,0,0],
[0,0,0,0,1,1,0,0,0,1,1,0,0,0,0],
[0,0,1,1,0,0,0,0,0,0,0,1,1,0,0],
[0,0,1,0,0,0,0,1,0,0,0,0,1,0,0],
[0,1,0,0,0,0,0,1,0,0,0,0,0,1,0],
[0,1,0,0,0,0,0,1,0,0,0,0,0,1,0],
[1,0,0,0,0,0,0,1,0,0,0,0,0,0,1],
[1,0,0,0,0,0,0,1,1,1,1,1,0,0,1],
[1,0,0,0,0,0,0,0,0,0,0,0,0,0,1],
[0,1,0,0,0,0,0,0,0,0,0,0,0,1,0],
[0,1,0,0,0,0,0,0,0,0,0,0,0,1,0],
[0,0,1,0,0,0,0,0,0,0,0,0,1,0,0],
[0,0,1,1,0,0,0,0,0,0,0,1,1,0,0],
[0,0,0,0,1,1,0,0,0,1,1,0,0,0,0],
[0,0,0,0,0,0,1,1,1,0,0,0,0,0,0],
]

def image_display(image_list, x, y, pos_x, pos_y):
    for x in range (0, x):
        for y in range (0, y):
            
