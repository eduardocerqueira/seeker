#date: 2022-03-14T16:52:06Z
#url: https://api.github.com/gists/37085de31d8cabfb7e583d28428c4721
#owner: https://api.github.com/users/RaphaelGoutmann

from kandinsky import * 
from math import *

BLACK = (0, 0, 0)
WHITE = (255, 255, 255)

RED = (255, 0, 0)
GREEN = (0, 255, 0)
BLUE = (0, 0, 255)

# vertical line

def vline( m, length, c = BLACK ):

    x = m[0]    

    for y in range(m[1], m[1] + length):
        set_pixel(x, y, c)

# horizontal line 

def hline( m, length, c = BLACK ):

    y = m[1]    

    for x in range(m[0], m[0] + length):
        set_pixel(x, y, c)

# line 

def line(m1, m2, c):

    # vline 
    if m1[0] == m2[0]: # same x pos



def main():
    vline( (10, 10), 50, (0, 0, 0) )
    hline( (10, 10), 50, (0, 0, 0) )


main()
