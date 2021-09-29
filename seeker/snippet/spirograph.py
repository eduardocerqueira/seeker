#date: 2021-09-29T16:48:44Z
#url: https://api.github.com/gists/43dc587a3714b0d0ac5142ab0ac30258
#owner: https://api.github.com/users/CireCunis

from math import *
from kandinsky import *
from time import sleep

def spiro(n1,n2,d):
    r1 = int(n1 * 70 / (n1 + n2))
    r2 = int(n2 * 70 / (n1 + n2))
    r3 = d * r2
    for i in range(360 * 100):
        a = pi * i / 180
        x = 160 + (r2 + r1) * cos(a) + r3 * cos(r1 * a / r2) 
        y = 110 + (r2 + r1) * sin(a) + r3 * sin(r1 * a / r2) 
        set_pixel(int(x),int(y),(0,0,0))
        sleep(.001)

spiro(40,60,0.7)
