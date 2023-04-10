#date: 2023-04-10T16:45:27Z
#url: https://api.github.com/gists/7d142a33c08b68f889a2699375f82b2e
#owner: https://api.github.com/users/ahollard2

from math import cos,sin,pi
from kandinsky import fill_rect
from time import sleep
from random import randint,random

def ecran(w):
    t = 0
    h = int(.8 * w)
    wd,hd = w//2, h//2
    ta = 320 // w
    while True:
     coul = randint(100,1000)  
     for c in range(w):       
        for l in range(h):
            r = (c - wd) ** 2 + (l - hd) ** 2
            d = abs(1 + int(ta / 2 * sin(2 * pi * (r + t) / 90)))
            if c == wd: dx = 0
            elif c > wd: dx = 1
            else: dx = -1
            if l == hd: dy = 0
            elif l > hd: dy = 1
            else: dy = -1
            tad = ta // 2
            fill_rect(ta * c + dx * tad,ta * l + dy * tad,3 * tad,3 * tad ,(0,0,0))
            fill_rect(ta * c + dx * d,ta * l + dy * d,d,d,(coul,.9 * coul,0))
     t = (t + 1) % 90    
     sleep(random()/100)    
   
fill_rect(0,0,320,222,(0,0,0)) 
ecran(16)
