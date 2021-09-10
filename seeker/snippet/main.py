#date: 2021-09-10T16:58:53Z
#url: https://api.github.com/gists/5df0e6862deb927e5016ee82b454276c
#owner: https://api.github.com/users/jeansantophil

from kandinsky import *
from time import *

BL, NR = (255,255,255), (0,0,0)

line = [0,]*320
case = [line,]*222

case[100][100] = 1
set_pixel(100,100,NR)
case[101][101] = 1
set_pixel(101,101,NR)
case[101][102] = 1
set_pixel(101,102,NR)
case[100][102] = 1
set_pixel(100,102,NR)
case[99][102] = 1
set_pixel(99,102,NR)

pixel = 0
pixel2 = 0
while True:
  pixel = 0
  pixel2 = 0
  for c in range(320):
    for l in range(222):
      pixel = 0
      if get_pixel(c,l) == NR:
        if get_pixel(c-1,l) == BL:
          pixel+=1
        if get_pixel(c-1,l-1) == BL:
          pixel+=1
        if get_pixel(c,l-1) == BL:
          pixel+=1
        if get_pixel(c+1,l-1) == BL:
          pixel+=1
        if get_pixel(c+1,l) == BL:
          pixel+=1
        if get_pixel(c+1,l+1) == BL:
          pixel+=1
        if get_pixel(c,l+1) == BL:
          pixel+=1
        if get_pixel(c-1,l+1) == BL:
          pixel+=1
      if get_pixel(c,l) == BL:
        if get_pixel(c-1,l) == NR:
          pixel2+=1
        if get_pixel(c-1,l-1) == NR:
          pixel2+=1
        if get_pixel(c,l-1) == NR:
          pixel2+=1
        if get_pixel(c+1,l-1) == NR:
          pixel2+=1
        if get_pixel(c+1,l) == NR:
          pixel2+=1
        if get_pixel(c+1,l+1) == NR:
          pixel2+=1
        if get_pixel(c,l+1) == NR:
          pixel2+=1
        if get_pixel(c-1,l+1) == NR:
          pixel2+=1
      if pixel == 1:
        case[c][l] = 0
      if pixel == 2:
        case[c][l] = 0
      if pixel == 4:
        case[c][l] = 0
      if pixel == 5:
        case[c][l] = 0
      if pixel == 6:
        case[c][l] = 0
      if pixel == 7:
        case[c][l] = 0
      if pixel == 8:
        case[c][l] = 0
      if pixel2 == 3:
        case[c][l] = 1
      pixel = 0
      pixel2 = 0
  for c in range(320):
    for l in range(222):
      if case[c][l] == 0:
        set_pixel(c,l,BL)
      if case[c][l] == 1:
        set_pixel(c,l,NR)
