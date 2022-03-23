#date: 2022-03-23T17:00:11Z
#url: https://api.github.com/gists/328b7f385f8117961f32ce9536f18bdb
#owner: https://api.github.com/users/RedGaming98

from math import *
from kandinsky import fill_rect, draw_string
from ion import *
from time import sleep
cursor = [0,0]
alive = []

Up = KEY_UP
Down = KEY_DOWN
Left = KEY_LEFT
Right = KEY_RIGHT
Ok = KEY_OK
Exe = KEY_EXE

"""
Values that work well: 40, 25, 8
"""
XPX = 40
YPX = 25
SPX = 8


draw_string("v2.0",0,204)

def move(direction):
  dirs = {Right:(0,1,5),Left:(0,-1,-5),Up:(1,-1,-5),Down:(1,1,5)}
  if keydown(direction):
    cursor[dirs[direction][0]] += dirs[direction][1+1*keydown(KEY_SHIFT)]
    if not(0 <= cursor[0] <= XPX-1 and 0 <= cursor[1] <= YPX-1):
      cursor[dirs[direction][0]] -= dirs[direction][1+1*keydown(KEY_SHIFT)]
    while(keydown(direction)):
      pass

while not keydown(Ok):
  prev_cursor = [] + cursor
  move(Right)
  move(Left)
  move(Up)
  move(Down)
  
  if keydown(Exe):
    if tuple(cursor) in alive:
      alive.remove(tuple(cursor))
    else:
      alive.append(tuple(cursor))
    # print(cursor)
    # print(alive)
    while(keydown(Exe)):
      pass
  
  if prev_cursor != cursor:
    if tuple(prev_cursor) in alive:
      fill_rect(prev_cursor[0]*SPX,prev_cursor[1]*SPX,SPX,SPX,'black')
    else:
      fill_rect(prev_cursor[0]*SPX,prev_cursor[1]*SPX,SPX,SPX,'white')
    fill_rect(cursor[0]*SPX,cursor[1]*SPX,SPX,SPX,'gray')

base_setup = [] + alive
while keydown(Ok):
  pass


while not keydown(Ok):
  prev_alive = [] + alive
  fill_rect(0,220,320,2,'white')
  for x in range(XPX):
    for y in range(YPX):
      fill_rect(0,220,int(((x*y)/(XPX*YPX))*320),2,'red')
      alives = 0
      for i in [(-1,-1),(0,-1),(1,-1),(1,0),(1,1),(0,1),(-1,1),(-1,0)]:
        if (x+i[0],y+i[1]) in prev_alive:
          alives += 1
      if (x,y) in prev_alive:
        if not 1 < alives < 4:
          alive.remove((x,y))
      else:
        if alives == 3:
          alive.append((x,y))
  fill_rect(0,0,320,205,'white')
  for cells in alive:
    fill_rect(cells[0]*SPX,cells[1]*SPX,SPX,SPX,'black')
