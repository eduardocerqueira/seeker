#date: 2022-02-23T17:03:03Z
#url: https://api.github.com/gists/df575141a88009bb6b200700af0747e5
#owner: https://api.github.com/users/Xikso

#!/usr/bin/env python3
# -*- coding: utf-8 -*-

__author__    = "Tomas Fabian"
__copyright__ = "(c)2021 VSB-TUO, FEECS, Dept. of Computer Science"
__email__     = "tomas.fabian@vsb.cz"
__version__   = "0.1.0"

from sqlite3 import Cursor
import time
import math



from pynput import keyboard



"""
ANSI Escape Sequences
   
For further reference see https://en.wikipedia.org/wiki/ANSI_escape_code
Note that ESC = '\x1b'
"""
CLS = "\x1b[H\x1b[J"
CURSOR_MOVE = "\x1b[{0};{1}H"
CURSOR_HIDE = "\x1b[?25l"
CURSOR_SHOW = "\x1b[?25h" 
PIXEL = "\x1b[48;2;{0};{1};{2};38;2;{3};{4};{5}m{6}\x1b[0m"

# 16x16 tiles of our playground with walls ('#') and empty spaces (' ') 
scene = [
"################",
"#              #",
"#              #",
"#              #",
"#######  #######",
"#              #",
"#        #######",
"#              #",
"#   ##         #",
"#              #",
"#              #",
"#      ##      #",
"#      ##      #",
"#  #########   #",
"#              #",
"################",
]
scene.reverse() # flip the y-axis to match our further computations

# TODO ! check the actual size of your command line buffer !
width = 120
height = 30

# horizontal field of view in radians
fov_x = math.radians(90)

# actual height of the wall
actual_wall_height = 0.5

# player's initial position on the map
player_x = len(scene[0]) / 2 
player_y = len(scene) / 2

# initial angle between the positive x-axis and the player's optical axis
player_alpha = math.radians(90) # at the beginning, the player is looking "north"

# current action of the player (None, 'w', 'a', 's', 'd' or 'q' as quit)
player_action = None

# TODO parametric equation of a shifted circle
def circle(center_x, center_y, radius, alpha):
  x = center_x + radius * math.cos(alpha)
  y = center_y + radius * math.sin(alpha)
  return (x, y) 
       

# set the action according to the pressed key
def on_press(key):
  global player_action
  try:
    player_action = key.char.lower()
  except AttributeError:
    pass

# cancel the action after releasing the key
def on_release(key):    
  global player_action
  player_action = None

# TODO move the cursor to home position ((1, 1) is the top-left corner of the screen buffer)
def move_cursor(r=1, c=1):
  print(CURSOR_MOVE.format(r, c), end='')
  

# TODO show the cursor  
def show_cursor():
  print(CURSOR_SHOW, end="")
  
# TODO hide the cursor  
def hide_cursor():
    print(CURSOR_HIDE, end="")
  


# TODO make ansi escape sequence for colored background and foreground of the given character
def make_pixel2(fg=(255, 255, 255), bg=(0, 0, 0), char=" "):  
  return  PIXEL.format(fg[0],fg[1],fg[2],bg[0],bg[1],bg[2], char)
  

# TODO make ansi escape sequence for a single colored space
def make_pixel(fg_bg=(255, 255, 255)):  
  return  make_pixel2(fg_bg,fg_bg)



# TODO calculate a new position of the player
def make_action(dx=0.1, da=2*math.pi/64):
  global player_x, player_y, player_alpha, player_action
  if player_action:
    if player_action == 'a':
      player_alpha += da
    elif player_action == 'd':
      player_alpha -= da
    elif player_action == 'w':
      player_x, player_y = circle(player_x, player_y, dx,player_alpha)
    elif player_action == 's':
      player_x, player_y = circle(player_x, player_y, -dx,player_alpha)
  

def main():
  """
  Python interpreter doesn't enable the processing of ANSI escape sequences
  but we have many options how to enable them:
  """



  # (A) Windows only
  #import ctypes 
  #kernel32 = ctypes.windll.kernel32
  #kernel32.SetConsoleMode(kernel32.GetStdHandle(-11), 7)
  # (B) Windows and Linux
  import colored   
  colored.attr("reset")
  # (C) Windows only
  #import colorama
  #colorama.init()
  
  # TODO hide cursor

  # set the keyboard listener
  listener = keyboard.Listener(on_press=on_press, on_release=on_release)
  listener.start()

  # TODO focal length 
  f = (width/(2*math.tan(fov_x/2)))
  
  
  # screen buffer will be represented as a list of lists of spaces
  
  

  
  screen =  [[' ' for x in range(width)]for x in range(height)]
  
  
  



  # TODO initialize the screen buffer in the form as follows:
  # [ [' ', ..., ' '],    
  # ...     
  # [' ', ..., ' '], ]
  # where the number of columns equals to width and the number of rows equals to height    

  # TODO drawing loop  
  while player_action != 'q':
    # do the action
    make_action()
  
    # do the ray marching for every i-th column of the screen   
    for i in range(0, width):
      # TODO angle between the player's optical axis and the current ray      
      theta = -math.atan((i- width/2 + 0.5)/f)
      # TODO angle between the positive x-axis and the current ray  
      phi = player_alpha + theta      
      #screen[width] =  make_pixel2()

      # TODO check the hit with the wall and compute its distance
      hit = False
      r_i = 0.1
      while hit == False and r_i < 23 :
        x_i = player_x + r_i * math.cos(phi)
        y_i = player_y + r_i * math.sin(phi)
        test_x = math.floor(x_i)
        test_y = math.floor(y_i)
        if test_x < 0 or test_y < 0 or test_x > 15 or test_y > 15:
          break
        if scene[test_y][test_x] == "#" :
          hit = True
          break
        else :
          r_i +=0.1
          
      # TODO if a hit occured, compute the projected wall height otherwise set it to zero
      h_i = 0      
      if hit:
        h_i = (actual_wall_height * f )/ r_i                         

      # TODO fill the i-th column of the screen buffer based on the computed projected wall height h_i
      for j in range(height):
       pos_tmp = (height - h_i) / 2
       if j < pos_tmp :
         screen[j][i] = make_pixel((56,56,56))
       elif j >= height - pos_tmp:
           screen[j][i] = make_pixel((133,133,133))
       else :
         lenght = math.pow(max(0,min(1, 1 - r_i / 23)),2)
         screen[j][i] =  make_pixel((int(lenght *200),int (lenght * 180),int(lenght *150)))

        # TODO fill it with the ceiling or wall or floor color/pattern
        # it should be something like screen[j][i] = make_pixel((56, 56, 56))
    
    # print/draw the screen buffer
    buffer = ""
    for j in range(height):
      buffer += "".join(screen[j])



    # TODO fill the buffer with strings in the screen variable
    print(buffer, end="")
    
    move_cursor()
    hide_cursor()
    # TODO move the cursor to the upper-left corner (1, 1) to prepare for a new frame    
    
  # end of frame loop
 
  
  # TODO clear the console and show the cursor  
  show_cursor()



if __name__ == "__main__":
  
  main()