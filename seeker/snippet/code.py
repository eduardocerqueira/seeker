#date: 2026-02-09T17:31:35Z
#url: https://api.github.com/gists/143697c7a4d3ae0f039d3bdd03c3aee1
#owner: https://api.github.com/users/29devonselfe-dot

import time
import board
import neopixel
from digitalio import DigitalInOut, Direction, Pull

switch = DigitalInOut(board.D3)
switch.direction = Direction.INPUT
switch.pull = Pull.UP

pixel_pin = board.D2
num_pixels = 16
 
pixels = neopixel.NeoPixel(pixel_pin, num_pixels, brightness=0.3, auto_write=False)
 
GREEN = (5, 145, 3)
CYAN = (0, 255, 255)
BLUE = (0, 0, 255)
PURPLE = (157, 15, 245)
WHITE = (255,255,255)
OFF = (0,0,0)

colors = [GREEN,CYAN,BLUE,PURPLE,WHITE, OFF]
now=0

while True:
    print(switch.value)
    if (switch.value==False):    #detect the button press
        now=now+1 
        if (now >= 6): #7 colors in the list
            now=0
        pixels.fill(colors[now])
        pixels.show()
    time.sleep(0.12)  # debounce delay