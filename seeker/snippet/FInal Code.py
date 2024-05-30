#date: 2024-05-30T17:07:37Z
#url: https://api.github.com/gists/820c0e84b64fe10babc1ff1c41b5a24c
#owner: https://api.github.com/users/Zachstrait

import time
import board
import neopixel
from digitalio import DigitalInOut, Direction, Pull

switch = DigitalInOut(board.D3)
switch.direction = Direction.INPUT
switch.pull = Pull.UP

pixel_pin = board.D2
num_pixels = 16
 
pixels = neopixel.NeoPixel(pixel_pin, num_pixels, brightness=0.9, auto_write=False)
 

GREEN = (0, 255, 0)
OFF = (0,0,0)

colors = [GREEN,OFF]
now=0

while True:
    print(switch.value)
    if (switch.value==False):    #detect the button press
        now=now+1 
        if (now >= 2): #2 colors in the list
            now=0
        pixels.fill(colors[now])
        pixels.show()
    time.sleep(0.12)  # debounce delay
