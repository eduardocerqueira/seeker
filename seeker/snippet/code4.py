#date: 2025-05-12T16:55:21Z
#url: https://api.github.com/gists/9d3619cdb98e2296c228a86e1e312d33
#owner: https://api.github.com/users/28atticuskilar

import time
import board
import neopixel
from digitalio import DigitalInOut, Direction, Pull

switch = DigitalInOut(board.D3)
switch.direction = Direction.INPUT
switch.pull = Pull.UP

pixel_pin = board.D2
num_pixels = 16
 
pixels = neopixel.NeoPixel(pixel_pin, num_pixels, brightness=0.7, auto_write=False)
on=0

while True:
    if (switch.value == 0 and on == 0):
        for num in range(0, 254, 2):
            COLOR=(num, num, num)
            for i in range(0, 16, 1):
                pixels[i] =COLOR
            pixels.show()
            on=1

    if (switch.value == 1 and on == 1):
        for num in range(254,-1,-2):
            COLOR=(num,num,num)
            for i in range(0,16,1):
                pixels[i] = COLOR
            pixels.show()
            on = 0
    time.sleep(0.1)
    print(on)
