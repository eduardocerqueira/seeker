#date: 2025-03-24T16:56:26Z
#url: https://api.github.com/gists/33ccfe8a6d1b9986aede91036b5f044c
#owner: https://api.github.com/users/mgebala888

import time
import board
import neopixel
from digitalio import DigitalInOut, Direction, Pull
import touchio


touch_pad = board.A0 # the ~1 pin
high_threshold = 3500
low_threshold = 2500
touch = touchio.TouchIn(touch_pad)

pixel_pin = board.D2
num_pixels = 16
 
pixels = neopixel.NeoPixel(pixel_pin, num_pixels, brightness=0.999, auto_write=False)
 
WHITE = (255, 255, 255)
OFF = (0, 0, 0)

colors = [WHITE,OFF]
now = 0

while True:
    time.sleep(0.15)
    print(touch.raw_value)
    if touch.raw_value < high_threshold and touch.raw_value > low_threshold:
        now=now+1
        print(now)
        print("touched")
        if (now >= len(colors)): #check to see if we exceed our list of colors
            now=0
        pixels.fill(colors[now])
        pixels.show()