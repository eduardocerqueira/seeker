#date: 2022-05-05T17:05:04Z
#url: https://api.github.com/gists/19073ebc32038c1ba6c67d2fb0fbca5f
#owner: https://api.github.com/users/sammynaser07

import time 
import board 
import neopixel
 
pixel_pin = board.D2
num_pixels = 12
 
pixels = neopixel.NeoPixel(pixel_pin, num_pixels, brightness=0.1, auto_write=False)
 
RED = (255, 0, 0) 
YELLOW = (255, 150, 0) 
GREEN = (0, 255, 0) 
CYAN = (17, 176, 245) 
BLUE = (0, 0, 255) 
PURPLE = (180, 0, 255) 
WHITE = (255, 255, 255) 
OFF = (0, 0, 0) 
COLOR = WHITE
 
while True: 
    for c in range(0, 5, 1):
        if (c == 0):
            COLOR = BLUE
        if (c == 1):
            COLOR = CYAN
        if (c == 2):
            COLOR = BLUE
        if (c == 3):
            COLOR = CYAN
        if (c == 4):
            COLOR = BLUE
        if (c == 5):
            COLOR = CYAN
        if (c == 6):
            COLOR = BLUE
        if (c == 7):
            COLOR = Cyan
        for i in range(0, 12, 1):
            pixels[i] = COLOR
            pixels.show()
            time.sleep(0.04)