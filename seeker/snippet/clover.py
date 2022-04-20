#date: 2022-04-20T17:00:36Z
#url: https://api.github.com/gists/edea3eb201f19235e6e0f4e9b167906d
#owner: https://api.github.com/users/FlynnGS

import time
import board
import neopixel
 
pixel_pin = board.D2   #the ring data is connected to this pin
num_pixels = 12        #number of leds pixels on the ring
 
pixels = neopixel.NeoPixel(pixel_pin, num_pixels, brightness=0.7, auto_write=False)
 
RED = (255, 0, 0) #RGB
YELLOW = (255, 150, 0)
GREEN = (0, 255, 0)
CYAN = (0, 255, 255)
BLUE = (0, 0, 255)
PURPLE = (180, 0, 255)
WHITE = (255,255,255)
OFF = (0,0,0)
 
while True:
    pixels.fill(GREEN)
    pixels.show()     #required to update pixels
    time.sleep(1)