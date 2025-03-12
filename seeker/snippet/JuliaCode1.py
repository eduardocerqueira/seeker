#date: 2025-03-12T16:48:53Z
#url: https://api.github.com/gists/1e4c49af5c07a04e6b4636c675193789
#owner: https://api.github.com/users/JuliaA-DT

import board
import neopixel
import time
 
pixel_pin = board.D2   #the ring data is connected to this pin
num_pixels = 16        #number of leds pixels on the ring
 
pixels = neopixel.NeoPixel(pixel_pin, num_pixels, brightness=0.3, auto_write=False)
 
RED = (255, 0, 0) # RGB
YELLOW = (255, 150, 0)
GREEN = (0, 255, 0)
CYAN = (0, 255, 255)
BLUE = (0, 0, 255)
PURPLE = (180, 0, 255)
OFF = (0, 0, 0)
ONE = (0, 150, 150)
TWO = (0, 75, 150)
THREE = (75, 0, 150)
FOUR = (150, 0, 150)
FIVE = (250, 20, 100)
SIX = (240, 0, 0)

 
while True:
    pixels[0] = ONE
    pixels.show()     #required to update pixels
    time.sleep(1)
    
    pixels[1] = TWO
    pixels.show()
    time.sleep(1)
    
    pixels[2] = THREE
    pixels.show()
    time.sleep(1)
    
    pixels[3] = FOUR
    pixels.show()
    time.sleep(1)
    
    pixels[4] = FIVE
    pixels.show()
    time.sleep(1)
    
    pixels[5] = SIX
    pixels.show()
    time.sleep(1)