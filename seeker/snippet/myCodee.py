#date: 2025-03-12T16:57:23Z
#url: https://api.github.com/gists/312b88e3462182f713bce8ab3d698c85
#owner: https://api.github.com/users/EmaS24

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
banana = (240, 77, 136)
apple = (232, 5, 118)
orange = (58, 240, 2)
lemon = (108, 227, 75)
 
while True:
    pixels[0] = banana
    pixels.show()     #required to update pixels
    time.sleep(1)
    
    pixels[1] = apple
    pixels.show()
    time.sleep(1)
    
    pixels[2] = orange
    pixels.show()
    time.sleep(1)
    
    pixels[3] = lemon
    pixels.show()
    time.sleep(1)
    
    pixels[4] = BLUE
    pixels.show()
    time.sleep(1)
    
    pixels[5] = PURPLE
    pixels.show()
    time.sleep(1)