#date: 2022-04-21T17:09:36Z
#url: https://api.github.com/gists/c729a4f761216229742c97f7b5620572
#owner: https://api.github.com/users/TaliaSlavin

import board
import neopixel
import time
 
pixel_pin = board.D2   # the ring data is connected to this pin
num_pixels = 12        # number of leds pixels on the ring
 
pixels = neopixel.NeoPixel(pixel_pin, num_pixels, brightness=0.3, auto_write=False)
 
RED = (69, 5, 5)   # RGB
ORANGE = (56, 24, 1)
YELLOW = (102, 82, 2)
GREEN = (3, 43, 1)
CYAN = (0, 255, 255)
BLUE = (5, 2, 43)
PURPLE = (19, 1, 36)
WHITE = (255, 255, 255)
PINK = (38, 1, 36)
OFF = (0, 0, 0)
 
while True:
    pixels[0] = RED
    pixels.show()     # required to update pixels
    time.sleep(0.2)
    
    pixels[1] = ORANGE
    pixels.show()
    time.sleep(0.2)
    
    pixels[2] = YELLOW
    pixels.show()
    time.sleep(0.2)
    
    pixels[3] = GREEN
    pixels.show()
    time.sleep(0.2)
    
    pixels[4] = BLUE
    pixels.show()
    time.sleep(0.2)
    
    pixels[5] = PURPLE
    pixels.show()
    time.sleep(0.2)
    
    pixels[6] = PINK
    pixels.show()
    time.sleep(0.2)
    
    pixels[7] = RED
    pixels.show()     # required to update pixels
    time.sleep(0.2)
    
    pixels[8] = ORANGE
    pixels.show()
    time.sleep(0.2)
    
    pixels[9] = YELLOW
    pixels.show()
    time.sleep(0.2)
    
    pixels[10] = GREEN
    pixels.show()
    time.sleep(0.2)
    
    pixels[11] = BLUE
    pixels.show()
    time.sleep(0.2)
    
    pixels[12] = PURPLE
    pixels.show()
    time.sleep(0.2)
    
   