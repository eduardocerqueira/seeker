#date: 2025-03-12T16:59:02Z
#url: https://api.github.com/gists/3bf1335919e1576030a2e8814c58902b
#owner: https://api.github.com/users/12emmasullivan

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
Pink = (210, 50, 50)
Maroon = (200, 0, 70)
Blueberry = (50, 10, 200)
 
while True:
    pixels[0] = Pink
    pixels.show()     #required to update pixels
    time.sleep(1)
    
    pixels[1] = YELLOW
    pixels.show()
    time.sleep(1)
    
    pixels[2] = GREEN
    pixels.show()
    time.sleep(1)
    
    pixels[3] = CYAN
    pixels.show()
    time.sleep(1)
    
    pixels[4] = BLUE
    pixels.show()
    time.sleep(1)
    
    pixels[5] = PURPLE
    pixels.show()
    time.sleep(1)
    
    pixels[6] = Maroon
    pixels.show()
    time.sleep(1)
    
    pixels[7] = Blueberry
    pixels.show()
    time.sleep(1)
