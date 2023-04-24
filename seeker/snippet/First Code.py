#date: 2023-04-24T17:03:24Z
#url: https://api.github.com/gists/ffc17c0ce205eb2d9c07f0570e9c10e2
#owner: https://api.github.com/users/ellamogannam

import time
import board
import neopixel
 
pixel_pin = board.D2   #the ring data is connected to this pin
num_pixels = 16        #number of leds pixels on the ring
 
pixels = neopixel.NeoPixel(pixel_pin, num_pixels, brightness=0.3, auto_write=False)
 
RED = (42, 0, 0) #RGB
YELLOW = (0, 0, 0)
GREEN = (0, 150, 0)
CYAN = (246, 73, 97)
BLUE = (0, 0, 132)
PURPLE = (0, 0, 0)
TEAL = (42, 150, 132)
ORANGE = (242, 155, 68)
WHITE = (255,255,255)
OFF = (0,0,0)
 
while True:
    
    pixels[6] = RED
    pixels.show()     #required to update pixels
    time.sleep(1)
    
    pixels[5] = ORANGE
    pixels.show()     #required to update pixels
    time.sleep(1)
    
    pixels[4] = YELLOW
    pixels.show()
    time.sleep(1)
    
    pixels[3] = GREEN
    pixels.show()
    time.sleep(1)
    
    pixels[2] = BLUE
    pixels.show()
    time.sleep(1)
    
    pixels[1] = TEAL
    pixels.show()
    time.sleep(1)
    
    pixels[0] = PURPLE
    pixels.show()
    time.sleep(1)
    
 
   