#date: 2024-05-06T16:55:24Z
#url: https://api.github.com/gists/e7905196003301ff779cbef6a3874d10
#owner: https://api.github.com/users/NolaEvans

import time
import board
import neopixel
 
pixel_pin = board.D2   #the ring data is connected to this pin
num_pixels = 16        #number of leds pixels on the ring
 
pixels = neopixel.NeoPixel(pixel_pin, num_pixels, brightness=0.3, auto_write=False)
 
RED = (255, 0, 0) #RGB
YELLOW = (255, 150, 0)
GREEN = (0, 255, 0)
CYAN = (0, 255, 255)
BLUE = (0, 0, 255)
PURPLE = (180, 0, 255)
WHITE = (255,255,255)
OFF = (0,0,0)

myCOLORS = [RED, YELLOW, GREEN, CYAN, BLUE, PURPLE, WHITE, OFF]

while True:  
    for color in myCOLORS:
        for light in range(0,16,1):
            pixels[light]=color
            pixels.show()
            time.sleep(.1)