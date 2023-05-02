#date: 2023-05-02T16:55:56Z
#url: https://api.github.com/gists/0b35d3d87c9e1969bdd695b036546b76
#owner: https://api.github.com/users/allisonxcohn

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

myCOLORS = [RED, YELLOW, GREEN, CYAN, BLUE, PURPLE, WHITE,]

while True:  
    for color in range(0,len(myCOLORS),1):
        pixels.fill(myCOLORS[color])
        pixels.show()
        time.sleep(5)

