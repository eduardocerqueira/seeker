#date: 2023-05-17T16:57:22Z
#url: https://api.github.com/gists/01aed7158d52185b79795df572fada33
#owner: https://api.github.com/users/NAF28

import time
import board
import neopixel
 
pixel_pin = board.D2   #the ring data is connected to this pin
num_pixels = 16        #number of leds pixels on the ring
 
pixels = neopixel.NeoPixel(pixel_pin, num_pixels, brightness=0.3, auto_write=False)
 
YELLOW = (252, 194, 3) #RGB
GREEN = (33, 250, 40)
SAGE = (65, 125, 71)
CYAN = (0, 255, 255)
BLUE = (0, 0, 242)
LAVENDER = (175, 91, 235)
WHITE = (255,255,255)
OFF = (0,0,0)


myCOLORS = [YELLOW, GREEN, SAGE, CYAN, BLUE, LAVENDER, WHITE, OFF]

while True:  
    for color in range(0,len(myCOLORS),1):
        for r in range(0,16,1):
            pixels[r]=myCOLORS[color]
            pixels.show()
            time.sleep(.1)# Write your code here :-)
