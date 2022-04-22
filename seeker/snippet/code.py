#date: 2022-04-22T17:14:50Z
#url: https://api.github.com/gists/d1c73e399a1c3a8a20815ac43a9dd17c
#owner: https://api.github.com/users/ShemMarah

import time
import board
import neopixel
 
pixel_pin = board.D2   #the ring data is connected to this pin
num_pixels = 12        #number of leds pixels on the ring
 
pixels = neopixel.NeoPixel(pixel_pin, num_pixels, brightness=0.3, auto_write=False)
 
RED = (255, 0, 0) #RGB
RORANGE = (255, 37.5, 0)
ORANGE = (255, 75, 0)
ORLLOW = (255, 112.5, 0)
YELLOW = (255, 150, 0)
YEEN= (112.5, 202.5, 0)
GREEN = (0, 255, 0)
OFF = (0,0,0)

myCOLORS = [RED, RORANGE, ORANGE, ORLLOW, YELLOW, YEEN, GREEN, OFF]

while True:  
    for color in range(0,len(myCOLORS),1):
        pixels.fill(myCOLORS[color])
        pixels.show()
        time.sleep(1)# Write your code here :-)
