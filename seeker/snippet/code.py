#date: 2022-04-21T17:09:50Z
#url: https://api.github.com/gists/d2099ea2071d7a593d40966283ee6410
#owner: https://api.github.com/users/RajanRao12

import time
import board
import neopixel
 
pixel_pin = board.D2   #the ring data is connected to this pin
num_pixels = 12        #number of leds pixels on the ring
 
pixels = neopixel.NeoPixel(pixel_pin, num_pixels, brightness=0.3, auto_write=False)
 
RED = (255, 0, 0) #RGB
YELLOW = (255, 150, 0)
GREEN = (0, 255, 0)
CYAN = (0, 255, 255)
BLUE = (0, 0, 255)
PURPLE = (180, 0, 255)
WHITE = (255,255,255)
OFF = (0,0,0)
ORANGEROBIN = (225,60,20)
REDROBIN = (222,35,5) #yum
 
while True:
    pixels.fill(REDROBIN)
    pixels.show()     #required to update pixels
    time.sleep(1)
    pixels.fill(ORANGEROBIN)
    pixels.show()
    time.sleep(1)