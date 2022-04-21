#date: 2022-04-21T17:05:16Z
#url: https://api.github.com/gists/4b82deecbdb355cf4f6273c03e9280a4
#owner: https://api.github.com/users/kyrazhen

import board
import neopixel
import time
 
pixel_pin = board.D2   #the ring data is connected to this pin
num_pixels = 12        #number of leds pixels on the ring
 
pixels = neopixel.NeoPixel(pixel_pin, num_pixels, brightness=0.3, auto_write=False)
 
RED = (255, 0, 0) #RGB
YELLOW = (255, 150, 0)
GREEN = (0, 255, 0)
CYAN = (0, 255, 255)
BLUE = (0, 0, 255)
PURPLE = (180, 0, 255)
MAUDE = (207, 27, 176)
LILAC = (100, 52, 145)
SAGE = (19, 110, 43)
COOL = (49, 142, 189)
SUN = (255, 123, 0)
MAROON = (166, 30, 18)
LIGHT = (255, 187, 0)
GOO = (179, 0, 161)
OFF = (0,0,0)
 
while True:
    pixels[0] = SUN
    pixels.show()     #required to update pixels
    time.sleep(0.5)
    
    pixels[1] = MAROON
    pixels.show()
    time.sleep(0.5)
    
    pixels[2] = LIGHT
    pixels.show()
    time.sleep(0.5)
    
    pixels[3] = GOO
    pixels.show()
    time.sleep(0.5)
    
    pixels[4] = SUN
    pixels.show()
    time.sleep(0.5)
    
    pixels[5] = MAROON
    pixels.show()
    time.sleep(0.5)
  
    pixels[6] = LIGHT
    pixels.show()     #required to update pixels
    time.sleep(0.5)
    
    pixels[7] = GOO
    pixels.show()
    time.sleep(0.5)
    
    pixels[8] = SUN
    pixels.show()
    time.sleep(0.5)
    
    pixels[9] = MAROON
    pixels.show()
    time.sleep(0.5)
    
    pixels[10] = LIGHT
    pixels.show()
    time.sleep(0.5)
    
    pixels[11] = GOO
    pixels.show()
    time.sleep(0.5)#
  
    pixels.fill(SUN)
    pixels.show()
    time.sleep(0.5)
    
    pixels.fill(MAROON)
    pixels.show()
    time.sleep(0.5)
    
    pixels.fill(LIGHT)
    pixels.show()
    time.sleep(0.5)
    
    pixels.fill(GOO)
    pixels.show()
    time.sleep(0.5)
    
    pixels[0] = GOO
    pixels.show()     #required to update pixels
    time.sleep(0.5)
    
    pixels[1] = LIGHT
    pixels.show()
    time.sleep(0.5)
    
    pixels[2] = MAROON
    pixels.show()
    time.sleep(0.5)
    
    pixels[3] = SUN
    pixels.show()
    time.sleep(0.5)
    
    pixels[4] = GOO
    pixels.show()
    time.sleep(0.5)
    
    pixels[5] = LIGHT
    pixels.show()
    time.sleep(0.5)
  
    pixels[6] = MAROON
    pixels.show()     #required to update pixels
    time.sleep(0.5)
    
    pixels[7] = SUN
    pixels.show()
    time.sleep(0.5)
    
    pixels[8] = GOO
    pixels.show()
    time.sleep(0.5)
    
    pixels[9] = LIGHT
    pixels.show()
    time.sleep(0.5)
    
    pixels[10] = MAROON
    pixels.show()
    time.sleep(0.5)
    
    pixels[11] = SUN
    pixels.show()
    time.sleep(0.5)#

