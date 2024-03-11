#date: 2024-03-11T16:58:17Z
#url: https://api.github.com/gists/002708756d033f8055c6aa32fa7634a3
#owner: https://api.github.com/users/ColetonNamie


import board
import neopixel
import time
 
pixel_pin = board.D2   #the ring data is connected to this pin
num_pixels = 16        #number of leds pixels on the ring
 
pixels = neopixel.NeoPixel(pixel_pin, num_pixels, brightness=.3, auto_write=False)

DARK = (118, 200, 94) 
RED = (255, 0, 0) #RGB
YELLOW = (255, 150, 0)
GREEN = (0, 255, 0)
CYAN = (0, 255, 255)
BLUE = (0, 0, 255)
PURPLE = (180, 0, 255)
OFF = (0,0,0)
 
while True:
    pixels[0] = CYAN
    pixels.show()     #required to update pixels
    time.sleep(.5)
    
    pixels[1] = YELLOW
    pixels.show()
    time.sleep(.5)
    
    pixels[2] = CYAN
    pixels.show()
    time.sleep(.5)
    
    pixels[3] = YELLOW
    pixels.show()
    time.sleep(.5)
    
    pixels[4] = CYAN
    pixels.show()
    time.sleep(.5)
    
    pixels[5] = YELLOW
    pixels.show()
    time.sleep(.5)

    pixels[6] = CYAN
    pixels.show()
    time.sleep(.5)

    pixels[7] = YELLOW
    pixels.show()
    time.sleep(.5)
    
    pixels[8] = CYAN
    pixels.show()
    time.sleep(.5)
    
    pixels[9] = YELLOW
    pixels.show()
    time.sleep(.5)
    
    pixels[10] = CYAN
    pixels.show()
    time.sleep(.5)
    
    pixels[11] = YELLOW
    pixels.show()
    time.sleep(.5)
    
    pixels[12] = CYAN
    pixels.show()
    time.sleep(.5)
    
    pixels[13] = YELLOW
    pixels.show()
    time.sleep(.5)
    
    pixels[14] = CYAN
    pixels.show()
    time.sleep(.5)
    
    pixels[15] = YELLOW
    pixels.show()
    time.sleep(.5)
    
    pixels[16] = CYAN
    pixels.show()
    time.sleep(.5)


