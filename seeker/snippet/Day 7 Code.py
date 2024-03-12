#date: 2024-03-12T16:52:08Z
#url: https://api.github.com/gists/0c5b2a03be3e46ec2752dcb4d72f0fb9
#owner: https://api.github.com/users/charliemoore08

import board
import neopixel
import time
 
pixel_pin = board.D2   #the ring data is connected to this pin
num_pixels = 16        #number of leds pixels on the ring
 
pixels = neopixel.NeoPixel(pixel_pin, num_pixels, brightness=0.3, auto_write=False)
 
RED = (255, 0, 0) #RGB
YELLOW = (255, 150, 0)
GREEN = (0, 255, 0)
CYAN = (0, 255, 255)
BLUE = (0, 0, 255)
PURPLE = (180, 0, 255)
OFF = (0,0,0)
 
while True:
    pixels[0] = RED
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
    
    pixels[6] = RED
    pixels.show()     #required to update pixels
    time.sleep(1)
    
    pixels[7] = YELLOW
    pixels.show()
    time.sleep(1)
    
    pixels[8] = GREEN
    pixels.show()
    time.sleep(1)
    
    pixels[9] = CYAN
    pixels.show()
    time.sleep(1)
    
    pixels[10] = BLUE
    pixels.show()
    time.sleep(1)
    
    pixels[11] = PURPLE
    pixels.show()
    time.sleep(1)
    
    pixels[12] = RED
    pixels.show()     #required to update pixels
    time.sleep(1)
    
    pixels[13] = YELLOW
    pixels.show()
    time.sleep(1)
    
    pixels[14] = GREEN
    pixels.show()
    time.sleep(1)
    
    pixels[15] = CYAN
    pixels.show()
    time.sleep(1)
    
    pixels[16] = BLUE
    pixels.show()
    time.sleep(1)
   