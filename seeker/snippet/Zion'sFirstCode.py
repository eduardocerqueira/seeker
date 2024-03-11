#date: 2024-03-11T16:54:33Z
#url: https://api.github.com/gists/b5fef2d7ed6a3e0f51bac97d13568134
#owner: https://api.github.com/users/zion1g

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
DARKPURPLE = (150, 0, 150)
BROWN = (150, 60, 0)
ORANGE = (250, 150, 0)
VIOLET = (170, 0, 100)
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
    
    pixels[6] = DARKPURPLE
    pixels.show()
    time.sleep(1)
    
    pixels[7] = VIOLET
    pixels.show()
    time.sleep(1)
    
    pixels[8] = BROWN
    pixels.show()
    time.sleep(1)
    
    pixels[9] = ORANGE
    pixels.show()
    time.sleep(1)
    
    pixels[10] = RED
    pixels.show()
    time.sleep(1)
    
    pixels[11] = YELLOW
    pixels.show()
    time.sleep(1)
    
    pixels[12] = GREEN
    pixels.show()
    time.sleep(1)
    
    pixels[13] = CYAN
    pixels.show()
    time.sleep(1)
    
    pixels[14] = BLUE
    pixels.show()
    time.sleep(1)
    
    pixels[15] = PURPLE
    pixels.show()
    time.sleep(1)
    
    pixels.fill(VIOLET)
    pixels.show()
    time.sleep(1)