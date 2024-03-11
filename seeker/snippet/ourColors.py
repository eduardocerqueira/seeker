#date: 2024-03-11T17:02:00Z
#url: https://api.github.com/gists/2b5f4a1d30ba8f3fc4d8dec27ec4a5e8
#owner: https://api.github.com/users/WilliamsI2009

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
MATEO = (183, 250, 167) 
GABBY = (70, 32, 74)
AMY = (134, 102, 209) 
MSBROOKS = (237, 122, 255) 
MARI = (122, 138, 255)
IMAN = (247, 116, 101)
TAYLORSWIFT = (245, 0, 288)
 
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
    
    pixels[6] = MATEO 
    pixels.show()
    time.sleep(1)
    
    pixels[7] = GABBY 
    pixels.show()
    time.sleep(1)
    
    pixels[8] = AMY
    pixels.show()
    time.sleep(1)
    
    pixels[9] = MSBROOKS
    pixels.show()
    time.sleep(1)
    
    pixels[10] = MARI
    pixels.show()
    time.sleep(1)
    
    pixels[11] = IMAN
    pixels.show()
    time.sleep(1)
    
    pixels[12] = TAYLORSWIFT
    pixels.show()
    time.sleep(1)