#date: 2022-04-22T17:05:07Z
#url: https://api.github.com/gists/05e919c7f8a49c004ad36d60d2e6376e
#owner: https://api.github.com/users/Michael-Andy-Moats

import time
import board
import neopixel
 
pixel_pin = board.D2
num_pixels = 12
 
pixels = neopixel.NeoPixel(pixel_pin, num_pixels, brightness=0.7, auto_write=False)

while True:
    for num in range(0,200,1):   #fade in loop
        COLOR=(num,0,num)
        for light in range (0,6,1): 
            pixels[light]=COLOR
        pixels.show()
    for num in range(200,0,-1):   #fade out loop
        COLOR=(num,0,num)
        for light in range (0,6,1): 
            pixels[light]=COLOR
        pixels.show()