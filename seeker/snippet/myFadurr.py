#date: 2023-05-01T16:50:23Z
#url: https://api.github.com/gists/be438ba96bccc77c9f8b23b33d5d0863
#owner: https://api.github.com/users/Maahirgg

import time
import board
import neopixel
 
pixel_pin = board.D2
num_pixels = 16
 
pixels = neopixel.NeoPixel(pixel_pin, num_pixels, brightness=0.7, auto_write=False)

while True:
    for num in range(0,254,2):   #fade in loop
        COLOR=(num,42,254-num)
        for light in range (0,16,1): 
            pixels[light]=COLOR
        pixels.show()
    for num in range(254,0,-2):   #fade out loop
        COLOR=(num,42,254-num)
        for light in range (0,16,1): 
            pixels[light]=COLOR
        pixels.show()