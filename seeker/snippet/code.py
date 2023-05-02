#date: 2023-05-02T16:55:51Z
#url: https://api.github.com/gists/317348a392a84c1670d65d96013b33e7
#owner: https://api.github.com/users/Jgillelen

import board
import neopixel
 
pixel_pin = board.D2
num_pixels = 16
 
pixels = neopixel.NeoPixel(pixel_pin, num_pixels, brightness=0.7, auto_write=False)

while True:
    for num in range(0,254,2):
        COLOR=(num,0,0)
        pixels[5]=COLOR
        pixels.show()
        COLOR=(num,0,0)
        pixels[6]=COLOR
        pixels.show()
        COLOR=(num,0,0)
        pixels[7]=COLOR
        pixels.show()
        COLOR=(num,0,0)
        pixels[8]=COLOR
        pixels.show()
    for num in range(254,0,-2):   
        COLOR=(num,0,0)
        pixels[8]=COLOR
        pixels.show()
        COLOR=(num,0,0)
        pixels[7]=COLOR
        pixels.show()
        COLOR=(num,0,0)
        pixels[6]=COLOR
        pixels.show()
        COLOR=(num,0,0)
        pixels[5]=COLOR