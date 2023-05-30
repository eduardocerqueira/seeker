#date: 2023-05-30T16:56:33Z
#url: https://api.github.com/gists/5aa1a5f52702a21bcfe09aa388ee9e2d
#owner: https://api.github.com/users/isaburke23

import time
import board
import neopixel

pixel_pin = board.D2
num_pixels = 16

pixels = neopixel.NeoPixel(pixel_pin, num_pixels, brightness=0.7, auto_write=False)

while True:
    for num in range(50,254,2):   #fade in loop
        COLOR=(0,0,num)
        for light in range (0,16,1):
            pixels[light]=COLOR
        pixels.show()
    for num in range(254,50,-2):   #fade out loop
        COLOR=(0,0,num)
        for light in range (0,16,1):
            pixels[light]=COLOR
        pixels.show()# Write your code here :-)


