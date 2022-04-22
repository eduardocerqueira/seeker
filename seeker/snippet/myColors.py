#date: 2022-04-22T17:04:58Z
#url: https://api.github.com/gists/ebbc07a8890ee20d20358529da5f0552
#owner: https://api.github.com/users/sammynaser07

import time
import board
import neopixel
 
pixel_pin = board.D2
num_pixels = 12
 
pixels = neopixel.NeoPixel(pixel_pin, num_pixels, brightness=0.5, auto_write=False)
RED = (255,0,0)
BLUE= (0,0,255)
GREEN=(0,255,0)
OFF = (0,0,0)

while True:
    for i in range(0,12,1):
        pixels[i]=RED
        pixels.show()
        time.sleep(.02)
        pixels[i]=OFF
        pixels.show()
    for k in range(0,12,1):
        pixels[k]=BLUE
        pixels.show()
        time.sleep(.02)
        pixels[k]=OFF
        pixels.show()
    for b in range(0,12,1):
        pixels[b]=GREEN
        pixels.show()
        time.sleep(.02)
        pixels[b]=OFF
        pixels.show()

