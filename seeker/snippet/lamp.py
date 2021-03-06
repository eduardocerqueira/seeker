#date: 2022-03-23T17:05:06Z
#url: https://api.github.com/gists/499d312832a151b9d07d506b134b1ba5
#owner: https://api.github.com/users/jdimayuga

import time
import board
import neopixel
from digitalio import DigitalInOut, Direction, Pull

pixel_pin = board.D2
num_pixels = 12
 
pixels = neopixel.NeoPixel(pixel_pin, num_pixels, brightness=0.3, auto_write=False)
 
TEAL = (3, 252, 223)
SKY = (3, 194, 252)
DARKSKY = (3, 140, 252)
CYAN = (0, 255, 255)
BLUE = (0, 0, 255)
SKYY = (0,182,242)
LIGHTSKY = (55,180,222)
TEALL = (8, 156, 158)
DARKBLUE = (22,127,247)

# helper function for fading - do not touch
def fade(c1, c2):
    diff = (c2[0]-c1[0], c2[1]-c1[1], c2[2]-c1[2])
    diffA = [1 if n==0 else abs(n) for n in diff]
    maxDiff = max(diffA)
    index = diffA.index(maxDiff)
    cFade = list(c1)
    increment = [int(diff[i]/diffA[i]) for i in range(3)]
    for i in range(0, maxDiff):
        for n in range(3):
            if(cFade[n] != c2[n]):
                cFade[n] += increment[n]
        pixels.fill(tuple(cFade))
        pixels.show()
        time.sleep(0.01)  # debounce delay
#end helper function

while True:
    fade(BLUE, SKY)
    fade(SKY, TEAL)
    fade(TEAL, CYAN)
    fade(CYAN, SKYY)
    fade(SKYY, LIGHTSKY)
    fade(LIGHTSKY, TEALL)
    fade(TEALL, DARKBLUE)
    fade(DARKBLUE, DARKSKY)
    fade(DARKSKY, BLUE)

    time.sleep(0.5)  # debounce delay# Write your code here :-)
# Write your code here :-)
# Write your code here :-)
# Write your code here :-)
# Write your code here :-)
# Write your code here :-)
# Write your code here :-)
# Write your code here :-)
# Write your code here :-)
