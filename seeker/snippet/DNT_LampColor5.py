#date: 2023-03-24T16:56:19Z
#url: https://api.github.com/gists/a6e3903bb80dd0e8a91ab7b8109a31b3
#owner: https://api.github.com/users/smv307

import time
import board
import neopixel
import touchio

touch_pad = board.A0
touch = touchio.TouchIn(touch_pad)

pixel_pin = board.D2
num_pixels = 16

pixels = neopixel.NeoPixel(pixel_pin, num_pixels, brightness=0.3, auto_write=False)

darkOrange = (69, 38, 1)
orange = (130, 73, 9)
lightOrange = (227, 138, 39)
white = (237, 170, 95)
off = (0, 0, 0)
colors = [darkOrange, orange, lightOrange, orange]
   
touch_count = 0

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
        time.sleep(0.01)

while True:

# add touch value
    if touch.value:
        touch_count += 1
        print(touch_count)
    
# off
    if touch_count == 0:
        pixels.fill(off)
        pixels.show()
        time.sleep(0.5)

# Multicolor
    elif touch_count == 1:
        print("multicolor")
        for i in range(num_pixels):
            pixels[i] = colors[i % len(colors)]
            time.sleep(0.1)
            pixels.show()

# Fade 
    elif touch_count == 2:
        fade(orange,lightOrange)
        fade(lightOrange,orange)
        time.sleep(0.5)
    
# Reset
    elif touch_count > 2:
        touch_count = 0
        print("off")
        