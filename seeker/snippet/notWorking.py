#date: 2026-02-10T17:47:08Z
#url: https://api.github.com/gists/d2e95f762fb20ffbd7fd3d8c652bb056
#owner: https://api.github.com/users/29adelahonigberg-bit

import time 
import board 
import neopixel

pixel_pin = board.D2
num_pixels = 16

pixels = neopixel.NeoPixel(pixel_pin, num_pixels, brightness=0.3, auto_write=False)

RED = (151, 230, 189) 
YELLOW = (157, 200, 237) 
GREEN = (151, 230, 189) 
CYAN = (131, 222, 181) 
BLUE = (151, 223, 230) 
PURPLE = (112, 219, 214) 
WHITE = (151, 223, 230) 
OFF = (137, 179, 217) 

colors = [RED, YELLOW, GREEN, CYAN, BLUE, PURPLE, WHITE, OFF]


# Main loop
while True:
    time.sleep(2.0)
    if (photocell.value <= 300): # this is the light threshold
        for color in colors:
        print(color)
        for i in range(num_pixels):
            pixels[i] = color
            pixels.show()
            time.sleep(.05)
        # Optional: clear after each chase
        pixels.show()
    else:
        pixels[0] = OFF
        pixels.show()