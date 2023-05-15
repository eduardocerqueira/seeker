#date: 2023-05-15T17:00:46Z
#url: https://api.github.com/gists/19fdc09d1b3c84ae061313ada66c4d10
#owner: https://api.github.com/users/allisonxcohn

import time
import board
import neopixel
from digitalio import DigitalInOut, Direction, Pull

switch = DigitalInOut(board.D3)
switch.direction = Direction.INPUT
switch.pull = Pull.UP

pixel_pin = board.D2
num_pixels = 16

pixels = neopixel.NeoPixel(pixel_pin, num_pixels, brightness=0.3, auto_write=False)

RED = (255, 0, 0)
YELLOW = (255, 150, 0)
GREEN = (0, 255, 0)
CYAN = (0, 255, 255)
BLUE = (0, 0, 255)
PURPLE = (180, 0, 255)
WHITE = (255, 255, 255)
OFF = (0, 0, 0)

COLORS = [RED, YELLOW, GREEN, CYAN, BLUE, PURPLE]
on = False

while True:
    print(switch.value)
    if switch.value == False:
        while switch.value == False:
            time.sleep(0.01)

        if on == True:
            pixels.fill(OFF)
            pixels.show()
            on = False

        else:
            pixels.fill(WHITE)
            pixels.show()
            on = True

        for i in range(0, 500, 1):
            if switch.value == False:
                for x in range(0, len(COLORS), 1):
                    pixels.fill(COLORS[x])
                    pixels.show()
                    time.sleep(0.15)
                    on = True

    # Write your code here :-)
