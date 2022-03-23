#date: 2022-03-23T17:08:10Z
#url: https://api.github.com/gists/ce9c5f9fa6b1d68c217aafc3cf98538c
#owner: https://api.github.com/users/camxk

import time
import board
import neopixel
import math
import analogio
from digitalio import DigitalInOut, Direction, Pull

switch = DigitalInOut(board.D3)
switch.direction = Direction.INPUT
switch.pull = Pull.UP

dial_pin = board.A2  # pin 0 is Analog input 2
dial = analogio.AnalogIn(dial_pin)

pixel_pin = board.D2
num_pixels = 12

pixels = neopixel.NeoPixel(pixel_pin, num_pixels,
                           brightness=0.2, auto_write=False)

now = 0

while True:
    if (switch.value == 0):  # detect the button press, 0 means pressed
        now = now+1
        if (now >= len(colors)):  # check to see if we exceed our list of colors
            now = 0
        time.sleep(0.2)  # debounce delay
    # scale down potentiometer values to fit within color range
    val = int(math.sqrt(dial.value)-20)
    print(val)


    WHITE = (181, 165, 65)
    OFF = (0, 0, 0)

    colors = [OFF, WHITE]

    dialcolor = colors[now]
    print(dialcolor[0], dialcolor[1], dialcolor[2])

    pixels.fill(dialcolor)
    pixels.show()
    time.sleep(0.1)
