#date: 2024-03-22T17:06:20Z
#url: https://api.github.com/gists/423dbb63eac0ccea6e16f415df6b939e
#owner: https://api.github.com/users/CarolineRuns

import time
import board
import neopixel
import random
from digitalio import DigitalInOut, Direction, Pull



pixel_pin = board.D2  # the ring data is connected to this pin
num_pixels = 16  # number of leds pixels on the ring

switch = DigitalInOut(board.D3)
switch.direction = Direction.INPUT
switch.pull = Pull.UP

pixels = neopixel.NeoPixel(pixel_pin, num_pixels, brightness=0.3, auto_write=False)



LIGHT_BLUE = (153, 214, 247)
LIGHT_ICE_BLUE = (227, 253, 255)
CYAN = (0, 255, 255)
BLUE = (0, 0, 255)
DARK_ICE_BLUE = (73, 232, 255)
DARK_BLUE = (19, 5, 150)
C030BFC = (3, 11, 252)
C050047 = (5, 0, 71)
C1200FF = (18, 0, 255)
C6F94B0 = (111,148,176)
CC06D6E4 = (192,214,228)
C22486D	= (34,72,109)
C5B7A92	= (91,122,146)
CDBE4EC = (219,228,236)
WHITE = (255, 255, 255)
OFF = (0, 0, 0)

myCOLORS = [LIGHT_BLUE, DARK_ICE_BLUE, LIGHT_ICE_BLUE, C030BFC, CYAN, BLUE, DARK_BLUE, WHITE, C050047, C1200FF, C6F94B0, CC06D6E4, C22486D, C5B7A92, CDBE4EC,]

def handleMode(mode):
    if (mode == False):
        animateOff()
    elif (mode == True):
        animateOn()

def animateOn():
    pixels.fill(LIGHT_BLUE)
    pixels.show()
    time.sleep(0.1)

def animateOff():
    pixels.fill(OFF)
    pixels.show()
    time.sleep(0.1)

while True:
    print(switch.value)
    if (switch.value == True):
        currentSwitchValue = switch.value
        if (currentSwitchValue == False and lastSwitchValue == True):
            pixels.fill(LIGHT_BLUE)
        else:
            led_choice = random.randint(0, 15)
            color_choice = random.choice(myCOLORS)
            pixels[led_choice] = color_choice
            pixels.show()
            time.sleep(0.2)
        lastSwitchValue = currentSwitchValue
        time.sleep(0.1)
    else:
        pixels.fill(OFF)

