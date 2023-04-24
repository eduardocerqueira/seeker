#date: 2023-04-24T16:58:25Z
#url: https://api.github.com/gists/b23ec6a325acef1f8eb8a348187d04e3
#owner: https://api.github.com/users/asheepo

import time
import board
import neopixel

pixel_pin = board.D2   #the ring data is connected to this pin
num_pixels = 16        #number of leds pixels on the ring

pixels = neopixel.NeoPixel(pixel_pin, num_pixels, brightness=0.3, auto_write=False)

RED = (255, 0, 0) #RGB
YELLOW = (255, 150, 0)
GREEN = (0, 255, 0)
CYAN = (0, 255, 255)
BLUE = (0, 0, 255)
PURPLE = (180, 0, 255)
WHITE = (255,255,255)
OFF = (0,0,0)
LIGHTNING = (255,255,0)

while True:

    pixels[0] = LIGHTNING
    pixels.show()     #required to update pixels
    time.sleep(.2)

    pixels[1] = OFF
    pixels.show()
    time.sleep(.2)

    pixels[2] = LIGHTNING
    pixels.show()
    time.sleep(.2)

    pixels[3] = LIGHTNING
    pixels.show()
    time.sleep(0.2)

    pixels[4] = OFF
    pixels.show()
    time.sleep(0.2)

    pixels[5] = LIGHTNING
    pixels.show()
    time.sleep(0.2)

    pixels[0] = LIGHTNING
    pixels.show()     #required to update pixels
    time.sleep(.2)

    pixels[6] = OFF
    pixels.show()
    time.sleep(.2)

    pixels[7] = LIGHTNING
    pixels.show()
    time.sleep(.2)

    pixels[8] = LIGHTNING
    pixels.show()
    time.sleep(0.2)

    pixels[9] = OFF
    pixels.show()
    time.sleep(0.2)

    pixels[10] = LIGHTNING
    pixels.show()
    time.sleep(0.2)

    pixels[11] = LIGHTNING
    pixels.show()
    time.sleep(0.2)

    pixels[12] = LIGHTNING
    pixels.show()     #required to update pixels
    time.sleep(.2)

    pixels[13] = OFF
    pixels.show()
    time.sleep(.2)

    pixels[14] = LIGHTNING
    pixels.show()
    time.sleep(.2)

    pixels[15] = LIGHTNING
    pixels.show()
    time.sleep(0.2)