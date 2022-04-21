#date: 2022-04-21T17:05:07Z
#url: https://api.github.com/gists/d0a302c28dec0582e60a5f7930f53f03
#owner: https://api.github.com/users/saskialee

# Write your code here :-)
# Write your code here :-)
import time
import board
import neopixel

pixel_pin = board.D2   #the ring data is connected to this pin
num_pixels = 12        #number of leds pixels on the ring

pixels = neopixel.NeoPixel(pixel_pin, num_pixels, brightness=0.3, auto_write=False)

RED = (255, 0, 0) #RGB
YELLOW = (255, 150, 0)
GREEN = (0, 255, 0)
CYAN = (0, 255, 255)
BLUE = (0, 0, 255)
PURPLE = (180, 0, 255)
WHITE = (255,255,255)
LIGHT_GREEN = (144, 238, 144)
OFF = (0,0,0)

while True:
    pixels.fill(CYAN)
    pixels.show()     #required to update pixels
    time.sleep(2)
    pixels.fill(PURPLE)
    pixels.show()
    time.sleep(2)
    pixels.fill(LIGHT_GREEN)
    pixels.show()
    time.sleep(2)
    pixels.fill(WHITE)
    pixels.show()
    time.sleep(2)
