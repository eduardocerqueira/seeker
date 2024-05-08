#date: 2024-05-08T16:57:37Z
#url: https://api.github.com/gists/6c3bea30cdf00d37ee5c9dc6188a15dd
#owner: https://api.github.com/users/YumpyMan

import time
import board
import neopixel
from digitalio import DigitalInOut, Direction, Pull
import touchio


touch_pad = board.A0 # the ~1 pin
touch = touchio.TouchIn(touch_pad)

pixel_pin = board.D2
num_pixels = 16

pixels = neopixel.NeoPixel(pixel_pin, num_pixels, brightness=0.3, auto_write=False)

RED = (255, 0, 0)
ORANGE = (255, 60, 0)
YELLOW = (255, 150, 0)
GREEN = (0, 255, 0)
CYAN = (0, 255, 255)
BLUE = (0, 0, 255)
PURPLE = (180, 0, 255)
WHITE = (255,255,255)
OFF = (0,0,0)

colors = [RED,YELLOW,GREEN,CYAN,BLUE,PURPLE,WHITE,OFF]
now=0

while True:
    if touch.value:
        now=now+1
        if (now >= len(colors)): #check to see if we exceed our list of colors
            now=0
        pixels[0]= RED
        pixels.show()
        time.sleep(0.1)  # debounce delay
        pixels[1]= RED
        pixels.show()
        time.sleep(0.1)
        pixels[2]= RED
        pixels.show()
        time.sleep(0.1)  # debounce delay
        pixels[3]= RED
        pixels.show()
        time.sleep(0.1)
        pixels[4]= RED
        pixels.show()
        time.sleep(0.1)  # debounce delay
        pixels[5]= RED
        pixels.show()
        time.sleep(0.1)
        pixels[6]= RED
        pixels.show()
        time.sleep(0.1)  # debounce delay
        pixels[7]= RED
        pixels.show()
        time.sleep(0.1)
        pixels[8]= RED
        pixels.show()
        time.sleep(0.1)  # debounce delay
        pixels[9]= RED
        pixels.show()
        time.sleep(0.1)
        pixels[10]= RED
        pixels.show()
        time.sleep(0.1)  # debounce delay
        pixels[11]= RED
        pixels.show()
        time.sleep(0.1)
        pixels[12]= RED
        pixels.show()
        time.sleep(0.1)  # debounce delay
        pixels[13]= RED
        pixels.show()
        time.sleep(0.1)
        pixels[14]= RED
        pixels.show()
        time.sleep(0.1)  # debounce delay
        pixels[15]= RED
        pixels.show()
        time.sleep(0.1)
        pixels[0]= ORANGE
        pixels.show()
        time.sleep(0.1)  # debounce delay
        pixels[1]= ORANGE
        pixels.show()
        time.sleep(0.1)
        pixels[2]= ORANGE
        pixels.show()
        time.sleep(0.1)  # debounce delay
        pixels[3]= ORANGE
        pixels.show()
        time.sleep(0.1)
        pixels[4]= ORANGE
        pixels.show()
        time.sleep(0.1)  # debounce delay
        pixels[5]= ORANGE
        pixels.show()
        time.sleep(0.1)
        pixels[6]= ORANGE
        pixels.show()
        time.sleep(0.1)  # debounce delay
        pixels[7]= ORANGE
        pixels.show()
        time.sleep(0.1)
        pixels[8]= ORANGE
        pixels.show()
        time.sleep(0.1)  # debounce delay
        pixels[9]= ORANGE
        pixels.show()
        time.sleep(0.1)
        pixels[10]= ORANGE
        pixels.show()
        time.sleep(0.1)  # debounce delay
        pixels[11]= ORANGE
        pixels.show()
        time.sleep(0.1)
        pixels[12]= ORANGE
        pixels.show()
        time.sleep(0.1)  # debounce delay
        pixels[13]= ORANGE
        pixels.show()
        time.sleep(0.1)
        pixels[14]= ORANGE
        pixels.show()
        time.sleep(0.1)  # debounce delay
        pixels[15]= ORANGE
        pixels.show()
        time.sleep(0.1)
        pixels[0]= YELLOW
        pixels.show()
        time.sleep(0.1)  # debounce delay
        pixels[1]= YELLOW
        pixels.show()
        time.sleep(0.1)
        pixels[2]= YELLOW
        pixels.show()
        time.sleep(0.1)  # debounce delay
        pixels[3]= YELLOW
        pixels.show()
        time.sleep(0.1)
        pixels[4]= YELLOW
        pixels.show()
        time.sleep(0.1)  # debounce delay
        pixels[5]= YELLOW
        pixels.show()
        time.sleep(0.1)
        pixels[6]= YELLOW
        pixels.show()
        time.sleep(0.1)  # debounce delay
        pixels[7]= YELLOW
        pixels.show()
        time.sleep(0.1)
        pixels[8]= YELLOW
        pixels.show()
        time.sleep(0.1)  # debounce delay
        pixels[9]= YELLOW
        pixels.show()
        time.sleep(0.1)
        pixels[10]= YELLOW
        pixels.show()
        time.sleep(0.1)  # debounce delay
        pixels[11]= YELLOW
        pixels.show()
        time.sleep(0.1)
        pixels[12]= YELLOW
        pixels.show()
        time.sleep(0.1)  # debounce delay
        pixels[13]= YELLOW
        pixels.show()
        time.sleep(0.1)
        pixels[14]= YELLOW
        pixels.show()
        time.sleep(0.1)  # debounce delay
        pixels[15]= YELLOW
        pixels.show()
        time.sleep(0.1)
        
       