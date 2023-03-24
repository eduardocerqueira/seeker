#date: 2023-03-24T16:53:53Z
#url: https://api.github.com/gists/be4ec904181c618b1a51f3f4fbc737fd
#owner: https://api.github.com/users/CerysHC

# Write your code here :-)
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
 
PINK = (255, 71, 185)
RED = (255,0,0)
ORANGE = (255, 142, 71)
YELLOW = (255,210, 0)
GREEN = (0, 180, 0)
TEAL = (0, 255, 255)
TURQUUOISE = (0, 211, 205)
PERIWINKLE = (100,100,255)
BLUE = (0, 0, 255)
PURPLE = (150, 0, 255)
MAGENTA = (211, 0, 127)
WHITE = (255, 255, 255)
OFF = (0,0,0) 

colors = [MAGENTA,RED,ORANGE,YELLOW,GREEN,TEAL,TURQUUOISE,BLUE,PERIWINKLE,PURPLE,PINK,WHITE, OFF]
now=0

while True:
    print(now)
    if (switch.value==False):    #detect the button press
        now=now+1 
        if (now >= 14): #7 colors in the list
            now=0
    if (now <=12):
        pixels.fill(colors[now])
        pixels.show()
    if (now==13):
        while True:
            pixels.fill(MAGENTA)
            pixels.show()     #required to update pixels
            time.sleep(3)
            pixels.fill(RED)
            pixels.show()
            time.sleep(3)
            pixels.fill(ORANGE)
            pixels.show()
            time.sleep(3)
            pixels.fill(YELLOW)
            pixels.show()
            time.sleep(3)
            pixels.fill(GREEN)
            pixels.show()
            time.sleep(3)
            pixels.fill(TEAL)
            pixels.show()
            time.sleep(3)
            pixels.fill(TURQUUOISE)
            pixels.show()
            time.sleep(3)
            pixels.fill(BLUE)
            pixels.show()
            time.sleep(3)
            pixels.fill(PERIWINKLE)
            pixels.show()
            time.sleep(3)
            pixels.fill(PURPLE)
            pixels.show()
            time.sleep(3)
            pixels.fill(PINK)
            pixels.show()
            time.sleep(3)
            pixels.fill(WHITE)
            pixels.show()
            time.sleep(3)
    time.sleep(0.12)  # debounce delay