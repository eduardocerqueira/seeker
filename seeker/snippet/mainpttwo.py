#date: 2023-03-24T16:54:42Z
#url: https://api.github.com/gists/f0a19480b9fe786969c098f445696964
#owner: https://api.github.com/users/Arbo2721

import time
import board
import neopixel
from digitalio import DigitalInOut, Direction, Pull

switch = DigitalInOut(board.D3)
switch.direction = Direction.INPUT
switch.pull = Pull.UP

pixel_pin = board.D2
num_pixels = 16
 
pixels = neopixel.NeoPixel(pixel_pin, num_pixels, brightness=0.7, auto_write=False)

press_count=0

off=(0,0,0)
red=(255,0,0)
yellow=(255,255,0)

def press_check():
    global press_count
    if (switch.value==False):
        press_count+=1
        if press_count>2:
            press_count=0
        time.sleep(0.3)
    print(press_count)

WHITE = (255,255,255)

while True:
    
    press_check()
    if press_count==0:
        pixels.fill(off)
        pixels.show()
        
    if press_count==1:
        for num in range(0,255,5):   #fade in loop
            COLOR=(num,num,num)
            if (switch.value==False):
                press_check()
            for i in range(0,16,1):
                pixels[i]=COLOR
                pixels.show()
        pixels.fill(WHITE)
        pixels.show()

    if press_count==2:
        pixels.fill(off)
        pixels.show()