#date: 2023-03-24T16:55:56Z
#url: https://api.github.com/gists/0f165522eb9f8af5c6f23b69ad290345
#owner: https://api.github.com/users/matth3w-yan

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
YELLOW = (255, 150, 0)
GREEN = (0, 255, 0)
CYAN = (0, 255, 255)
BLUE = (0, 0, 255)
PURPLE = (180, 0, 255)
WHITE = (255,255,255)
Logan = (34,224,224)
Purple2 = (63,16,120)
OFF = (0,0,0)

colors = [YELLOW,RED,GREEN,CYAN,OFF]
now=0

while True:
    if touch.value:
        now=now+1
        print(now)
        if (now < 5): #check to see if we exceed our list of colors
            pixels.fill(colors[now])
            pixels.show()
        if (now == 5):
            pixels[0] = RED
            pixels.show()     #required to update pixels
    
            pixels[1] = YELLOW
            pixels.show()
    
            pixels[2] = GREEN
            pixels.show()
    
            pixels[3] = CYAN
            pixels.show()
    
            pixels[4] = BLUE
            pixels.show()
    
            pixels[5] = PURPLE
            pixels.show()
                        
            pixels[6] = RED
            pixels.show()     
    
            pixels[7] = YELLOW
            pixels.show()
    
            pixels[8] = GREEN
            pixels.show()
    
            pixels[9] = CYAN
            pixels.show()
    
            pixels[10] = BLUE
            pixels.show()
    
            pixels[11] = PURPLE
            pixels.show()
            
            pixels[12] = RED
            pixels.show()     
    
            pixels[13] = YELLOW
            pixels.show()
    
            pixels[14] = GREEN
            pixels.show()
    
            pixels[15] = CYAN
            pixels.show()
            
            if touch.value:
                nw = now+1
        if(now>5):
            now=0
    time.sleep(0.2)