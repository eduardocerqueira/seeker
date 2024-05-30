#date: 2024-05-30T17:00:10Z
#url: https://api.github.com/gists/67cc731f73254437e0d29bc1e178c1ff
#owner: https://api.github.com/users/Zachstrait

# Write your code here :-)
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

myCOLORS = [RED, YELLOW, GREEN, CYAN, BLUE, PURPLE, WHITE, OFF]

while True:  
    for color in myCOLORS:
        pixels.fill(color)
        pixels.show()
        time.sleep(1)
