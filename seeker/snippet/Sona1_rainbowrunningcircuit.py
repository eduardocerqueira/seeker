#date: 2023-05-01T16:56:46Z
#url: https://api.github.com/gists/49e98efdfc922ec29b9f17870f56341a
#owner: https://api.github.com/users/SonaKempner

import board
import neopixel
import time

pixel_pin = board.D2   #the ring data is connected to this pin
num_pixels = 16        #number of leds pixels on the ring

pixels = neopixel.NeoPixel(pixel_pin, num_pixels, brightness=0.2, auto_write=False)

RED = (255, 0, 0) #RGB
REDORANGE = (255, 85, 0)
YELLOW = (255, 157, 0)
LIME = (200, 255, 0)
GREEN = (21, 255, 0)
CYAN = (0, 255, 85)
BLUE = (0, 229, 255)
BLUE2 = (0, 0, 255)
PURPLE = (93, 0, 255)
PURPLE2 = (153, 0, 255)
MAGENTA = (234, 0, 255)
PINK = (255, 0, 212)
HOTPINK = (255, 0, 162)
ROSE = (255, 0, 93)
REDPINK = (255, 0, 43)
LASTONE = (255, 0, 21)
OFF = (0,0,0)


while True:
    for i in range(0,16,1):
        pixels[i] = RED
        pixels.show()     #required to update pixels
        time.sleep(0.1)

    for a in range(0,16,1):
        pixels[a] = REDORANGE
        pixels.show()
        time.sleep(0.1)

    for b in range(0,16,1):
        pixels[b] = YELLOW
        pixels.show()
        time.sleep(0.1)

    for c in range(0,16,1):
        pixels[c] = LIME
        pixels.show()
        time.sleep(0.1)

    for d in range(0,16,1):
        pixels[d] = GREEN
        pixels.show()
        time.sleep(0.1)

    for e in range(0,16,1):
        pixels[e] = CYAN
        pixels.show()
        time.sleep(0.1)
    
    for f in range(0,16,1):
        pixels[f] = BLUE
        pixels.show()
        time.sleep(0.1)

    for g in range(0,16,1):
        pixels[g] = BLUE2
        pixels.show()
        time.sleep(0.1)

    for h in range(0,16,1):
        pixels[h] = PURPLE
        pixels.show()
        time.sleep(0.1)

    for j in range(0,16,1):
        pixels[j] = PURPLE2
        pixels.show()
        time.sleep(0.1)

    for k in range(0,16,1):
        pixels[k] = MAGENTA
        pixels.show()
        time.sleep(0.1)
 
    for l in range(0,16,1): 
        pixels[l] = PINK
        pixels.show()
        time.sleep(0.1)
   
    for m in range(0,16,1): 
        pixels[m] = HOTPINK
        pixels.show()
        time.sleep(0.1)
    
    for n in range(0,16,1):
        pixels[n] = ROSE
        pixels.show()
        time.sleep(0.1)
   
    for o in range(0,16,1): 
        pixels[o] = REDPINK
        pixels.show()
        time.sleep(0.1)
    
    for p in range(0,16,1):
        pixels[p] = LASTONE
        pixels.show()
        time.sleep(0.1)
