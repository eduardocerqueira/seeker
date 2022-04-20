#date: 2022-04-20T17:05:31Z
#url: https://api.github.com/gists/f980d5793bd5977cb636eb7e07eecd48
#owner: https://api.github.com/users/eva968

import board
import neopixel
import time
 
pixel_pin = board.D2   #the ring data is connected to this pin
num_pixels = 12        #number of leds pixels on the ring
 
pixels = neopixel.NeoPixel(pixel_pin, num_pixels, brightness=0.3, auto_write=False)
 
RED = (255, 0, 0) #RGB
YELLOW = (255, 150, 0)
GREEN = (0, 255, 0)
CYAN = (0, 255, 255)
BLUE = (0, 0, 255)
PURPLE = (180, 0, 255)
SALMON = (235, 106, 014)
OLIVE = (66, 70, 50)
AZUREBLUE = (2, 86, 105)
SIGNALVIOLET = (144,70,132)
ZINCYELLOW = (248,243,53)
STRAWBERRYRED = (213,48,50)
WATERBLUE = (37, 109, 123)
ROSE = (230, 50, 68)
CHROMEGREEN = (46, 58, 35)
PASTELVIOLET = (164, 125, 144)
ANTHRACITEGRAY = (41,49,51)
OFF = (0,0,0)

pixels[0] = SALMON
pixels.show()     #required to update pixels
time.sleep(1)

pixels[1] = OLIVE
pixels.show()
time.sleep(1)

pixels[2] = AZUREBLUE
pixels.show()
time.sleep(1)

pixels[3] = CYAN
pixels.show()
time.sleep(1)

pixels[4] = SIGNALVIOLET
pixels.show()
time.sleep(1)

pixels[5] = ZINCYELLOW
pixels.show()
time.sleep(1)

    
pixels[6] = STRAWBERRYRED
pixels.show()
time.sleep(1)

pixels[7] = WATERBLUE
pixels.show()
time.sleep(1)

pixels[8] = ROSE
pixels.show()
time.sleep(1)

pixels[9] = CHROMEGREEN
pixels.show()
time.sleep(1)

pixels[10] = PASTELVIOLET
pixels.show()
time.sleep(1)

pixels[11] = ANTHRACITEGRAY
pixels.show()
time.sleep(1)

pixels.fill(WATERBLUE)
pixels.show()
time.sleep(3)

pixels.fill(ROSE)
pixels.show()
time.sleep(1)
